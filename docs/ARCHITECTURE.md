# Architecture

## 1. System overview

A fully local Retrieval-Augmented Generation (RAG) system with an
automated evaluation layer. Three independent stages — ingestion,
retrieval+generation, and evaluation — share a common local vector
store and a common local LLM runtime (Ollama). No network calls leave
the machine at any point.

```mermaid
graph TB
    subgraph Local Machine
        subgraph Ollama Runtime
            EMB[nomic-embed-text<br/>embedding model]
            GEN[llama3.2<br/>generation model]
        end
        subgraph Storage
            FS[data/ *.txt files]
            DB[(chroma_db/<br/>persistent vector store)]
        end
        subgraph Application Code
            ING[ingest.py]
            RET[retrieve.py]
            EVL[evals/test_rag.py]
        end
    end

    FS --> ING
    ING --> EMB
    ING --> DB
    RET --> EMB
    RET --> DB
    RET --> GEN
    EVL --> RET
    EVL --> GEN
```

## 2. Component responsibilities

| Component | File | Responsibility | Depends on |
|---|---|---|---|
| Ingestion | `src/ragbot/ingest.py` | Turn raw `.txt` files into searchable vectors | Chroma, Ollama (embedding) |
| Retrieval + Generation | `src/ragbot/retrieve.py` | Answer a question grounded in stored documents | Chroma, Ollama (embedding + chat) |
| Agent (planned) | `src/ragbot/agent.py` | Tool-calling loop on top of retrieval | retrieve.py, Ollama |
| Evaluation | `evals/test_rag.py` | Automatically verify correctness and safety | retrieve.py, DeepEval, Ollama (as judge) |

## 3. Ingestion — detailed flow

```mermaid
sequenceDiagram
    participant U as ingest.py main()
    participant FS as data/ folder
    participant C as chunk_text()
    participant E as embed_text()
    participant O as Ollama (nomic-embed-text)
    participant DB as Chroma

    U->>FS: os.listdir(data_dir)
    FS-->>U: list of .txt filenames
    loop for each document
        U->>U: load_documents() reads file as UTF-8 string
        U->>C: chunk_text(text, chunk_size=500, overlap=50)
        C-->>U: list[str] of overlapping chunks
        loop for each chunk
            U->>E: embed_text(chunk)
            E->>O: ollama.embeddings(model, prompt=chunk)
            O-->>E: {"embedding": [float, float, ...]}
            E-->>U: embedding vector
            U->>DB: collection.add(ids, embeddings, documents, metadatas)
        end
    end
```

**Exact data shapes at each step:**
- `load_documents()` returns `list[dict]`, each `{"id": "<filename>", "text": "<full file contents as str>"}`
- `chunk_text(text, chunk_size=500, overlap=50)` returns `list[str]` — implemented as a sliding window: `start = 0`, slice `text[start:start+500]`, then `start += 450` (500 − 50 overlap), repeat until `start >= len(text)`
- `embed_text(chunk)` returns `list[float]` — a single embedding vector, dimensionality determined by `nomic-embed-text` (768 dimensions)
- Each Chroma `collection.add()` call stores exactly one chunk with:
  - `ids=["<filename>_<chunk_index>"]` — globally unique per chunk
  - `embeddings=[<768-float vector>]`
  - `documents=["<chunk text>"]` — the raw text, kept for later retrieval
  - `metadatas=[{"source": "<filename>"}]` — enables tracing an answer back to its source file

**Persistence:** `chromadb.PersistentClient(path="chroma_db")` writes to disk immediately on `.add()` — no explicit save/flush step needed, and the data survives between script runs.

## 4. Retrieval + Generation — detailed flow

```mermaid
sequenceDiagram
    participant User
    participant R as retrieve.py
    participant O1 as Ollama (nomic-embed-text)
    participant DB as Chroma
    participant O2 as Ollama (llama3.2)

    User->>R: question: str (via input())
    R->>O1: embed_query(question)
    O1-->>R: query_embedding: list[float]
    R->>DB: collection.query(query_embeddings=[query_embedding], n_results=3)
    DB-->>R: results["documents"][0]: list[str] (top-3 chunks)
    alt no chunks found
        R-->>User: "No relevant documents found."
    else chunks found
        R->>R: generate_answer(question, chunks)
        R->>R: build prompt string:<br/>context + instruction + question
        R->>O2: ollama.chat(model="llama3.2", messages=[{role:user, content:prompt}])
        O2-->>R: {"message": {"content": "<answer text>"}}
        R-->>User: print answer
    end
```

**The exact prompt template sent to the LLM:**
```
Answer the question using only the context below.
If the context doesn't contain the answer, say "I don't know based on the available documents."

Context:
{context}

Question: {question}
Answer:
```
where `{context}` is the retrieved chunks joined with `"\n\n"`, and `{question}` is the raw user input. This template is the single point of control for the hallucination guardrail — any change to grounding behavior happens here.

**Why `n_results=3` (top_k):** a tunable trade-off. Too few chunks risks missing the answer if it's split across chunks; too many dilutes the prompt with irrelevant text and increases the chance the model latches onto the wrong passage. 3 is a reasonable default for a small document set.

## 5. Evaluation — detailed flow

```mermaid
sequenceDiagram
    participant T as test_rag.py
    participant R as retrieve.py functions
    participant J as OllamaModel (judge, llama3.2)
    participant D as DeepEval framework

    T->>R: run_rag(question)
    R-->>T: (answer, chunks)
    T->>D: LLMTestCase(input, actual_output=answer, retrieval_context=chunks)
    T->>D: FaithfulnessMetric(threshold=0.7, model=J)
    T->>D: AnswerRelevancyMetric(threshold=0.7, model=J)
    D->>J: judge prompt: "does this answer contradict/invent facts vs context?"
    J-->>D: faithfulness score (0-1)
    D->>J: judge prompt: "does this answer address the question?"
    J-->>D: relevancy score (0-1)
    D->>D: assert_test — fails if either score < threshold
    D-->>T: pass / fail with reasoning
```

**Two distinct test cases, two distinct purposes:**
1. `test_sourdough_question` — positive case. Confirms the pipeline retrieves the *correct* chunk and produces a faithful, relevant answer when the documents genuinely contain the answer.
2. `test_unrelated_question_says_dont_know` — negative/safety case. Confirms the pipeline does *not* hallucinate when the documents don't contain the answer — this is a direct, automated test of the prompt guardrail described in section 4.

**Why the judge model matters:** `FaithfulnessMetric` and `AnswerRelevancyMetric` don't use simple string matching — they use an LLM to *reason* about whether the answer is supported by the context. `judge_model = OllamaModel(model="llama3.2", base_url="http://localhost:11434")` points that reasoning step at the same local model already running, so evaluation costs nothing and needs no external API key.

## 6. Configuration surface

| Constant | Location | Current value | What changing it affects |
|---|---|---|---|
| `CHUNK_SIZE` | `ingest.py` | 500 chars | Larger = fewer, broader chunks; smaller = more precise but more fragmented retrieval |
| `OVERLAP` | `ingest.py` | 50 chars | Reduces risk of splitting an idea across chunk boundaries |
| `EMBED_MODEL` | both files | `nomic-embed-text` | Must be identical in ingestion and retrieval, or vectors aren't comparable |
| `CHAT_MODEL` | `retrieve.py` | `llama3.2` | The generation model; swappable for any Ollama-compatible model |
| `top_k` | `retrieve_chunks()` | 3 | Number of chunks retrieved per query |
| `threshold` | `evals/test_rag.py` | 0.7 | Minimum acceptable faithfulness/relevancy score before a test fails |

## 7. Known limitations (explicit, not hidden)

- **Small corpus behavior:** with very few documents, Chroma still returns its "closest available" match even when nothing is truly relevant — retrieval quality depends on having a reasonably sized, topically diverse document set.
- **No prompt-injection defense yet:** a malicious instruction embedded inside a source document could currently influence the LLM's behavior. Planned for Phase 3.5.
- **No chunk deduplication:** re-running `ingest.py` on the same files adds duplicate entries rather than updating existing ones — acceptable for a learning project, not yet production-safe.
- **Single-turn only:** `retrieve.py` has no conversation memory; each question is independent.

## 8. Planned extensions (not yet built)

- **Agent loop (`agent.py`):** wrap `retrieve_chunks`/`generate_answer` as a callable tool inside a decide → act → observe loop, adding the ability to call external tools beyond retrieval.
- **MCP server:** expose one external API (still to be chosen) as a Model Context Protocol tool the agent can call.
- **Prompt-injection test suite:** adversarial documents in `evals/` designed to try to hijack the system prompt, with pass/fail assertions.