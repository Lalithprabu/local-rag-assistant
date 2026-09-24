# Local RAG Assistant

A fully local, zero-cost Retrieval-Augmented Generation (RAG) system with
an automated evaluation suite. Runs entirely on-device via Ollama — no
API keys, no external calls, no cost.

## What it does

Answers questions grounded in a set of source documents, rather than
relying on the language model's raw memory. If the answer isn't
supported by the retrieved documents, it says so instead of guessing —
this is enforced both by prompt design and by an automated test suite.

## Why this project

Most AI demos skip evaluation entirely. This one treats "does it
actually work reliably" as a first-class requirement, not an
afterthought — the `evals/` suite checks faithfulness (no invented
facts) and relevancy on every run, and includes a specific safety test
for unanswerable questions.

## Stack

- [Ollama](https://ollama.com) — runs `llama3.2` (generation) and
  `nomic-embed-text` (embeddings) locally
- [Chroma](https://www.trychroma.com/) — local vector database
- [DeepEval](https://github.com/confident-ai/deepeval) — automated LLM
  evaluation, judged by a local model
- Python 3.10

## Architecture

**Ingestion (one-time setup, `ingest.py`):**
1. Reads all `.txt` files from `data/`
2. Splits each into ~500-character overlapping chunks, so no single
   embedding has to represent an entire document
3. Embeds each chunk using `nomic-embed-text`
4. Stores chunk text, embedding, and source filename in a local Chroma
   database (`chroma_db/`)

**Query time (`retrieve.py`):**
1. Embeds the incoming question with the same embedding model, so the
   question and documents live in the same "meaning space"
2. Asks Chroma for the top-k most similar stored chunks
3. Builds a prompt including those chunks as context, explicitly
   instructing the model to answer only from them, and to say "I don't
   know" otherwise
4. Sends the prompt to `llama3.2` via Ollama and returns the answer

**Evaluation (`evals/test_rag.py`):**
1. Runs the full pipeline against known test questions
2. `FaithfulnessMetric` checks the answer doesn't invent facts beyond
   the retrieved context
3. `AnswerRelevancyMetric` checks the answer actually addresses the
   question
4. A dedicated safety test confirms unanswerable questions correctly
   trigger a refusal instead of a hallucinated answer
5. All metrics are judged by a local model (llama3.2) — no external API
   calls, no cost

Full narrative version: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Usage

```bash
# 1. Add .txt files to data/
# 2. Build the index
python src/ragbot/ingest.py

# 3. Ask questions
python src/ragbot/retrieve.py

# 4. Run the evaluation suite
deepeval test run evals/test_rag.py
```

## Project plan

| Phase | Status | Description |
|---|---|---|
| 1. Retrieval-Augmented Generation | Done | Chunking, embedding, semantic retrieval, grounded generation |
| 2. Agentic systems | Done | Tool/function calling, agent loop, custom MCP server |
| 3. Evaluation & safety | Done | Faithfulness, relevancy, and hallucination-refusal tests via DeepEval |
| 4. Integration & polish | Planned | Real external API integration, tagged releases |

## Progress log

| Date | Change |
|---|---|
| 2026-09-22 | RAG pipeline working end-to-end (ingest + retrieve + generate), fully local |
| 2026-09-23 | Automated evaluation suite added: faithfulness, relevancy, safety checks |
| 2026-09-23 | Full eval suite passing (3/3): switched faithfulness check from DeepEval's built-in metric to GEval after the built-in metric gave unreliable scores with the local judge model |

## What I learned building this

- How chunking and overlap affect retrieval accuracy in RAG
- Why embedding queries and documents with the same model is required
  for semantic search to work
- How to build in explicit hallucination guardrails at the prompt level
- How to configure and debug an LLM-judged evaluation pipeline running
  entirely on local infrastructure, with no external API dependency