# Architecture

## Overview

This is a Retrieval-Augmented Generation (RAG) system that runs entirely
locally using Ollama, with no external API calls or cost. It answers
questions grounded in a set of source documents rather than relying on
the LLM's raw memory, reducing hallucination.

## Flow

User question
│
▼
┌─────────────────┐
│ embed_query() │ → nomic-embed-text turns the question into a vector
└─────────────────┘
│
▼
┌─────────────────┐
│ Chroma similarity│ → compares the question's vector against every
│ search │ stored chunk vector, returns the closest matches
└─────────────────┘
│
▼
┌─────────────────┐
│ generate_answer()│ → llama3.2 receives the question + retrieved
│ │ chunks as context, and is instructed to answer
│ │ only from that context
└─────────────────┘
│
▼
Final answer


## Components

**`ingest.py`** — one-time (or repeatable) setup step.
1. Reads all `.txt` files from `data/`
2. Splits each into ~500-character overlapping chunks (`chunk_text`) so
   no single embedding has to represent an entire document
3. Embeds each chunk via Ollama's `nomic-embed-text` model
4. Stores the chunk text, its embedding, and its source filename in a
   local Chroma database on disk (`chroma_db/`)

**`retrieve.py`** — the query-time pipeline.
1. Embeds the incoming question with the same embedding model, so the
   question and the documents live in the same "meaning space"
2. Asks Chroma for the `top_k` most similar stored chunks
3. Builds a prompt that includes those chunks as context and instructs
   the model to answer only from them, saying "I don't know" otherwise
4. Sends that prompt to `llama3.2` via Ollama and returns the answer

## Why this design

- **Chunking with overlap** avoids splitting a sentence or idea across
  two disconnected chunks, which would otherwise weaken retrieval
  accuracy.
- **Same embedding model for documents and queries** is required —
  otherwise their vectors wouldn't be comparable.
- **Explicit "don't know" instruction** in the prompt is a basic safety
  guardrail against hallucination: the model is directed to defer to
  the retrieved context rather than invent an answer.
- **Fully local (Ollama + Chroma)** keeps this at $0 cost and keeps all
  data on-device — no external API calls.

## Known limitations (honest, not hidden)

- With very few documents, retrieval can return an irrelevant chunk
  simply because it's the "closest available" match — this is expected
  to improve as the document set grows.
- No evaluation layer yet — retrieval and answer quality are currently
  checked manually. This is the next planned addition (see `evals/`).
- No safeguards yet against prompt injection via document content —
  also planned.