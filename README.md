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

Question → embed → Chroma similarity search → top-k chunks
│
▼
LLM (llama3.2) generates answer,
grounded only in retrieved context

