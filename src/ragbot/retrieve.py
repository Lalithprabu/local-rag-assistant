"""
retrieve.py — takes a question, embeds it, finds the most relevant
chunks from Chroma, and generates a grounded answer using the local LLM.
"""

import chromadb
import ollama

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"


def embed_query(query: str) -> list[float]:
    """Embed the user's question the same way documents were embedded."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    return response["embedding"]


def retrieve_chunks(query: str, top_k: int = 3) -> list[str]:
    """Return the top_k most relevant chunks for a given query."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results["documents"][0] if results["documents"] else []


def generate_answer(question: str, chunks: list[str]) -> str:
    """Use the local LLM to answer the question, grounded in retrieved chunks."""
    context = "\n\n".join(chunks)
    prompt = f"""Answer the question using only the context below.
If the context doesn't contain the answer, say "I don't know based on the available documents."

Context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    question = input("Ask a question: ")
    chunks = retrieve_chunks(question)

    if not chunks:
        print("No relevant documents found.")
    else:
        answer = generate_answer(question, chunks)
        print(f"\nAnswer: {answer}\n")