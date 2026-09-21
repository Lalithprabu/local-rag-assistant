"""
ingest.py — loads text documents, splits them into chunks,
embeds them, and stores them in a local Chroma vector database.
"""

import os
import chromadb
import ollama

DATA_DIR = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents"
EMBED_MODEL = "nomic-embed-text"


def load_documents(data_dir: str) -> list[dict]:
    """Load all .txt files from data_dir into memory."""
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                docs.append({"id": filename, "text": f.read()})
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks of roughly chunk_size characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_text(text: str) -> list[float]:
    """Get an embedding vector for a piece of text using a local Ollama model."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def main():
    if not os.path.exists(DATA_DIR):
        print(f"No '{DATA_DIR}' folder found. Create it and add some .txt files first.")
        return

    documents = load_documents(DATA_DIR)
    if not documents:
        print(f"No .txt files found in '{DATA_DIR}'. Add some documents and try again.")
        return

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    chunk_id = 0
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            embedding = embed_text(chunk)
            collection.add(
                ids=[f"{doc['id']}_{chunk_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": doc["id"]}],
            )
            chunk_id += 1
        print(f"Ingested {doc['id']} ({len(chunks)} chunks)")

    print(f"Done. {chunk_id} chunks stored in '{CHROMA_PATH}'.")


if __name__ == "__main__":
    main()