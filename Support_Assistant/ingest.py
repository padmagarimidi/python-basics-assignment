"""
Ingestion stage of the RAG pipeline: ingestion -> embedding -> (storage in ChromaDB).

- load_documents(): reads the 8 policy .txt files from docs/
- chunk_document(): splits each document's text into fixed-size chunks (a document
  this short generally becomes a single chunk, but the function still applies a
  real fixed-size chunking scheme rather than special-casing "one chunk per file").
- build_or_get_collection(): embeds every chunk locally with sentence-transformers'
  all-MiniLM-L6-v2 model (no API key, no network call) and upserts the embeddings,
  along with their text and metadata, into a persistent ChromaDB collection called
  "zepto_policies".

This module has no dependency on MOCK_LLM: embedding and vector storage always run
for real in both modes, since they require no LLM API key and no network call.
"""

import os
import glob
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_store")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE_CHARS = 400  # fixed-size chunking scheme (chars per chunk)
CHUNK_OVERLAP_CHARS = 50

_model_cache = {}


def get_embedding_model() -> SentenceTransformer:
    """Lazily load and cache the local sentence-transformers embedding model."""
    if "model" not in _model_cache:
        _model_cache["model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_cache["model"]


def load_documents() -> Dict[str, str]:
    """Read every doc_XX.txt file in docs/ into {doc_id: full_text}."""
    doc_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt")))
    documents = {}
    for path in doc_paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r", encoding="utf-8") as f:
            documents[doc_id] = f.read().strip()
    return documents


def chunk_document(doc_id: str, text: str, chunk_size: int = CHUNK_SIZE_CHARS,
                    overlap: int = CHUNK_OVERLAP_CHARS) -> List[Dict[str, str]]:
    """
    Fixed-size character chunking with overlap. For these short policy documents
    this typically yields a single chunk per document, but the same logic scales
    to longer documents without modification.
    Returns a list of {"chunk_id": ..., "doc_id": ..., "text": ...} dicts.
    """
    chunks = []
    if len(text) <= chunk_size:
        chunks.append({"chunk_id": f"{doc_id}_chunk0", "doc_id": doc_id, "text": text})
        return chunks

    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append({"chunk_id": f"{doc_id}_chunk{idx}", "doc_id": doc_id, "text": chunk_text})
        idx += 1
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk_all_documents(documents: Dict[str, str]) -> List[Dict[str, str]]:
    all_chunks = []
    for doc_id, text in documents.items():
        all_chunks.extend(chunk_document(doc_id, text))
    return all_chunks


def build_or_get_collection(force_rebuild: bool = False):
    """
    Embeds every chunk with all-MiniLM-L6-v2 and upserts into the persistent
    ChromaDB collection "zepto_policies". Safe to call repeatedly: uses upsert
    keyed by chunk_id, so re-running ingestion is idempotent.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity, per spec
    )

    documents = load_documents()
    chunks = chunk_all_documents(documents)

    if collection.count() >= len(chunks) and not force_rebuild:
        # Already ingested.
        return collection

    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"doc_id": c["doc_id"]} for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return collection


def get_collection():
    """Convenience accessor used by the retrieval node: ensures data is ingested."""
    return build_or_get_collection(force_rebuild=False)


if __name__ == "__main__":
    col = build_or_get_collection(force_rebuild=True)
    print(f"Ingested {col.count()} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
