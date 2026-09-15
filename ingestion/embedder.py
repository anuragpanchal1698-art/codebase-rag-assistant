"""Embeds chunks and stores them in Qdrant. Also serializes a BM25-ready corpus."""
import pickle
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import (
    EMBEDDING_MODEL, EMBEDDING_DIM, QDRANT_URL, QDRANT_COLLECTION,
)

_embeddings = None


def get_embedder():
    global _embeddings
    if _embeddings is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL} (first run downloads weights)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},          # switch to "cuda" if you have a GPU
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def ensure_collection(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        print(f"Creating Qdrant collection '{QDRANT_COLLECTION}'...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def embed_and_store(chunks: list[dict], batch_size: int = 64) -> str:
    """Embeds all chunks and upserts into Qdrant. Also pickles the corpus for BM25.
    Returns the path to the pickled corpus (chunk text + metadata + ids)."""
    embedder = get_embedder()
    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client)

    corpus_for_bm25 = []  # list of {id, content, metadata} kept in same order

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c["content"] for c in batch]
        vectors = embedder.embed_documents(texts)

        points = []
        for chunk, vector in zip(batch, vectors):
            point_id = str(uuid.uuid4())
            payload = {"content": chunk["content"], **chunk["metadata"]}
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            corpus_for_bm25.append({"id": point_id, "content": chunk["content"], "metadata": chunk["metadata"]})

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"Embedded + upserted {start + len(batch)}/{len(chunks)} chunks")

    bm25_path = "./data/bm25_corpus.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(corpus_for_bm25, f)
    print(f"Saved BM25 corpus ({len(corpus_for_bm25)} entries) -> {bm25_path}")

    return bm25_path
