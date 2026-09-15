"""
Manages a separate Qdrant collection for user-uploaded PDFs, kept apart from
the codebase collection so PDF and code search results never mix.
"""
import re
import pickle
from rank_bm25 import BM25Okapi
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import QDRANT_URL, EMBEDDING_DIM
from ingestion.embedder import get_embedder

PDF_COLLECTION = "user_pdfs"

_client = None
_pdf_bm25 = None
_pdf_bm25_corpus = None
PDF_BM25_PATH = "/app/data/pdf_bm25_corpus.pkl"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text.lower())


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def ensure_pdf_collection():
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if PDF_COLLECTION not in collections:
        client.create_collection(
            collection_name=PDF_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def embed_and_store_pdf_chunks(chunks: list[dict], batch_size: int = 32):
    """Embeds PDF chunks and upserts them into the dedicated PDF collection.
    Also appends to the PDF BM25 corpus so keyword search covers new uploads."""
    embedder = get_embedder()
    client = get_client()
    ensure_pdf_collection()

    new_bm25_entries = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c["content"] for c in batch]
        vectors = embedder.embed_documents(texts)

        points = []
        for chunk, vector in zip(batch, vectors):
            point_id = str(uuid.uuid4())
            payload = {"content": chunk["content"], **chunk["metadata"]}
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            new_bm25_entries.append({
                "id": point_id,
                "content": chunk["content"],
                "metadata": chunk["metadata"],
            })

        client.upsert(collection_name=PDF_COLLECTION, points=points)

    _append_to_pdf_bm25(new_bm25_entries)


def _load_pdf_bm25_corpus() -> list[dict]:
    try:
        with open(PDF_BM25_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []


def _append_to_pdf_bm25(new_entries: list[dict]):
    """Adds newly uploaded chunks to the persistent PDF BM25 corpus and
    invalidates the in-memory index so it rebuilds on next search."""
    global _pdf_bm25, _pdf_bm25_corpus

    corpus = _load_pdf_bm25_corpus()
    corpus.extend(new_entries)

    with open(PDF_BM25_PATH, "wb") as f:
        pickle.dump(corpus, f)

    _pdf_bm25 = None  # force rebuild on next search
    _pdf_bm25_corpus = None


def _get_pdf_bm25():
    global _pdf_bm25, _pdf_bm25_corpus
    if _pdf_bm25 is not None:
        return _pdf_bm25, _pdf_bm25_corpus

    _pdf_bm25_corpus = _load_pdf_bm25_corpus()
    if not _pdf_bm25_corpus:
        return None, []

    tokenized = [_tokenize(entry["content"]) for entry in _pdf_bm25_corpus]
    _pdf_bm25 = BM25Okapi(tokenized)
    return _pdf_bm25, _pdf_bm25_corpus


def sparse_search_pdfs(query: str, top_k: int = 5) -> list[dict]:
    """BM25 keyword search against uploaded PDF content."""
    bm25, corpus = _get_pdf_bm25()
    if bm25 is None:
        return []

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "id": corpus[i]["id"],
            "score": float(scores[i]),
            "content": corpus[i]["content"],
            "source": corpus[i]["metadata"].get("source", "unknown"),
            "page": corpus[i]["metadata"].get("page"),
        }
        for i in ranked_idx if scores[i] > 0
    ]
def search_pdfs(query: str, top_k: int = 5) -> list[dict]:
    """Dense search against the PDF collection only."""
    embedder = get_embedder()
    query_vector = embedder.embed_query(query)

    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if PDF_COLLECTION not in collections:
        return []  # no PDFs uploaded yet

    results = client.query_points(
        collection_name=PDF_COLLECTION,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        {
            "id": str(r.id),
            "score": r.score,
            "content": r.payload.get("content", ""),
            "source": r.payload.get("source", "unknown"),
            "page": r.payload.get("page"),
        }
        for r in results
    ]
def list_uploaded_pdfs() -> list[str]:
    """Returns the distinct set of PDF filenames currently indexed."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if PDF_COLLECTION not in collections:
        return []

    sources = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=PDF_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for r in records:
            if r.payload.get("source"):
                sources.add(r.payload["source"])
        if offset is None:
            break
    return sorted(sources)


def delete_pdf(doc_name: str):
    """Deletes all chunks belonging to a specific uploaded PDF."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = get_client()
    client.delete(
        collection_name=PDF_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=doc_name))]
        ),
    )
def hybrid_search_pdfs(query: str, top_k: int = 5, rrf_k: int = 60) -> list[dict]:
    """Combines dense + BM25 search over uploaded PDFs via Reciprocal Rank
    Fusion, matching the same approach used for codebase search."""
    dense_results = search_pdfs(query, top_k=10)
    sparse_results = sparse_search_pdfs(query, top_k=10)

    fused_scores = {}
    doc_lookup = {}

    for result_list in [dense_results, sparse_results]:
        for rank, doc in enumerate(result_list):
            doc_id = doc["id"]
            doc_lookup[doc_id] = doc
            fused_scores.setdefault(doc_id, 0.0)
            fused_scores[doc_id] += 1.0 / (rrf_k + rank + 1)

    ranked_ids = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)
    return [doc_lookup[doc_id] for doc_id in ranked_ids[:top_k]]