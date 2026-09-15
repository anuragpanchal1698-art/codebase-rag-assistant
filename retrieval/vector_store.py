"""Dense (semantic) search against Qdrant."""
from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_COLLECTION, TOP_K_DENSE
from ingestion.embedder import get_embedder

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def dense_search(query: str, top_k: int = TOP_K_DENSE) -> list[dict]:
    """Returns list of {id, score, content, metadata} ranked by cosine similarity."""
    embedder = get_embedder()
    query_vector = embedder.embed_query(query)

    client = get_client()
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        {
            "id": str(r.id),
            "score": r.score,
            "content": r.payload.get("content", ""),
            "metadata": {k: v for k, v in r.payload.items() if k != "content"},
        }
        for r in results
    ]
