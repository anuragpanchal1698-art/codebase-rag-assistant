"""
Hybrid search = dense (semantic) + sparse (BM25) results fused via
Reciprocal Rank Fusion (RRF), then reranked with a cross-encoder for the
final top-K. This is the piece most tutorials skip - RRF fusion + reranking
is what actually makes hybrid search outperform either method alone.
"""
from sentence_transformers import CrossEncoder
from config import TOP_K_DENSE, TOP_K_SPARSE, TOP_K_FINAL, RRF_K
from retrieval.vector_store import dense_search
from retrieval.bm25_store import sparse_search

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        print("Loading cross-encoder reranker (bge-reranker-base)...")
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Fuses multiple ranked lists into one, using RRF scoring:
    score(doc) = sum(1 / (k + rank)) across all lists it appears in.
    """
    fused_scores = {}
    doc_lookup = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            doc_id = doc["id"]
            doc_lookup[doc_id] = doc
            fused_scores.setdefault(doc_id, 0.0)
            fused_scores[doc_id] += 1.0 / (k + rank + 1)

    ranked_ids = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)
    return [
        {**doc_lookup[doc_id], "rrf_score": fused_scores[doc_id]}
        for doc_id in ranked_ids
    ]


def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_FINAL) -> list[dict]:
    """Cross-encoder reranking: scores (query, doc) pairs jointly - much more
    accurate than cosine similarity alone, at the cost of being slower (only
    run on the small fused candidate set, not the whole corpus)."""
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [[query, c["content"]] for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


def hybrid_search(query: str, top_k: int = TOP_K_FINAL, use_reranker: bool = True) -> list[dict]:
    dense_results = dense_search(query, top_k=TOP_K_DENSE)
    sparse_results = sparse_search(query, top_k=TOP_K_SPARSE)

    fused = reciprocal_rank_fusion([dense_results, sparse_results])

    if use_reranker:
        final = rerank(query, fused, top_k=top_k)
    else:
        final = fused[:top_k]

    return final
