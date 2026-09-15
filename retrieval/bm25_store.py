"""Sparse (keyword) search using BM25 - great for exact symbol/function-name matches."""
import pickle
import re
from rank_bm25 import BM25Okapi
from config import TOP_K_SPARSE

_bm25 = None
_corpus = None


def _tokenize(text: str) -> list[str]:
    # Simple tokenizer that keeps identifiers like snake_case / camelCase intact-ish
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text.lower())


def load_bm25(corpus_path: str = "./data/bm25_corpus.pkl"):
    global _bm25, _corpus
    with open(corpus_path, "rb") as f:
        _corpus = pickle.load(f)

    tokenized = [_tokenize(entry["content"]) for entry in _corpus]
    _bm25 = BM25Okapi(tokenized)
    print(f"Loaded BM25 index with {len(_corpus)} documents")
    return _bm25


def sparse_search(query: str, top_k: int = TOP_K_SPARSE) -> list[dict]:
    global _bm25, _corpus
    if _bm25 is None:
        load_bm25()

    tokenized_query = _tokenize(query)
    scores = _bm25.get_scores(tokenized_query)

    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "id": _corpus[i]["id"],
            "score": float(scores[i]),
            "content": _corpus[i]["content"],
            "metadata": _corpus[i]["metadata"],
        }
        for i in ranked_idx if scores[i] > 0
    ]
