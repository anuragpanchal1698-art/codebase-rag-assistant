"""
Retrieval evaluation harness.

Runs each test case through hybrid_search() and checks whether the
expected file path appears in the top-K results. Reports precision-style
metrics: hit rate, average rank of the correct result, and per-case detail.

Usage:
    python -m eval.run_eval
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.hybrid_search import hybrid_search
from eval.test_cases import TEST_CASES

TOP_K = 5


def run_case(case: dict) -> dict:
    query = case["query"]
    expected = case["expected_path_contains"]

    results = hybrid_search(query, top_k=TOP_K, use_reranker=True)

    hit = False
    rank = None
    for i, r in enumerate(results):
        path = r["metadata"].get("path", "")
        if expected.lower() in path.lower():
            hit = True
            rank = i + 1  # 1-indexed
            break

    return {
        "query": query,
        "expected": expected,
        "hit": hit,
        "rank": rank,
        "top_paths": [r["metadata"].get("path", "?") for r in results],
    }


def run_eval():
    print(f"Running retrieval evaluation on {len(TEST_CASES)} test cases (top_k={TOP_K})...\n")

    results = []
    for case in TEST_CASES:
        result = run_case(case)
        results.append(result)

        status = "✅ HIT" if result["hit"] else "❌ MISS"
        rank_str = f"(rank {result['rank']})" if result["hit"] else ""
        print(f"{status} {rank_str:12} {result['query']}")
        if not result["hit"]:
            print(f"           expected path containing: '{result['expected']}'")
            print(f"           got: {result['top_paths']}")
        print()

    hits = sum(1 for r in results if r["hit"])
    total = len(results)
    hit_rate = hits / total if total else 0

    ranks = [r["rank"] for r in results if r["hit"]]
    avg_rank = sum(ranks) / len(ranks) if ranks else None

    print("=" * 60)
    print(f"Hit rate:        {hits}/{total} ({hit_rate:.1%})")
    if avg_rank:
        print(f"Avg rank of hit: {avg_rank:.1f} (lower is better, best=1)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_eval()