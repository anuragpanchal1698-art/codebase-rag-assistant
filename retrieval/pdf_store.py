"""
PDF-only retrieval system.

Retrieves ONLY from the uploaded PDF collection.

The retrieval system:
    1. Uses dense semantic search
    2. Uses BM25 keyword search
    3. Combines results with RRF
    4. Expands relevant chunks with nearby chunks
    5. Keeps expansion inside the same PDF/section when possible

No poem names, story names, or question types are hardcoded.
"""

import re
import pickle
import uuid

from rank_bm25 import BM25Okapi

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from config import QDRANT_URL, EMBEDDING_DIM
from ingestion.embedder import get_embedder


# ============================================================
# CONFIG
# ============================================================

PDF_COLLECTION = "user_pdfs"

PDF_BM25_PATH = "/app/data/pdf_bm25_corpus.pkl"

_client = None

_pdf_bm25 = None
_pdf_bm25_corpus = None

# Number of neighboring chunks added around a relevant chunk.
NEIGHBOR_RADIUS = 2


# ============================================================
# TOKENIZATION
# ============================================================

def _tokenize(text: str) -> list[str]:

    return re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|\d+",
        text.lower(),
    )


# ============================================================
# QDRANT
# ============================================================

def get_client() -> QdrantClient:

    global _client

    if _client is None:

        _client = QdrantClient(
            url=QDRANT_URL
        )

    return _client


def ensure_pdf_collection():

    client = get_client()

    collections = [
        c.name
        for c in client.get_collections().collections
    ]

    if PDF_COLLECTION not in collections:

        client.create_collection(
            collection_name=PDF_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )


# ============================================================
# STORE PDF CHUNKS
# ============================================================

def embed_and_store_pdf_chunks(
    chunks: list[dict],
    batch_size: int = 32,
):

    if not chunks:
        return

    embedder = get_embedder()

    client = get_client()

    ensure_pdf_collection()

    new_entries = []

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):

        batch = chunks[
            start:start + batch_size
        ]

        texts = [
            chunk["content"]
            for chunk in batch
        ]

        vectors = embedder.embed_documents(
            texts
        )

        points = []

        for chunk, vector in zip(
            batch,
            vectors,
        ):

            point_id = str(
                uuid.uuid4()
            )

            metadata = chunk.get(
                "metadata",
                {},
            )

            payload = {
                "content": chunk["content"],
                **metadata,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

            new_entries.append(
                {
                    "id": point_id,
                    "content": chunk["content"],
                    "metadata": metadata,
                }
            )

        client.upsert(
            collection_name=PDF_COLLECTION,
            points=points,
        )

    _append_to_pdf_bm25(
        new_entries
    )


# ============================================================
# BM25
# ============================================================

def _load_pdf_bm25_corpus() -> list[dict]:

    try:

        with open(
            PDF_BM25_PATH,
            "rb",
        ) as f:

            return pickle.load(f)

    except Exception:

        return []


def _append_to_pdf_bm25(
    new_entries: list[dict],
):

    global _pdf_bm25
    global _pdf_bm25_corpus

    if not new_entries:
        return

    corpus = (
        _load_pdf_bm25_corpus()
    )

    corpus.extend(
        new_entries
    )

    with open(
        PDF_BM25_PATH,
        "wb",
    ) as f:

        pickle.dump(
            corpus,
            f,
        )

    _pdf_bm25 = None
    _pdf_bm25_corpus = None


def _get_pdf_bm25():

    global _pdf_bm25
    global _pdf_bm25_corpus

    if _pdf_bm25 is not None:

        return (
            _pdf_bm25,
            _pdf_bm25_corpus,
        )

    _pdf_bm25_corpus = (
        _load_pdf_bm25_corpus()
    )

    if not _pdf_bm25_corpus:

        return (
            None,
            [],
        )

    tokenized = [
        _tokenize(
            entry["content"]
        )
        for entry in _pdf_bm25_corpus
    ]

    _pdf_bm25 = BM25Okapi(
        tokenized
    )

    return (
        _pdf_bm25,
        _pdf_bm25_corpus,
    )


# ============================================================
# BM25 SEARCH
# ============================================================

def sparse_search_pdfs(
    query: str,
    top_k: int = 20,
) -> list[dict]:

    bm25, corpus = (
        _get_pdf_bm25()
    )

    if bm25 is None:
        return []

    query_tokens = _tokenize(
        query
    )

    if not query_tokens:
        return []

    scores = bm25.get_scores(
        query_tokens
    )

    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    results = []

    for i in ranked:

        if len(results) >= top_k:
            break

        if scores[i] <= 0:
            continue

        entry = corpus[i]

        metadata = entry.get(
            "metadata",
            {},
        )

        results.append(
            {
                "id": entry["id"],
                "score": float(
                    scores[i]
                ),
                "content": entry[
                    "content"
                ],
                "source": metadata.get(
                    "source",
                    "unknown",
                ),
                "page": metadata.get(
                    "page"
                ),
                "section": metadata.get(
                    "section",
                    "",
                ),
                "chunk_index": metadata.get(
                    "chunk_index"
                ),
            }
        )

    return results


# ============================================================
# DENSE SEARCH
# ============================================================

def search_pdfs(
    query: str,
    top_k: int = 20,
) -> list[dict]:

    if not query.strip():
        return []

    embedder = get_embedder()

    query_vector = (
        embedder.embed_query(
            query
        )
    )

    client = get_client()

    collections = [
        c.name
        for c in client.get_collections().collections
    ]

    if PDF_COLLECTION not in collections:

        return []

    results = client.query_points(
        collection_name=PDF_COLLECTION,
        query=query_vector,
        limit=top_k,
    ).points

    output = []

    for result in results:

        payload = (
            result.payload or {}
        )

        output.append(
            {
                "id": str(
                    result.id
                ),

                "score": float(
                    result.score
                ),

                "content": payload.get(
                    "content",
                    "",
                ),

                "source": payload.get(
                    "source",
                    "unknown",
                ),

                "page": payload.get(
                    "page"
                ),

                "section": payload.get(
                    "section",
                    "",
                ),

                "chunk_index": payload.get(
                    "chunk_index"
                ),

                "part": payload.get(
                    "part"
                ),
            }
        )

    return output


# ============================================================
# GET ALL CHUNKS FROM ONE PDF
# ============================================================

def get_pdf_chunks(
    source: str,
) -> list[dict]:

    client = get_client()

    collections = [
        c.name
        for c in client.get_collections().collections
    ]

    if PDF_COLLECTION not in collections:

        return []

    records = []

    offset = None

    while True:

        points, offset = client.scroll(
            collection_name=PDF_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
        )

        for point in points:

            payload = (
                point.payload or {}
            )

            if payload.get(
                "source"
            ) != source:

                continue

            records.append(
                {
                    "id": str(
                        point.id
                    ),

                    "content": payload.get(
                        "content",
                        "",
                    ),

                    "source": payload.get(
                        "source",
                        source,
                    ),

                    "page": payload.get(
                        "page"
                    ),

                    "section": payload.get(
                        "section",
                        "",
                    ),

                    "chunk_index": payload.get(
                        "chunk_index"
                    ),

                    "part": payload.get(
                        "part"
                    ),
                }
            )

        if offset is None:
            break

    # New chunker provides chunk_index.
    # Fall back to page/part for older data.
    records.sort(
        key=lambda x: (
            x.get(
                "chunk_index"
            )
            if isinstance(
                x.get(
                    "chunk_index"
                ),
                int,
            )
            else (
                x.get("page")
                if isinstance(
                    x.get("page"),
                    (int, float),
                )
                else 999999
            ),

            x.get(
                "part"
            )
            if isinstance(
                x.get("part"),
                int,
            )
            else 0,
        )
    )

    return records


# ============================================================
# NEIGHBOR EXPANSION
# ============================================================

def expand_with_neighbors(
    results: list[dict],
    radius: int = NEIGHBOR_RADIUS,
) -> list[dict]:
    """
    For every relevant result, retrieve nearby chunks.

    This is the important part for stories/poems.

    Example:

        chunk 20 = matching discussion question
        chunk 19 = poem
        chunk 18 = poem
        chunk 21 = poem

    The surrounding content can now be supplied to the LLM.

    When section metadata exists, expansion stays inside
    the same section whenever possible.
    """

    if not results:
        return []

    # Cache complete PDFs only when necessary.
    source_cache = {}

    expanded = {}

    for result in results:

        source = result.get(
            "source"
        )

        if not source:
            continue

        if source not in source_cache:

            source_cache[source] = (
                get_pdf_chunks(
                    source
                )
            )

        all_chunks = source_cache[
            source
        ]

        if not all_chunks:
            continue

        # ----------------------------------------------------
        # Locate matching chunk.
        # ----------------------------------------------------

        result_id = result.get(
            "id"
        )

        index = None

        for i, chunk in enumerate(
            all_chunks
        ):

            if chunk.get(
                "id"
            ) == result_id:

                index = i
                break

        if index is None:
            continue

        target_section = (
            result.get(
                "section",
                "",
            )
            or ""
        )

        # ----------------------------------------------------
        # Add nearby chunks.
        # ----------------------------------------------------

        start = max(
            0,
            index - radius,
        )

        end = min(
            len(all_chunks),
            index + radius + 1,
        )

        for neighbor in all_chunks[
            start:end
        ]:

            neighbor_section = (
                neighbor.get(
                    "section",
                    "",
                )
                or ""
            )

            # If both chunks have section information,
            # do not cross into another section.
            if (
                target_section
                and neighbor_section
                and target_section
                != neighbor_section
            ):
                continue

            expanded[
                neighbor["id"]
            ] = neighbor

    # --------------------------------------------------------
    # Preserve document order.
    # --------------------------------------------------------

    output = list(
        expanded.values()
    )

    output.sort(
        key=lambda x: (
            x.get(
                "source",
                "",
            ),

            x.get(
                "chunk_index"
            )
            if isinstance(
                x.get(
                    "chunk_index"
                ),
                int,
            )
            else (
                x.get("page")
                if isinstance(
                    x.get("page"),
                    (int, float),
                )
                else 999999
            ),

            x.get(
                "part"
            )
            if isinstance(
                x.get("part"),
                int,
            )
            else 0,
        )
    )

    return output


# ============================================================
# FIND RELEVANT PDF
# ============================================================

def find_relevant_pdf(
    query: str,
    min_score: float = 0.30,
) -> tuple[str | None, float]:

    results = search_pdfs(
        query,
        top_k=20,
    )

    if not results:

        return (
            None,
            0.0,
        )

    # Find strongest source rather than blindly
    # assuming result 0 is always the best document.
    best_by_source = {}

    for result in results:

        source = result.get(
            "source"
        )

        score = float(
            result.get(
                "score",
                0.0,
            )
        )

        if not source:
            continue

        if (
            source not in best_by_source
            or score
            > best_by_source[source]
        ):

            best_by_source[
                source
            ] = score

    if not best_by_source:

        return (
            None,
            0.0,
        )

    source, score = max(
        best_by_source.items(),
        key=lambda x: x[1],
    )

    if score < min_score:

        return (
            None,
            score,
        )

    return (
        source,
        score,
    )


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search_pdfs(
    query: str,
    top_k: int = 10,
    rrf_k: int = 60,
    min_dense_score: float = 0.30,
) -> list[dict]:
    """
    Hybrid semantic + BM25 retrieval.

    After finding relevant chunks, neighboring chunks are
    automatically added so the LLM receives enough context
    to understand the actual story/poem/document section.
    """

    if not query.strip():
        return []

    # --------------------------------------------------------
    # DENSE SEARCH
    # --------------------------------------------------------

    dense_results = search_pdfs(
        query,
        top_k=20,
    )

    print(
        "\n========== PDF RETRIEVAL =========="
    )

    print(
        "QUERY:",
        query,
    )

    for result in dense_results[:10]:

        print(
            f"SCORE: "
            f"{result.get('score', 0.0):.4f} | "
            f"SOURCE: "
            f"{result.get('source')} | "
            f"PAGE: "
            f"{result.get('page')} | "
            f"SECTION: "
            f"{result.get('section', '')}"
        )

        print(
            result.get(
                "content",
                "",
            )[:250]
        )

        print(
            "-----------------------------------"
        )

    # --------------------------------------------------------
    # SEMANTIC FILTER
    # --------------------------------------------------------

    relevant_dense = [
        result
        for result in dense_results
        if float(
            result.get(
                "score",
                0.0,
            )
        ) >= min_dense_score
    ]

    if not relevant_dense:

        print(
            "NO RELEVANT PDF CONTENT"
        )

        print(
            "===================================\n"
        )

        return []

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    sparse_results = sparse_search_pdfs(
        query,
        top_k=20,
    )

    relevant_ids = {
        result["id"]
        for result in relevant_dense
    }

    filtered_sparse = [
        result
        for result in sparse_results
        if result["id"]
        in relevant_ids
    ]

    # --------------------------------------------------------
    # RECIPROCAL RANK FUSION
    # --------------------------------------------------------

    fused_scores = {}

    lookup = {}

    for result_list in (
        relevant_dense,
        filtered_sparse,
    ):

        for rank, result in enumerate(
            result_list
        ):

            doc_id = result[
                "id"
            ]

            lookup[
                doc_id
            ] = result

            fused_scores.setdefault(
                doc_id,
                0.0,
            )

            fused_scores[
                doc_id
            ] += (
                1.0
                / (
                    rrf_k
                    + rank
                    + 1
                )
            )

    ranked_ids = sorted(
        fused_scores,
        key=lambda doc_id:
            fused_scores[doc_id],
        reverse=True,
    )

    primary_results = [
        lookup[doc_id]
        for doc_id in ranked_ids[
            :top_k
        ]
    ]

    # --------------------------------------------------------
    # EXPAND CONTEXT
    # --------------------------------------------------------

    expanded_results = (
        expand_with_neighbors(
            primary_results,
            radius=2,
        )
    )

    print(
        f"PRIMARY RESULTS: "
        f"{len(primary_results)}"
    )

    print(
        f"EXPANDED RESULTS: "
        f"{len(expanded_results)}"
    )

    print(
        "===================================\n"
    )

    return expanded_results


# ============================================================
# LIST UPLOADED PDFS
# ============================================================

def list_uploaded_pdfs() -> list[str]:

    client = get_client()

    collections = [
        c.name
        for c in client.get_collections().collections
    ]

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

        for record in records:

            payload = (
                record.payload or {}
            )

            source = payload.get(
                "source"
            )

            if source:

                sources.add(
                    source
                )

        if offset is None:
            break

    return sorted(
        sources
    )


# ============================================================
# DELETE PDF
# ============================================================

def delete_pdf(
    doc_name: str,
):

    client = get_client()

    collections = [
        c.name
        for c in client.get_collections().collections
    ]

    if PDF_COLLECTION not in collections:
        return

    # --------------------------------------------------------
    # Delete Qdrant points.
    # --------------------------------------------------------

    client.delete(
        collection_name=PDF_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(
                        value=doc_name
                    ),
                )
            ]
        ),
    )

    # --------------------------------------------------------
    # Delete BM25 entries.
    # --------------------------------------------------------

    global _pdf_bm25
    global _pdf_bm25_corpus

    corpus = (
        _load_pdf_bm25_corpus()
    )

    filtered = [
        entry
        for entry in corpus
        if entry.get(
            "metadata",
            {},
        ).get(
            "source"
        ) != doc_name
    ]

    try:

        with open(
            PDF_BM25_PATH,
            "wb",
        ) as f:

            pickle.dump(
                filtered,
                f,
            )

    except Exception:
        pass

    _pdf_bm25 = None
    _pdf_bm25_corpus = None