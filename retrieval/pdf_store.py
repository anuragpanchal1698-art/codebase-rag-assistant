"""
PDF-only retrieval system with chat isolation.

Retrieves ONLY from uploaded PDF files belonging to the
current chat.

Retrieval:
    1. Dense semantic search
    2. BM25 keyword search
    3. Reciprocal Rank Fusion
    4. Neighbor chunk expansion
    5. Chat-based filtering

Every PDF chunk is associated with a chat_id so PDFs from
different conversations can never be mixed.
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
    chat_id: str,
    batch_size: int = 32,
):
    """
    Embed and store PDF chunks belonging to one chat.

    chat_id is stored in BOTH:
        - Qdrant payload
        - BM25 metadata

    This guarantees that retrieval can be isolated per chat.
    """

    if not chunks:
        return

    if not chat_id:
        raise ValueError(
            "chat_id is required when storing PDF chunks."
        )

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

            metadata = dict(
                chunk.get(
                    "metadata",
                    {},
                )
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Associate this chunk with the current chat.
            # ------------------------------------------------

            metadata["chat_id"] = chat_id

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
    chat_id: str,
    top_k: int = 20,
) -> list[dict]:
    """
    BM25 search restricted to the current chat.
    """

    if not chat_id:
        return []

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

        # ----------------------------------------------------
        # CHAT ISOLATION
        # ----------------------------------------------------

        if metadata.get(
            "chat_id"
        ) != chat_id:

            continue

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

                "part": metadata.get(
                    "part"
                ),

                "chat_id": metadata.get(
                    "chat_id"
                ),
            }
        )

    return results


# ============================================================
# DENSE SEARCH
# ============================================================

def search_pdfs(
    query: str,
    chat_id: str,
    top_k: int = 20,
) -> list[dict]:
    """
    Dense semantic search restricted to the current chat.
    """

    if not query.strip():
        return []

    if not chat_id:
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

    # --------------------------------------------------------
    # IMPORTANT:
    # Qdrant only searches chunks belonging to this chat.
    # --------------------------------------------------------

    chat_filter = Filter(
        must=[
            FieldCondition(
                key="chat_id",
                match=MatchValue(
                    value=chat_id
                ),
            )
        ]
    )

    results = client.query_points(
        collection_name=PDF_COLLECTION,
        query=query_vector,
        query_filter=chat_filter,
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

                "chat_id": payload.get(
                    "chat_id"
                ),
            }
        )

    return output


# ============================================================
# GET ALL CHUNKS FROM ONE PDF
# ============================================================

def get_pdf_chunks(
    source: str,
    chat_id: str,
) -> list[dict]:
    """
    Get chunks from one PDF belonging to one chat only.
    """

    if not source or not chat_id:
        return []

    client = get_client()

    collections = [
        c.name
        for c in client.get_collections().collections
    ]

    if PDF_COLLECTION not in collections:

        return []

    records = []

    offset = None

    chat_filter = Filter(
        must=[
            FieldCondition(
                key="chat_id",
                match=MatchValue(
                    value=chat_id
                ),
            ),

            FieldCondition(
                key="source",
                match=MatchValue(
                    value=source
                ),
            ),
        ]
    )

    while True:

        points, offset = client.scroll(
            collection_name=PDF_COLLECTION,
            scroll_filter=chat_filter,
            limit=100,
            offset=offset,
            with_payload=True,
        )

        for point in points:

            payload = (
                point.payload or {}
            )

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

                    "chat_id": payload.get(
                        "chat_id"
                    ),
                }
            )

        if offset is None:
            break

    # --------------------------------------------------------
    # Preserve document order.
    # --------------------------------------------------------

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
    chat_id: str,
    radius: int = NEIGHBOR_RADIUS,
) -> list[dict]:
    """
    Expand relevant chunks using nearby chunks from the
    SAME PDF and SAME CHAT.
    """

    if not results:
        return []

    if not chat_id:
        return []

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
                    source,
                    chat_id,
                )
            )

        all_chunks = source_cache[
            source
        ]

        if not all_chunks:
            continue

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

            # Keep expansion within the same section
            # whenever both sections are known.
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
    chat_id: str,
    min_score: float = 0.30,
) -> tuple[str | None, float]:

    results = search_pdfs(
        query,
        chat_id=chat_id,
        top_k=20,
    )

    if not results:

        return (
            None,
            0.0,
        )

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
    chat_id: str,
    top_k: int = 10,
    rrf_k: int = 60,
    min_dense_score: float = 0.30,
) -> list[dict]:
    """
    Hybrid semantic + BM25 retrieval restricted to one chat.
    """

    if not query.strip():
        return []

    if not chat_id:
        return []

    # --------------------------------------------------------
    # DENSE SEARCH
    # --------------------------------------------------------

    dense_results = search_pdfs(
        query,
        chat_id=chat_id,
        top_k=20,
    )

    print(
        "\n========== PDF RETRIEVAL =========="
    )

    print(
        "CHAT:",
        chat_id,
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
        chat_id=chat_id,
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
    # RRF
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
    # NEIGHBOR EXPANSION
    # --------------------------------------------------------

    expanded_results = (
        expand_with_neighbors(
            primary_results,
            chat_id=chat_id,
            radius=NEIGHBOR_RADIUS,
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

def list_uploaded_pdfs(
    chat_id: str,
) -> list[str]:
    """
    Return PDFs belonging ONLY to the specified chat.
    """

    if not chat_id:
        return []

    client = get_client()

    collections = [
        c.name
        for c in client.get_collections().collections
    ]

    if PDF_COLLECTION not in collections:

        return []

    sources = set()

    offset = None

    chat_filter = Filter(
        must=[
            FieldCondition(
                key="chat_id",
                match=MatchValue(
                    value=chat_id
                ),
            )
        ]
    )

    while True:

        records, offset = client.scroll(
            collection_name=PDF_COLLECTION,
            scroll_filter=chat_filter,
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
    chat_id: str,
):
    """
    Delete a PDF ONLY from the specified chat.
    """

    if not doc_name or not chat_id:
        return

    client = get_client()

    collections = [
        c.name
        for c in client.get_collections().collections
    ]

    if PDF_COLLECTION not in collections:
        return

    # --------------------------------------------------------
    # Delete Qdrant points belonging to:
    #
    # current chat + requested PDF
    # --------------------------------------------------------

    delete_filter = Filter(
        must=[
            FieldCondition(
                key="chat_id",
                match=MatchValue(
                    value=chat_id
                ),
            ),

            FieldCondition(
                key="source",
                match=MatchValue(
                    value=doc_name
                ),
            ),
        ]
    )

    client.delete(
        collection_name=PDF_COLLECTION,
        points_selector=delete_filter,
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
        if not (
            entry.get(
                "metadata",
                {},
            ).get(
                "source"
            ) == doc_name
            and entry.get(
                "metadata",
                {},
            ).get(
                "chat_id"
            ) == chat_id
        )
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


# ============================================================
# DELETE ALL PDFs FOR A CHAT
# ============================================================

def delete_chat_pdfs(
    chat_id: str,
):
    """
    Delete every PDF belonging to one chat.

    This will be used when the user deletes a chat.
    """

    if not chat_id:
        return

    client = get_client()

    collections = [
        c.name
        for c in client.get_collections().collections
    ]

    if PDF_COLLECTION not in collections:
        return

    chat_filter = Filter(
        must=[
            FieldCondition(
                key="chat_id",
                match=MatchValue(
                    value=chat_id
                ),
            )
        ]
    )

    client.delete(
        collection_name=PDF_COLLECTION,
        points_selector=chat_filter,
    )

    # --------------------------------------------------------
    # Remove BM25 entries for this chat.
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
            "chat_id"
        ) != chat_id
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