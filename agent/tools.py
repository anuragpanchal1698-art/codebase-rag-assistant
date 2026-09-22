"""
Tools available to the PDF-only RAG assistant.

IMPORTANT:
There is deliberately NO codebase search,
GitHub search, file reader, or MCP tool here.
"""

from langchain_core.tools import tool

from retrieval.pdf_store import (
    hybrid_search_pdfs,
)


@tool
def search_pdf_documents(
    query: str,
) -> str:
    """
    Search ONLY uploaded PDF documents.
    """

    results = hybrid_search_pdfs(
        query,
        top_k=8,
    )

    if not results:

        return (
            "NO_RELEVANT_PDF_CONTENT: "
            "No sufficiently relevant information "
            "was found in the uploaded PDF. "
            "Do not answer using general knowledge."
        )

    formatted = []

    for result in results:

        source = result.get(
            "source",
            "unknown",
        )

        page = result.get(
            "page"
        )

        location = source

        if page:
            location += (
                f" (page {page})"
            )

        formatted.append(
            f"[PDF SOURCE: {location}]\n"
            f"{result.get('content', '')}"
        )

    return "\n\n---\n\n".join(
        formatted
    )


ALL_TOOLS = [
    search_pdf_documents
]