"""Tools available to the agent: codebase search, PDF search, file reading."""
from langchain_core.tools import tool
from retrieval.hybrid_search import hybrid_search
from retrieval.pdf_store import hybrid_search_pdfs


@tool
def search_codebase(query: str) -> str:
    """Search the LangChain codebase (source code + documentation) using hybrid
    semantic + keyword search. Use this for questions about how LangChain's
    code works, architecture, or its documentation."""
    results = hybrid_search(query, top_k=3)
    if not results:
        return "No relevant results found in the codebase for this query."

    formatted = []
    for r in results:
        meta = r["metadata"]
        loc = meta.get("path", "unknown")
        symbol = meta.get("symbol")
        if symbol:
            loc += f" :: {symbol}()"
        formatted.append(f"[{loc}]\n{r['content'][:400]}")

    return "\n\n---\n\n".join(formatted)


@tool
def search_pdf_documents(query: str) -> str:
    """Search documents the user has uploaded (PDFs). Use this when the user
    asks about content from a PDF they uploaded, or asks a question unrelated
    to the LangChain codebase."""
    results = hybrid_search_pdfs(query, top_k=5)
    if not results:
        return "No uploaded PDFs found, or no relevant results for this query."

    formatted = []
    for r in results:
        loc = f"{r['source']}"
        if r.get("page"):
            loc += f" (page {r['page']})"
        formatted.append(f"[{loc}]\n{r['content'][:350]}")

    return "\n\n---\n\n".join(formatted)


@tool
def get_file_content(file_path: str) -> str:
    """Fetch the full content of a specific file from the cloned LangChain
    repo by its relative path."""
    import os
    from config import REPO_CLONE_DIR

    full_path = os.path.join(REPO_CLONE_DIR, file_path)
    if not os.path.exists(full_path):
        return f"File not found: {file_path}"

    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if len(content) > 6000:
        content = content[:6000] + "\n... [truncated]"
    return content


ALL_TOOLS = [search_codebase, search_pdf_documents, get_file_content]