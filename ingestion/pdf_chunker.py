"""Chunks extracted PDF page text into embedding-ready pieces."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)


def chunk_pdf_pages(pages: list[dict], doc_name: str) -> list[dict]:
    """Takes {page_number, text} dicts and returns chunk dicts with metadata
    tagging them as belonging to this PDF and page."""
    chunks = []
    for page in pages:
        pieces = _splitter.split_text(page["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "content": piece,
                "metadata": {
                    "source": doc_name,
                    "page": page["page_number"],
                    "type": "pdf",
                    "part": i,
                },
            })
    return chunks