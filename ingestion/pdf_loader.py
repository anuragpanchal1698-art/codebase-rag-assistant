"""Extracts text from uploaded PDF files, page by page."""
from pypdf import PdfReader


def extract_pdf_text(file_path: str) -> list[dict]:
    """Returns a list of {page_number, text} dicts, one per page."""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page_number": i + 1, "text": text})
    return pages