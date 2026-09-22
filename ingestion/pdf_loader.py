"""
Extract text from uploaded PDF files page by page.

Each page keeps its original page number so later retrieval
can preserve document structure.
"""

from pypdf import PdfReader


def extract_pdf_text(file_path: str) -> list[dict]:
    """
    Extract PDF text page by page.

    Returns:
        [
            {
                "page_number": 1,
                "text": "..."
            },
            ...
        ]
    """

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        text = page.extract_text() or ""

        # Normalize line endings.
        text = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        # Remove excessive blank lines.
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        text = "\n".join(lines)

        if not text.strip():
            continue

        pages.append(
            {
                "page_number": page_number,
                "text": text,
            }
        )

    return pages