"""
Chunk extracted PDF pages while preserving document structure.

The chunker dynamically detects likely headings/titles from
the PDF instead of hardcoding names such as particular poems
or stories.
"""

import re

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


# ============================================================
# CHUNKER
# ============================================================

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
    ],
)


# ============================================================
# HEADING DETECTION
# ============================================================

def _looks_like_heading(
    line: str,
) -> bool:
    """
    Dynamically identify likely headings/titles.

    This does NOT contain any specific poem/story names.

    Examples it may detect:

        The Road Not Taken
        Introduction
        Chapter 1
        Questions
        Discussion
        I Sell My Dreams

    It intentionally avoids treating long sentences as headings.
    """

    line = line.strip()

    if not line:
        return False

    # Too long to reasonably be a heading.
    if len(line) > 120:
        return False

    # Very short lines are usually not useful headings.
    if len(line) < 3:
        return False

    # Don't treat normal sentences as headings.
    if line.endswith(
        (".", ",", ";", ":", "?", "!")
    ):
        return False

    # Ignore lines that look like page numbers.
    if re.fullmatch(
        r"[\d\s\-]+",
        line,
    ):
        return False

    words = line.split()

    # A heading usually isn't an entire paragraph.
    if len(words) > 15:
        return False

    # ALL CAPS headings.
    if (
        line.upper() == line
        and any(c.isalpha() for c in line)
    ):
        return True

    # Chapter/section numbering.
    if re.match(
        r"^(chapter|section|part|unit|lesson)\s+\d+",
        line,
        re.IGNORECASE,
    ):
        return True

    # Title Case-like headings.
    alpha_words = [
        word
        for word in words
        if any(c.isalpha() for c in word)
    ]

    if not alpha_words:
        return False

    capitalized = sum(
        1
        for word in alpha_words
        if word[0].isupper()
    )

    if (
        len(alpha_words) <= 10
        and capitalized
        / len(alpha_words)
        >= 0.60
    ):
        return True

    return False


# ============================================================
# EXTRACT PAGE SECTIONS
# ============================================================

def _extract_sections(
    text: str,
) -> list[dict]:
    """
    Split a page into logical sections using dynamically
    detected headings.

    If no heading is found, the complete page remains one
    section.
    """

    lines = text.splitlines()

    sections = []

    current_title = None
    current_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if _looks_like_heading(line):

            # Save previous section.
            if current_lines:

                sections.append(
                    {
                        "title": current_title,
                        "text": "\n".join(
                            current_lines
                        ),
                    }
                )

            current_title = line

            current_lines = []

        else:

            current_lines.append(
                line
            )

    # Save final section.
    if current_lines:

        sections.append(
            {
                "title": current_title,
                "text": "\n".join(
                    current_lines
                ),
            }
        )

    # If parsing didn't produce anything,
    # keep the complete page.
    if not sections:

        sections.append(
            {
                "title": None,
                "text": text,
            }
        )

    return sections


# ============================================================
# CHUNK PDF
# ============================================================

def chunk_pdf_pages(
    pages: list[dict],
    doc_name: str,
) -> list[dict]:
    """
    Convert extracted PDF pages into structured chunks.

    Each chunk contains:

        content
        source
        page
        section/title
        chunk index
        section index

    The title/section is also included in the embedded text
    so semantic retrieval has additional context.
    """

    chunks = []

    global_chunk_index = 0

    for page in pages:

        page_number = page[
            "page_number"
        ]

        page_text = page.get(
            "text",
            "",
        ).strip()

        if not page_text:
            continue

        sections = _extract_sections(
            page_text
        )

        for section_index, section in enumerate(
            sections
        ):

            section_title = section.get(
                "title"
            )

            section_text = section.get(
                "text",
                "",
            ).strip()

            if not section_text:
                continue

            # ------------------------------------------------
            # Add structural context to the text.
            # ------------------------------------------------

            if section_title:

                embedding_text = (
                    f"Section/Title: "
                    f"{section_title}\n\n"
                    f"{section_text}"
                )

            else:

                embedding_text = (
                    section_text
                )

            pieces = _splitter.split_text(
                embedding_text
            )

            for part, piece in enumerate(
                pieces
            ):

                if not piece.strip():
                    continue

                chunks.append(
                    {
                        "content": piece,

                        "metadata": {
                            "source": doc_name,

                            "page": page_number,

                            "type": "pdf",

                            "part": part,

                            "chunk_index":
                                global_chunk_index,

                            "section_index":
                                section_index,

                            "section":
                                section_title
                                or "",

                            "page_start":
                                page_number,

                            "page_end":
                                page_number,
                        },
                    }
                )

                global_chunk_index += 1

    return chunks