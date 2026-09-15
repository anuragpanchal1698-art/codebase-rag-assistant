"""
Chunking strategy:
- Code: split by function/class boundaries using Python's `ast` module when the
  file is .py (most reliable, no compiled parser dependency issues). For other
  languages, fall back to a generous line-based split that tries to respect
  blank-line boundaries (poor man's "don't cut mid-function").
- Docs: split by markdown headers first, then by size within sections.
"""
import ast
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from config import CODE_CHUNK_SIZE, CODE_CHUNK_OVERLAP, DOC_CHUNK_SIZE, DOC_CHUNK_OVERLAP

_fallback_code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CODE_CHUNK_SIZE,
    chunk_overlap=CODE_CHUNK_OVERLAP,
    separators=["\n\n", "\nclass ", "\ndef ", "\nfunction ", "\n\t", "\n", " "],
)

_doc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DOC_CHUNK_SIZE,
    chunk_overlap=DOC_CHUNK_OVERLAP,
)

_md_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)


def chunk_python_file(content: str, path: str) -> list[dict]:
    """AST-based chunking: one chunk per top-level function/class, with fallback."""
    chunks = []
    try:
        tree = ast.parse(content)
        lines = content.splitlines()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno  # requires Python 3.8+
                snippet = "\n".join(lines[start:end])

                if len(snippet) > CODE_CHUNK_SIZE * 3:
                    # huge class/function: sub-split it further
                    sub_chunks = _fallback_code_splitter.split_text(snippet)
                    for i, sc in enumerate(sub_chunks):
                        chunks.append({
                            "content": sc,
                            "metadata": {
                                "path": path, "symbol": node.name,
                                "type": "code", "node_type": type(node).__name__,
                                "start_line": start + 1, "part": i,
                            },
                        })
                else:
                    chunks.append({
                        "content": snippet,
                        "metadata": {
                            "path": path, "symbol": node.name,
                            "type": "code", "node_type": type(node).__name__,
                            "start_line": start + 1,
                        },
                    })

        if not chunks:
            # No top-level defs found (e.g. script-style file) -> fallback
            raise ValueError("no top-level defs")

    except (SyntaxError, ValueError):
        # Fallback: generic recursive split, tag symbol as unknown
        for i, piece in enumerate(_fallback_code_splitter.split_text(content)):
            chunks.append({
                "content": piece,
                "metadata": {"path": path, "symbol": None, "type": "code", "part": i},
            })

    return chunks


def chunk_generic_code_file(content: str, path: str) -> list[dict]:
    """For non-Python code (JS/TS/Go/etc.) - line-based fallback splitter."""
    pieces = _fallback_code_splitter.split_text(content)
    return [
        {"content": p, "metadata": {"path": path, "symbol": None, "type": "code", "part": i}}
        for i, p in enumerate(pieces)
    ]


def chunk_doc_file(content: str, path: str) -> list[dict]:
    """Markdown-aware chunking: split by headers, then by size within each section."""
    chunks = []
    try:
        header_sections = _md_header_splitter.split_text(content)
    except Exception:
        header_sections = [type("S", (), {"page_content": content, "metadata": {}})()]

    for section in header_sections:
        section_text = section.page_content
        section_meta = section.metadata
        for i, piece in enumerate(_doc_splitter.split_text(section_text)):
            chunks.append({
                "content": piece,
                "metadata": {
                    "path": path, "type": "doc", "part": i,
                    **section_meta,
                },
            })
    return chunks


def chunk_file(file_record: dict) -> list[dict]:
    """Dispatch to the right chunker based on file type/extension."""
    path, content, ftype, ext = (
        file_record["path"], file_record["content"],
        file_record["type"], file_record["extension"],
    )
    if ftype == "code" and ext == ".py":
        return chunk_python_file(content, path)
    elif ftype == "code":
        return chunk_generic_code_file(content, path)
    else:
        return chunk_doc_file(content, path)


def chunk_all_files(files: list[dict]) -> list[dict]:
    all_chunks = []
    for f in files:
        all_chunks.extend(chunk_file(f))
    print(f"Produced {len(all_chunks)} chunks from {len(files)} files")
    return all_chunks
