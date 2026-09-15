"""
FastAPI backend for the React frontend.

Run with: uvicorn api.main:app --reload --port 8000

Exposes:
  POST /chat          - non-streaming: send a message + history, get full answer back
  POST /chat/stream    - streaming: same, but tokens stream back as they're generated
  POST /upload-pdf     - upload a PDF, chunk + embed it into the PDF collection
  GET  /pdfs           - list currently indexed PDF filenames
  GET  /health         - simple healthcheck
"""
import sys
import os
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import ask, astream, _ask_async

app = FastAPI(title="Codebase RAG Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_HISTORY_TURNS = 3  # keep last N user+assistant exchanges only, to cap token usage

UPLOAD_DIR = "/app/data/uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str


def _history_to_messages(history: list[ChatMessage]):
    """Convert simple {role, content} history from the frontend into LangChain
    message objects."""
    messages = []
    for m in history:
        if m.role == "user":
            messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            messages.append(AIMessage(content=m.content))
    return messages


def _capped_history_messages(history: list[ChatMessage]):
    capped = history[-(MAX_HISTORY_TURNS * 2):]
    return _history_to_messages(capped)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming endpoint - waits for the full answer before responding."""
    history_messages = _capped_history_messages(req.history)
    answer, _ = await _ask_async(req.message, chat_history=history_messages)
    return ChatResponse(answer=answer)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming endpoint - yields text chunks as the agent generates them."""
    history_messages = _capped_history_messages(req.history)

    async def event_generator():
        try:
            async for chunk in astream(req.message, chat_history=history_messages):
                yield chunk
        except Exception as e:
            yield f"\n\n[Error: {str(e)}]"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF, extracts text, chunks it, embeds it, and stores it in
    the dedicated PDF Qdrant collection so it becomes searchable."""
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    from ingestion.pdf_loader import extract_pdf_text
    from ingestion.pdf_chunker import chunk_pdf_pages
    from retrieval.pdf_store import embed_and_store_pdf_chunks

    pages = extract_pdf_text(save_path)
    chunks = chunk_pdf_pages(pages, doc_name=file.filename)
    embed_and_store_pdf_chunks(chunks)

    return {"status": "success", "filename": file.filename, "chunks_indexed": len(chunks)}


@app.get("/pdfs")
def list_pdfs():
    """Lists filenames of all currently indexed PDFs."""
    from retrieval.pdf_store import list_uploaded_pdfs
    return {"pdfs": list_uploaded_pdfs()}