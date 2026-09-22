"""
FastAPI backend for the React frontend.

Run with:
    uvicorn api.main:app --reload --port 8000

Exposes:
    POST /chats
    GET  /chats
    GET  /chats/{chat_id}
    DELETE /chats/{chat_id}

    POST /chat
    POST /chat/stream

    POST /upload-pdf
    GET  /pdfs

    GET  /health
"""

import sys
import os
import shutil

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    ),
)

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.responses import (
    StreamingResponse,
)

from pydantic import (
    BaseModel,
    Field,
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from agent.graph import (
    ask,
    astream,
    _ask_async,
)

from db.chat_db import (
    create_chat,
    delete_chat,
    get_chat,
    get_messages,
    init_db,
    list_chats,
    add_message,
    update_chat_title,
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Codebase RAG Assistant API"
)

init_db()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIG
# ============================================================

MAX_HISTORY_TURNS = 3

UPLOAD_DIR = "/app/data/uploaded_pdfs"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True,
)


# ============================================================
# MODELS
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    chat_id: str
    message: str
    history: list[ChatMessage] = Field(
        default_factory=list
    )


class ChatResponse(BaseModel):
    answer: str


# ============================================================
# HELPERS
# ============================================================

def _history_to_messages(
    history: list[ChatMessage],
):
    """
    Convert frontend chat history into
    LangChain message objects.
    """

    messages = []

    for message in history:

        if message.role == "user":

            messages.append(
                HumanMessage(
                    content=message.content
                )
            )

        elif message.role == "assistant":

            messages.append(
                AIMessage(
                    content=message.content
                )
            )

    return messages


def _capped_history_messages(
    history: list[ChatMessage],
):
    """
    Keep only the most recent exchanges
    to reduce token usage and latency.
    """

    capped = history[
        -(MAX_HISTORY_TURNS * 2):
    ]

    return _history_to_messages(
        capped
    )


def _make_chat_title(
    message: str,
) -> str:
    """
    Create a simple chat title from the
    user's first message.

    No LLM call is needed.
    """

    title = " ".join(
        message.strip().split()
    )

    if not title:
        return "New Chat"

    if len(title) > 45:
        title = title[:45].rstrip() + "..."

    return title


def _chat_upload_dir(
    chat_id: str,
) -> str:
    """
    Return the physical PDF directory
    belonging to one chat.
    """

    path = os.path.join(
        UPLOAD_DIR,
        chat_id,
    )

    os.makedirs(
        path,
        exist_ok=True,
    )

    return path


# ============================================================
# CHAT MANAGEMENT
# ============================================================

@app.post("/chats")
def new_chat():
    """
    Create a new independent conversation.
    """

    return create_chat()


@app.get("/chats")
def get_chats():
    """
    Return all conversations for the sidebar.
    """

    return {
        "chats": list_chats()
    }


@app.get("/chats/{chat_id}")
def get_chat_data(
    chat_id: str,
):
    """
    Load one conversation and its messages.
    """

    chat = get_chat(
        chat_id
    )

    if not chat:

        return {
            "error": "Chat not found."
        }

    return {
        "chat": chat,
        "messages": get_messages(
            chat_id
        ),
    }


@app.delete("/chats/{chat_id}")
def remove_chat(
    chat_id: str,
):
    """
    Delete a chat and all PDFs associated
    with that chat.
    """

    chat = get_chat(
        chat_id
    )

    if not chat:

        return {
            "error": "Chat not found."
        }

    # --------------------------------------------------------
    # Delete PDFs from Qdrant/BM25.
    # --------------------------------------------------------

    from retrieval.pdf_store import (
        delete_chat_pdfs,
    )

    delete_chat_pdfs(
        chat_id
    )

    # --------------------------------------------------------
    # Delete physical PDF files.
    # --------------------------------------------------------

    chat_pdf_dir = os.path.join(
        UPLOAD_DIR,
        chat_id,
    )

    if os.path.isdir(
        chat_pdf_dir
    ):
        shutil.rmtree(
            chat_pdf_dir,
            ignore_errors=True,
        )

    # --------------------------------------------------------
    # Delete SQLite chat history.
    # --------------------------------------------------------

    delete_chat(
        chat_id
    )

    return {
        "status": "deleted",
        "chat_id": chat_id,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    req: ChatRequest,
):
    """
    Non-streaming chat endpoint.

    Saves the user and assistant messages to SQLite.
    """

    # --------------------------------------------------------
    # Validate chat.
    # --------------------------------------------------------

    chat = get_chat(
        req.chat_id
    )

    if not chat:

        return ChatResponse(
            answer="Chat not found."
        )

    # --------------------------------------------------------
    # Create title from first message.
    # --------------------------------------------------------

    existing_messages = get_messages(
        req.chat_id
    )

    if not existing_messages:

        update_chat_title(
            req.chat_id,
            _make_chat_title(
                req.message
            ),
        )

    # --------------------------------------------------------
    # Save user message.
    # --------------------------------------------------------

    add_message(
        req.chat_id,
        "user",
        req.message,
    )

    # --------------------------------------------------------
    # Convert history.
    # --------------------------------------------------------

    history_messages = (
        _capped_history_messages(
            req.history
        )
    )

    # --------------------------------------------------------
    # Generate answer.
    # --------------------------------------------------------

    try:

        answer, _ = await _ask_async(
            req.message,
            chat_id=req.chat_id,
            chat_history=history_messages,
        )

    except Exception as e:

        return ChatResponse(
            answer=f"[Error: {str(e)}]"
        )

    # --------------------------------------------------------
    # Save assistant message.
    # --------------------------------------------------------

    add_message(
        req.chat_id,
        "assistant",
        answer,
    )

    return ChatResponse(
        answer=answer
    )


# ============================================================
# STREAMING CHAT
# ============================================================

@app.post(
    "/chat/stream"
)
async def chat_stream(
    req: ChatRequest,
):
    """
    Streaming chat endpoint.

    The user message is saved before generation.

    The assistant response is accumulated while streaming
    and saved to SQLite after successful generation.
    """

    # --------------------------------------------------------
    # Validate chat.
    # --------------------------------------------------------

    chat = get_chat(
        req.chat_id
    )

    if not chat:

        async def missing_chat():
            yield "Chat not found."

        return StreamingResponse(
            missing_chat(),
            media_type="text/plain",
        )

    # --------------------------------------------------------
    # Create title from first message.
    # --------------------------------------------------------

    existing_messages = get_messages(
        req.chat_id
    )

    if not existing_messages:

        update_chat_title(
            req.chat_id,
            _make_chat_title(
                req.message
            ),
        )

    # --------------------------------------------------------
    # Save user message.
    # --------------------------------------------------------

    add_message(
        req.chat_id,
        "user",
        req.message,
    )

    # --------------------------------------------------------
    # Convert frontend history.
    # --------------------------------------------------------

    history_messages = (
        _capped_history_messages(
            req.history
        )
    )

    # --------------------------------------------------------
    # Streaming generator.
    # --------------------------------------------------------

    async def event_generator():

        accumulated = ""

        try:

            async for chunk in astream(
                req.message,
                chat_id=req.chat_id,
                chat_history=history_messages,
            ):

                if chunk:

                    accumulated += chunk

                    yield chunk

            # ------------------------------------------------
            # Save complete assistant response.
            # ------------------------------------------------

            if accumulated.strip():

                add_message(
                    req.chat_id,
                    "assistant",
                    accumulated,
                )

        except Exception as e:

            error_message = (
                f"[Error: {str(e)}]"
            )

            yield (
                "\n\n" + error_message
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ============================================================
# PDF UPLOAD
# ============================================================

@app.post(
    "/upload-pdf"
)
async def upload_pdf(
    file: UploadFile = File(...),
    chat_id: str = Form(...),
):
    """
    Upload a PDF and associate it with one chat.

    Every chunk stored in Qdrant/BM25 receives
    the current chat_id.
    """

    # --------------------------------------------------------
    # Validate chat.
    # --------------------------------------------------------

    chat = get_chat(
        chat_id
    )

    if not chat:

        return {
            "error": "Chat not found."
        }

    # --------------------------------------------------------
    # Validate file.
    # --------------------------------------------------------

    if not file.filename:

        return {
            "error": "No filename provided."
        }

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        return {
            "error": "Only PDF files are supported."
        }

    # --------------------------------------------------------
    # Sanitize filename.
    # --------------------------------------------------------

    filename = os.path.basename(
        file.filename
    )

    if not filename:

        return {
            "error": "Invalid filename."
        }

    # --------------------------------------------------------
    # Chat-specific physical directory.
    # --------------------------------------------------------

    chat_pdf_dir = _chat_upload_dir(
        chat_id
    )

    save_path = os.path.join(
        chat_pdf_dir,
        filename,
    )

    # --------------------------------------------------------
    # Save PDF.
    # --------------------------------------------------------

    with open(
        save_path,
        "wb",
    ) as f:

        shutil.copyfileobj(
            file.file,
            f,
        )

    # --------------------------------------------------------
    # Import PDF pipeline.
    # --------------------------------------------------------

    from ingestion.pdf_loader import (
        extract_pdf_text,
    )

    from ingestion.pdf_chunker import (
        chunk_pdf_pages,
    )

    from retrieval.pdf_store import (
        embed_and_store_pdf_chunks,
    )

    # --------------------------------------------------------
    # Extract.
    # --------------------------------------------------------

    pages = extract_pdf_text(
        save_path
    )

    # --------------------------------------------------------
    # Chunk.
    # --------------------------------------------------------

    chunks = chunk_pdf_pages(
        pages,
        doc_name=filename,
    )

    # --------------------------------------------------------
    # Store with chat_id.
    # --------------------------------------------------------

    embed_and_store_pdf_chunks(
        chunks,
        chat_id=chat_id,
    )

    return {
        "status": "success",
        "filename": filename,
        "chunks_indexed": len(chunks),
        "chat_id": chat_id,
    }


# ============================================================
# LIST PDFs
# ============================================================

@app.get(
    "/pdfs"
)
def list_pdfs(
    chat_id: str,
):
    """
    List only PDFs belonging to the current chat.
    """

    chat = get_chat(
        chat_id
    )

    if not chat:

        return {
            "error": "Chat not found."
        }

    from retrieval.pdf_store import (
        list_uploaded_pdfs,
    )

    return {
        "pdfs": list_uploaded_pdfs(
            chat_id=chat_id
        )
    }