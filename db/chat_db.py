import os
import sqlite3
import uuid
from datetime import datetime, timezone

DB_PATH = os.getenv(
    "CHAT_DB_PATH",
    "/app/data/chat_history.db",
)


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    return conn


def init_db():
    conn = _connect()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(chat_id) REFERENCES chats(id)
        )
        """
    )

    conn.commit()
    conn.close()


def create_chat(title: str = "New Chat") -> dict:
    chat_id = str(uuid.uuid4())

    now = datetime.now(timezone.utc).isoformat()

    conn = _connect()

    conn.execute(
        """
        INSERT INTO chats
        (id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            chat_id,
            title,
            now,
            now,
        ),
    )

    conn.commit()
    conn.close()

    return {
        "id": chat_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
    }


def list_chats() -> list[dict]:
    conn = _connect()

    rows = conn.execute(
        """
        SELECT id, title, created_at, updated_at
        FROM chats
        ORDER BY updated_at DESC
        """
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def get_chat(chat_id: str):
    conn = _connect()

    row = conn.execute(
        """
        SELECT id, title, created_at, updated_at
        FROM chats
        WHERE id = ?
        """,
        (chat_id,),
    ).fetchone()

    conn.close()

    return dict(row) if row else None


def update_chat_title(
    chat_id: str,
    title: str,
):
    conn = _connect()

    conn.execute(
        """
        UPDATE chats
        SET title = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            title,
            datetime.now(timezone.utc).isoformat(),
            chat_id,
        ),
    )

    conn.commit()
    conn.close()


def touch_chat(chat_id: str):
    conn = _connect()

    conn.execute(
        """
        UPDATE chats
        SET updated_at = ?
        WHERE id = ?
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            chat_id,
        ),
    )

    conn.commit()
    conn.close()


def add_message(
    chat_id: str,
    role: str,
    content: str,
):
    conn = _connect()

    conn.execute(
        """
        INSERT INTO messages
        (chat_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            chat_id,
            role,
            content,
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    conn.commit()
    conn.close()

    touch_chat(chat_id)


def get_messages(
    chat_id: str,
) -> list[dict]:

    conn = _connect()

    rows = conn.execute(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE chat_id = ?
        ORDER BY id ASC
        """,
        (chat_id,),
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def delete_chat(
    chat_id: str,
):
    conn = _connect()

    conn.execute(
        """
        DELETE FROM messages
        WHERE chat_id = ?
        """,
        (chat_id,),
    )

    conn.execute(
        """
        DELETE FROM chats
        WHERE id = ?
        """,
        (chat_id,),
    )

    conn.commit()
    conn.close()