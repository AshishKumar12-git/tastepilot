"""
Shared SQLite memory for the Qloo recommendation application.

Stores:
1. Episodic memory: privacy-minimized execution outcomes.
2. Semantic memory: explicitly verified, reusable application knowledge.

No user IDs, raw prompts, or personal conversation histories are stored.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


MEMORY_DIR = Path(__file__).resolve().parent
DB_PATH = MEMORY_DIR / "shared_memory.db"


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize_memory():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                signature TEXT NOT NULL,
                initial_take INTEGER,
                attempts INTEGER NOT NULL,
                retrieved_count INTEGER NOT NULL,
                ranked_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                final_decision TEXT,
                retry_improved INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episode_category
            ON episodic_memory(category)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                source TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_category
            ON semantic_memory(category)
        """)


def save_episode(
    category,
    signature,
    initial_take,
    attempts,
    retrieved_count,
    ranked_count,
    status,
    final_decision,
    retry_improved,
):
    """Persist execution metrics, not the raw user prompt or personal data."""
    initialize_memory()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO episodic_memory (
                category, signature, initial_take, attempts,
                retrieved_count, ranked_count, status,
                final_decision, retry_improved, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            category,
            signature,
            initial_take,
            attempts,
            retrieved_count,
            ranked_count,
            status,
            final_decision,
            int(bool(retry_improved)),
            _now(),
        ))


def add_semantic_memory(
    title,
    content,
    category=None,
    tags=None,
    source="manual",
    confidence=1.0,
    verified=False,
):
    """
    Add reusable knowledge.

    Set verified=True only when the content has been checked against
    reliable evidence (e.g. official API response/documentation or a test).
    """
    if not title or not content:
        raise ValueError("title and content are required.")

    confidence = max(0.0, min(1.0, float(confidence)))
    tags_json = json.dumps(tags or [], ensure_ascii=False)
    timestamp = _now()

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO semantic_memory (
                category, title, content, tags, source,
                confidence, verified, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            category,
            title.strip(),
            content.strip(),
            tags_json,
            source,
            confidence,
            int(bool(verified)),
            timestamp,
            timestamp,
        ))
        return cursor.lastrowid
