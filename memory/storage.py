"""
memory/storage.py — SQLite-backed local persistent memory for MIKU.

Provides lightweight zero-overhead storage for:
1. Multi-turn conversation message logs (session-aware).
2. Explicit user profile facts (cross-session persistence).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MikuMemoryStore:
    """
    SQLite persistent memory store for local dialog history and key-value user facts.
    """

    def __init__(self, db_path: str = "data/memory/miku_memory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL,
                    source_session TEXT,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session's dialog history."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM messages WHERE session_id = ?",
                (session_id,),
            )
            next_turn = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO messages (session_id, turn_index, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, next_turn, role, content.strip(), now),
            )
            conn.commit()

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Retrieve ordered messages for a given session."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            if limit:
                cur.execute(
                    """
                    SELECT role, content FROM (
                        SELECT role, content, turn_index FROM messages
                        WHERE session_id = ?
                        ORDER BY turn_index DESC LIMIT ?
                    ) ORDER BY turn_index ASC
                    """,
                    (session_id, limit),
                )
            else:
                cur.execute(
                    "SELECT role, content FROM messages WHERE session_id = ? ORDER BY turn_index ASC",
                    (session_id,),
                )
            rows = cur.fetchall()
            return [{"role": r[0], "content": r[1]} for r in rows]

    def store_fact(self, key: str, value: str, source_session: Optional[str] = None) -> None:
        """Store or update a key-value user fact for cross-session recall."""
        now = datetime.now(timezone.utc).isoformat()
        clean_key = key.strip().lower()
        clean_val = value.strip()
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO facts (key, value, source_session, timestamp)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    source_session=excluded.source_session,
                    timestamp=excluded.timestamp
                """,
                (clean_key, clean_val, source_session, now),
            )
            conn.commit()

    def get_all_facts(self) -> Dict[str, str]:
        """Return all stored facts as a dictionary."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT key, value FROM facts ORDER BY id ASC")
            return {r[0]: r[1] for r in cur.fetchall()}

    def search_facts(self, query: str) -> List[Tuple[str, str]]:
        """
        Keyword overlap retrieval with regex tokenization and entity specificity scoring.
        Finds facts whose key or value shares words with query, ranking specific entity matches first.
        """
        import re
        words = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
        all_facts = self.get_all_facts()
        scored = []
        for k, v in all_facts.items():
            k_words = set(k.lower().replace("_", " ").split())
            v_words = set(v.lower().replace("_", " ").split())
            overlap = words & (k_words | v_words)
            if overlap:
                score = len(overlap)
                # Specificity bonus: if query matches the specific domain entity (e.g. 'dog', 'color')
                # rather than just the generic interrogative 'name', rank it higher.
                domain_entities = words - {"name", "user", "what", "how"}
                if any(term in k for term in domain_entities):
                    score += 2
                scored.append((score, k, v))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(k, v) for _, k, v in scored]

    def clear_session(self, session_id: str) -> None:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.commit()

    def clear_all(self) -> None:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM messages")
            cur.execute("DELETE FROM facts")
            conn.commit()
