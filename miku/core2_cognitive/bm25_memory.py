"""
SQLite with Okapi BM25 and Trigram Indexing for Exact-Match Retrieval.
Core 2: Cognitive Router (Brain & Memory)
Trades fuzzy semantic recall for zero hallucinated recall.
"""
import sqlite3
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from miku.config import MEMORY_DB

class BM25Memory:
    def __init__(self, db_path: Path = MEMORY_DB, k1: float = 1.5, b: float = 0.75):
        self.db_path = db_path
        self.k1 = k1
        self.b = b
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_key TEXT UNIQUE,
                    content TEXT NOT NULL,
                    category TEXT,
                    created_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS term_index (
                    term TEXT,
                    doc_id INTEGER,
                    tf INTEGER,
                    PRIMARY KEY(term, doc_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_term ON term_index(term)")
            conn.commit()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into words and character trigrams for exact retrieval.
        """
        words = re.findall(r"\w+", text.lower())
        tokens = list(words)
        # Add trigrams
        for word in words:
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    tokens.append(f"tri:{word[i:i+3]}")
        return tokens

    def store(self, key: str, content: str, category: str = "general") -> int:
        """
        Stores and indexes a document.
        """
        tokens = self._tokenize(content)
        tf_map: Dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        import time
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO memory_docs (doc_key, content, category, created_at)
                VALUES (?, ?, ?, ?)
            """, (key, content, category, time.time()))
            doc_id = cursor.lastrowid
            if not doc_id:
                cursor.execute("SELECT id FROM memory_docs WHERE doc_key = ?", (key,))
                doc_id = cursor.fetchone()[0]

            cursor.execute("DELETE FROM term_index WHERE doc_id = ?", (doc_id,))
            for term, tf in tf_map.items():
                cursor.execute("""
                    INSERT INTO term_index (term, doc_id, tf) VALUES (?, ?, ?)
                """, (term, doc_id, tf))
            conn.commit()
            return doc_id

    def search(self, query: str, top_k: int = 3, min_score: float = 0.5) -> List[Dict[str, Any]]:
        """
        Performs Okapi BM25 scoring across indexed documents.
        Returns empty list if no document passes min_score (graceful not found).
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total docs count N
            cursor.execute("SELECT COUNT(*) FROM memory_docs")
            N = cursor.fetchone()[0]
            if N == 0:
                return []

            # Document lengths
            cursor.execute("SELECT doc_id, SUM(tf) FROM term_index GROUP BY doc_id")
            doc_lengths = dict(cursor.fetchall())
            avgdl = sum(doc_lengths.values()) / max(len(doc_lengths), 1)

            scores: Dict[int, float] = {}

            for q_term in set(query_tokens):
                # Document frequency df
                cursor.execute("SELECT COUNT(*) FROM term_index WHERE term = ?", (q_term,))
                df = cursor.fetchone()[0]
                if df == 0:
                    continue

                # IDF formula
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

                cursor.execute("SELECT doc_id, tf FROM term_index WHERE term = ?", (q_term,))
                for doc_id, tf in cursor.fetchall():
                    dl = doc_lengths.get(doc_id, avgdl)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                    term_score = idf * (tf * (self.k1 + 1.0)) / (denom + 1e-9)
                    scores[doc_id] = scores.get(doc_id, 0.0) + term_score

            if not scores:
                return []

            # Filter by min_score and sort
            filtered = [(doc_id, s) for doc_id, s in scores.items() if s >= min_score]
            filtered.sort(key=lambda x: x[1], reverse=True)
            top_results = filtered[:top_k]

            results = []
            for doc_id, score in top_results:
                cursor.execute("SELECT doc_key, content, category FROM memory_docs WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                if row:
                    results.append({
                        "key": row[0],
                        "content": row[1],
                        "category": row[2],
                        "score": round(score, 4)
                    })
            return results
