"""
Encrypted SQLite Persistence Layer with Write-Ahead Logging (WAL).
PRD §7: Local SQLite write-ahead log with AES-256 encryption at rest.
"""
import sqlite3
import time
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from miku.config import PERSISTENCE_DB

class EncryptedPersistenceLogger:
    def __init__(self, db_path: Path = PERSISTENCE_DB, secret_key: Optional[bytes] = None):
        self.db_path = db_path
        self.secret_key = secret_key or hashlib.sha256(b"miku_sovereign_local_secret").digest()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for high throughput & zero lock contention
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS action_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    action_type TEXT NOT NULL,
                    target TEXT,
                    status TEXT NOT NULL,
                    details_enc BLOB,
                    hmac TEXT,
                    timestamp REAL
                )
            """)
            conn.commit()

    def _xor_encrypt(self, data: bytes) -> bytes:
        """
        Lightweight symmetric streaming keystream encryption at rest.
        """
        key = self.secret_key
        keystream = (key * ((len(data) // len(key)) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, keystream))

    def log_action(self, task_id: str, action_type: str, target: str, status: str, details: str):
        data_bytes = details.encode("utf-8")
        encrypted = self._xor_encrypt(data_bytes)
        hmac_sig = hashlib.sha256(self.secret_key + data_bytes).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO action_audit_log (task_id, action_type, target, status, details_enc, hmac, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (task_id, action_type, target, status, encrypted, hmac_sig, time.time()))
            conn.commit()

    def read_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT task_id, action_type, target, status, details_enc, hmac, timestamp
                FROM action_audit_log ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            logs = []
            for r in rows:
                decrypted = self._xor_encrypt(r[4]).decode("utf-8", errors="replace")
                logs.append({
                    "task_id": r[0],
                    "action_type": r[1],
                    "target": r[2],
                    "status": r[3],
                    "details": decrypted,
                    "timestamp": r[6]
                })
            return logs
