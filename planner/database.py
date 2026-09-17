"""
Local SQLite database schema and helper for Miku Assistant.
Zero external dependencies; uses Python's built-in sqlite3.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "miku_store.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH):
    """Initialize database tables for tasks, calendar, devices, and audit logs."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'pending',
                due_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT
            )
        """)
        
        # Calendar events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                location TEXT,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Paired & known devices table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_type TEXT NOT NULL, -- 'bluetooth' or 'wifi'
                name TEXT NOT NULL,
                address TEXT NOT NULL UNIQUE,
                paired_at TEXT DEFAULT CURRENT_TIMESTAMP,
                is_trusted INTEGER DEFAULT 0,
                last_seen TEXT
            )
        """)
        
        # Audit log for actions and security confirmations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                command TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                confirmed INTEGER DEFAULT 0,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        """)
        conn.commit()


# Task CRUD operations
def add_task(title: str, description: str = "", priority: str = "medium", due_date: str = None, db_path: str = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tasks (title, description, priority, due_date) VALUES (?, ?, ?, ?)",
            (title, description, priority, due_date)
        )
        conn.commit()
        return cursor.lastrowid


def list_tasks(status: Optional[str] = None, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if status:
            cursor.execute("SELECT * FROM tasks WHERE status = ? ORDER BY id DESC", (status,))
        else:
            cursor.execute("SELECT * FROM tasks ORDER BY id DESC")
        return [dict(row) for row in cursor.fetchall()]


def update_task_status(task_id: int, status: str, db_path: str = DB_PATH) -> bool:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        completed_at = datetime.now().isoformat() if status == "completed" else None
        cursor.execute(
            "UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?",
            (status, completed_at, task_id)
        )
        conn.commit()
        return cursor.rowcount > 0


# Calendar operations
def add_calendar_event(title: str, start_time: str, end_time: Optional[str] = None, location: Optional[str] = None, description: Optional[str] = None, db_path: str = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO calendar_events (title, start_time, end_time, location, description) VALUES (?, ?, ?, ?, ?)",
            (title, start_time, end_time, location, description)
        )
        conn.commit()
        return cursor.lastrowid


def list_calendar_events(date_str: Optional[str] = None, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if date_str:
            cursor.execute("SELECT * FROM calendar_events WHERE start_time LIKE ? ORDER BY start_time ASC", (f"{date_str}%",))
        else:
            cursor.execute("SELECT * FROM calendar_events ORDER BY start_time ASC")
        return [dict(row) for row in cursor.fetchall()]


# Audit log
def log_audit(action_type: str, command: str, risk_level: str, confirmed: bool, details: Optional[Dict[str, Any]] = None, db_path: str = DB_PATH):
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        details_str = json.dumps(details) if details else ""
        cursor.execute(
            "INSERT INTO audit_log (action_type, command, risk_level, confirmed, details) VALUES (?, ?, ?, ?, ?)",
            (action_type, command, risk_level, 1 if confirmed else 0, details_str)
        )
        conn.commit()
