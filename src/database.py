#!/usr/bin/env python3
"""
database.py - SQLite database for storing analysis sessions.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Tuple, Optional

DB_PATH = "emg_sessions.db"


def init_db():
    """Create the sessions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT,
            results_json TEXT NOT NULL,
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_session(filename: str, results_json: str, notes: str = ""):
    """Save a session to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute('''
        INSERT INTO sessions (timestamp, filename, results_json, notes)
        VALUES (?, ?, ?, ?)
    ''', (now, filename, results_json, notes))
    conn.commit()
    conn.close()


def load_sessions(limit: int = 10) -> List[Tuple]:
    """Retrieve recent sessions."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, timestamp, filename, results_json, notes
        FROM sessions
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows


def load_session_by_id(session_id: int) -> Optional[Tuple]:
    """Load a specific session by ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, timestamp, filename, results_json, notes
        FROM sessions
        WHERE id = ?
    ''', (session_id,))
    row = c.fetchone()
    conn.close()
    return row


def delete_session(session_id: int):
    """Delete a session by ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()