import os
import json
import sqlite3
from typing import Optional
from contextlib import contextmanager
from app.agent.models import AgentSession

DB_PATH = os.getenv("DB_PATH", "/data/sessions.db")


@contextmanager
def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                data       TEXT NOT NULL
            )
        """)


_init_db()


def save_session(session: AgentSession) -> None:
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO sessions (session_id, data) VALUES (?, ?)",
            (session.session_id, session.model_dump_json()),
        )


def get_session(session_id: str) -> Optional[AgentSession]:
    with _conn() as con:
        row = con.execute(
            "SELECT data FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    if row:
        return AgentSession.model_validate(json.loads(row["data"]))
    return None


def list_sessions() -> list[AgentSession]:
    with _conn() as con:
        rows = con.execute(
            "SELECT data FROM sessions ORDER BY rowid DESC"
        ).fetchall()
    return [AgentSession.model_validate(json.loads(r["data"])) for r in rows]


def delete_session(session_id: str) -> bool:
    with _conn() as con:
        cur = con.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
    return cur.rowcount > 0
