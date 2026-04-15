from app.agent.models import AgentSession
from typing import Optional

# In-memory store for now, Postgres in Phase 3
_sessions: dict[str, AgentSession] = {}


def save_session(session: AgentSession) -> None:
    _sessions[session.session_id] = session


def get_session(session_id: str) -> Optional[AgentSession]:
    return _sessions.get(session_id)


def list_sessions() -> list[AgentSession]:
    return list(_sessions.values())
