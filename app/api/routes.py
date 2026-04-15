from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.agent.models import AgentSession
from app.agent.runner import run_agent
from app.agent.store import save_session, get_session, list_sessions
from app.observability.metrics import sessions_created

router = APIRouter()


class CreateSessionRequest(BaseModel):
    task: str


def _run_and_save(session: AgentSession):
    updated = run_agent(session)
    save_session(updated)


@router.post("/sessions", status_code=202)
def create_session(req: CreateSessionRequest, background_tasks: BackgroundTasks):
    session = AgentSession(task=req.task)
    sessions_created.inc()
    save_session(session)
    background_tasks.add_task(_run_and_save, session)
    return {"session_id": session.session_id, "status": session.status}


@router.get("/sessions")
def get_all_sessions():
    return list_sessions()


@router.get("/sessions/{session_id}")
def get_session_by_id(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    from app.agent.store import _sessions
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    return {"deleted": session_id}
