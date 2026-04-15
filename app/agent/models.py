from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


class AgentSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str
    status: str = "running"  # running | completed | failed
    messages: list = Field(default_factory=list)
    tool_calls: list = Field(default_factory=list)
    result: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
