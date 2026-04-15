import os
import time
import anthropic
from datetime import datetime
from app.agent.models import AgentSession
from app.tools.definitions import TOOLS
from app.tools.executor import execute_tool
from app.observability.tracing import get_tracer
from app.observability.metrics import (
    sessions_completed,
    tool_calls_total,
    session_duration_seconds,
    active_sessions
)
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
tracer = get_tracer()


def run_agent(session: AgentSession) -> AgentSession:
    active_sessions.inc()
    start_time = time.time()

    with tracer.start_as_current_span("agent.session") as span:
        span.set_attribute("session.id", session.session_id)
        span.set_attribute("session.task", session.task)

        messages = [{"role": "user", "content": session.task}]

        try:
            while True:
                with tracer.start_as_current_span("agent.llm_call") as llm_span:
                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=4096,
                        tools=TOOLS,
                        messages=messages
                    )
                    llm_span.set_attribute("stop_reason", response.stop_reason)
                    llm_span.set_attribute("input_tokens", response.usage.input_tokens)
                    llm_span.set_attribute("output_tokens", response.usage.output_tokens)

                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    for block in response.content:
                        if hasattr(block, "text"):
                            session.result = block.text
                    break

                if response.stop_reason == "tool_use":
                    tool_results = []

                    for block in response.content:
                        if block.type == "tool_use":
                            with tracer.start_as_current_span("agent.tool_call") as tool_span:
                                tool_span.set_attribute("tool.name", block.name)
                                tool_span.set_attribute("tool.input", str(block.input))

                                result = execute_tool(block.name, block.input)

                                tool_span.set_attribute("tool.result_length", len(result))
                                tool_calls_total.labels(tool_name=block.name).inc()

                            session.tool_calls.append({
                                "tool": block.name,
                                "input": block.input,
                                "result": result
                            })

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            })

                    messages.append({"role": "user", "content": tool_results})

                else:
                    session.status = "failed"
                    session.result = f"Unexpected stop reason: {response.stop_reason}"
                    break

            session.status = "completed"
            sessions_completed.labels(status="completed").inc()

        except Exception as e:
            session.status = "failed"
            session.result = str(e)
            sessions_completed.labels(status="failed").inc()
            span.record_exception(e)

        finally:
            duration = time.time() - start_time
            session_duration_seconds.observe(duration)
            active_sessions.dec()
            session.completed_at = datetime.utcnow()
            span.set_attribute("session.status", session.status)

    return session
