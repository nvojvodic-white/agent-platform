from app.agent.models import AgentSession
from app.agent.runner import run_agent

session = AgentSession(task="Calculate the first 10 fibonacci numbers using the execute_code tool.")
result = run_agent(session)

print(f"\nStatus: {result.status}")
print(f"Tool calls: {len(result.tool_calls)}")
print(f"\nResult:\n{result.result}")
