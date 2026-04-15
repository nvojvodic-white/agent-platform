import subprocess
import sys
from io import StringIO


def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "web_search":
        return _web_search(tool_input["query"])
    elif tool_name == "execute_code":
        return _execute_code(tool_input["code"])
    elif tool_name == "read_file":
        return _read_file(tool_input["path"])
    else:
        return f"Unknown tool: {tool_name}"


def _web_search(query: str) -> str:
    # Stub for now — will wire to real search in Phase 2
    return f"[web_search stub] Results for: {query}"


def _execute_code(code: str) -> str:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {})
        output = sys.stdout.getvalue()
        return output if output else "Code executed successfully with no output."
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout


def _read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"
