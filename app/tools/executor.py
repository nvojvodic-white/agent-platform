import os
import sys
import httpx
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
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return f"[web_search] TAVILY_API_KEY not set. Query was: {query}"

    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": 5,
                "search_depth": "basic",
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return "No results found."

        lines = []
        for r in results:
            lines.append(f"**{r.get('title', 'No title')}**")
            lines.append(r.get("url", ""))
            lines.append(r.get("content", "")[:300])
            lines.append("")
        return "\n".join(lines)

    except Exception as e:
        return f"Search error: {str(e)}"


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
