TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for current information on a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "execute_code",
        "description": "Execute a Python code snippet and return the output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file by path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path to read"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "search_middle_earth",
        "description": (
            "Search a curated corpus of Middle-earth lore (Fandom LotR wiki and "
            "Wikipedia) for information about Tolkien's legendarium: characters, "
            "places, events, battles, artifacts, languages, and history. Use this "
            "whenever the user asks about anything from The Hobbit, The Lord of the "
            "Rings, The Silmarillion, or Middle-earth generally. Returns an answer "
            "grounded in the corpus with cited sources. Do NOT use it for questions "
            "unrelated to Tolkien or Middle-earth."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The lore question to answer, phrased clearly and specifically."
                },
                "k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve (default 4, max 10)."
                }
            },
            "required": ["question"]
        }
    }
]
