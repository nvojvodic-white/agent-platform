"""Compatibility shim: let ragas 0.4.0 import under langchain-community >= 0.4.2.

ragas 0.4.0's ragas.llms.base does a top-level
`from langchain_community.chat_models.vertexai import ChatVertexAI`. That module
was removed in langchain-community 0.4.2, which our stack needs because
langchain-experimental 0.4.2 (semantic chunking) requires community >= 0.4.2.

ragas never instantiates ChatVertexAI unless a Vertex judge is selected (we use
Claude), so the import is the only thing that breaks. Register a stub module
with a placeholder class so the import resolves. Import this module BEFORE ragas.
"""
import sys
import types


def install() -> None:
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return
    try:
        import langchain_community.chat_models  # noqa: F401
    except Exception:
        return
    try:
        # If a real one ever comes back, don't shadow it.
        __import__(mod_name)
        return
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType(mod_name)

    class ChatVertexAI:  # placeholder; never instantiated in our usage
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ChatVertexAI is a compat stub; this stack does not use a "
                "Vertex judge. Use the Claude judge configured in ragas_eval."
            )

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[mod_name] = stub


install()
