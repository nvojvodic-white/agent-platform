.PHONY: rag-rebuild rag-probes

# Rebuild the agent-api image and restart the container so new app/rag/* code
# goes live in the docker-compose stack. Without this, `docker compose restart`
# alone runs the old image and silently serves stale RAG code.
rag-rebuild:
	docker compose up -d --build agent-api

# Run the day-2/day-3 probe sweep against the live service. Override RAG_URL
# to point at a different port (e.g. RAG_URL=http://localhost:8124/... make rag-probes).
rag-probes:
	. venv/bin/activate && python -m app.rag.eval.run_probes
