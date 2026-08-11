.PHONY: demo demo-check rag-rebuild rag-probes eval-one eval-all eval-compare

# Bring up the whole local stack (API + RAG + UI). Handles venv creation,
# dependency install, and preflight; see scripts/demo.py.
demo:
	./demo.sh

# Preflight only — reports missing venv / keys / indices and starts nothing.
demo-check:
	./demo.sh --check

# Run the RAGAS eval for one retriever, n-runs averaged.
# Override on the command line, e.g.: make eval-one RETRIEVER=hyde N_RUNS=3
RETRIEVER ?= dense
K ?= 4
N_RUNS ?= 1
eval-one:
	. venv/bin/activate && python -m app.rag.eval.ragas_eval \
		--retriever $(RETRIEVER) --k $(K) --n-runs $(N_RUNS)

# Run the full retriever matrix at N_RUNS each (default 3 for noise smoothing).
EVAL_RETRIEVERS ?= dense sparse hybrid hybrid_40_60 hyde multi_query pdr semantic
EVAL_N_RUNS ?= 3
eval-all:
	. venv/bin/activate && for r in $(EVAL_RETRIEVERS); do \
		echo "=== $$r ==="; \
		python -m app.rag.eval.ragas_eval --retriever $$r --k $(K) --n-runs $(EVAL_N_RUNS) || exit 1; \
	done

# Diff the latest RETRIEVER history against the latest BASELINE_RETRIEVER
# history (default baseline: dense). E.g.: make eval-compare RETRIEVER=hyde
BASELINE_RETRIEVER ?= dense
eval-compare:
	. venv/bin/activate && python -m app.rag.eval.compare_to_baseline \
		--baseline-retriever $(BASELINE_RETRIEVER) --retriever $(RETRIEVER) --k $(K)


# Rebuild the agent-api image and restart the container so new app/rag/* code
# goes live in the docker-compose stack. Without this, `docker compose restart`
# alone runs the old image and silently serves stale RAG code.
rag-rebuild:
	docker compose up -d --build agent-api

# Run the probe sweep against the live service. Override RAG_URL
# to point at a different port (e.g. RAG_URL=http://localhost:8124/... make rag-probes).
rag-probes:
	. venv/bin/activate && python -m app.rag.eval.run_probes
