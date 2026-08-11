# Deployment

The deploy decision for this project, written out rather than executed. This
doc is the honest version of the "Path B" close-out: the system can run
on EKS, the work to get it there is bounded and known, and that work is not
where the portfolio value of the RAG project sits (it sits in
[`app/rag/eval/findings.md`](app/rag/eval/findings.md) and the measured
routing decisions). The companion repo `dev-platform` is where the
EKS/Karpenter/Argo CD work lives.

## Architecture as it stands

The system has four services on the happy path. Most of the wiring predates the
RAG service:

```
React frontend (frontend/)
   |
   | HTTP
   v
agent-api  (FastAPI, Python agent runner in app/agent/runner.py)
   |
   | tool dispatch: search_middle_earth
   | HTTP via RAG_SERVICE_URL
   v
RAG service (FastAPI, LangGraph routing agent in app/rag/agent/graph.py)
   |
   v
Chroma indices (data/chroma_middle_earth, data/chroma_semantic, data/chroma_pdr)
```

A user question entered in the React UI creates an agent session via
`POST /api/v1/sessions`. The agent's tool registry exposes `search_middle_earth`
([`app/tools/rag_search.py`](app/tools/rag_search.py)) which HTTP-calls the
RAG service. The RAG service's routing agent classifies the question, retrieves
via the appropriate index, grades, optionally retries, and returns a cited
answer. The agent passes that through to the session result.

If all four pieces run and `RAG_SERVICE_URL` resolves and the Chroma DBs hold
data, the React app answering a Middle-earth question routes through the agent,
fires the RAG tool, hits Chroma, and returns a grounded answer with citations.
This is already demonstrable on local `docker-compose`.

## What changes on EKS

Four real things, none hard, all worth knowing before spending a Saturday on
the deploy:

### 1. Service discovery and RAG_SERVICE_URL

Locally: `http://localhost:8000` or `http://rag:8000` via the docker-compose
service name. In Kubernetes: `http://rag.<namespace>.svc.cluster.local:8000`,
the in-cluster DNS name. If both services are in the same namespace, the short
`http://rag:8000` also resolves. Configured via the `RAG_SERVICE_URL` env var
on the agent deployment; the value is trivial to set and easy to forget, then
debug for thirty minutes as "agent can't reach RAG."

### 2. Where Chroma lives

The one architecturally interesting question. Three options, with the tradeoff
each forces:

- **Bake the index into the RAG container image.** Container starts, Chroma DB
  is already on disk under `/app/data/chroma_middle_earth/`. No persistent
  volume needed. Adds ~150-200 MB to the image (we have three indices: dense
  ~65 MB, semantic ~60 MB, PDR ~95 MB). Re-indexing means rebuild + push. For a
  demo with a static corpus, this is the move.
- **PVC with the Chroma DB on it.** Container stays lean; index lives on a
  persistent volume. Better for production where the index is rebuilt
  periodically or shared across replicas. Awkward with our current setup:
  Chroma here is local file I/O, only one pod can write at a time, so multi-
  replica requires either ReadWriteMany (rare) or a leader-pod pattern.
- **Hosted vector DB** (Pinecone, Qdrant Cloud, Weaviate Cloud). The
  multi-replica production answer. Overkill for portfolio scope and adds
  ongoing cost.

For "can I show this working on EKS": bake it into the image, single replica,
container has everything it needs, no PV gymnastics. That's the same call the
GitHub Actions workflow header already flags as TODO (it currently warns that
index restore isn't wired).

### 3. Secrets

The RAG service needs both `ANTHROPIC_API_KEY` (judge + chain) and
`OPENAI_API_KEY` (embeddings). The agent service needs only
`ANTHROPIC_API_KEY`. In Kubernetes both come from a Secret mounted as env
vars. If `dev-platform` has External Secrets Operator wired in, use that
pattern; otherwise plain `kubectl create secret generic` is fine for a demo.

The current Helm chart (`charts/agent-platform/`) ships a `secret.yaml`
template wired to `ANTHROPIC_API_KEY` and `PLATFORM_API_KEY` only. Adding
`OPENAI_API_KEY` is a one-line template change plus a `values.yaml` entry. A
real Path-A execution would do that change first.

### 4. Ingress

Only the React app needs to be externally reachable. The agent-api and RAG
service should stay cluster-internal: agent is called by frontend, RAG is
called by agent, neither needs public access. Three services, one external
surface. Easy to accidentally over-expose and worth being deliberate about.

## The Helm chart shape

Following the subchart pattern from `dev-platform`:

```
charts/agent-platform/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── frontend-deployment.yaml     # React app (the only public surface)
│   ├── frontend-service.yaml
│   ├── frontend-ingress.yaml
│   ├── agent-deployment.yaml        # Python agent runner
│   ├── agent-service.yaml           # ClusterIP, internal
│   ├── rag-deployment.yaml          # Python RAG service, Chroma baked in
│   ├── rag-service.yaml             # ClusterIP, internal
│   └── secrets.yaml                 # or External Secrets reference
```

The agent deployment sets `RAG_SERVICE_URL=http://rag:8000`. The frontend
deployment sets `AGENT_API_URL=http://agent:8000`. The frontend ingress is the
only manifest with a public hostname.

Note: the chart currently in `charts/agent-platform/` ships templates for the
agent service (the pre-RAG portfolio shape). Splitting it into
agent + rag + frontend per the structure above is the real Path-A work item.

## What's true today vs what's needed for Path A

| piece | state today | gap to ship on EKS |
|---|---|---|
| RAG application code | done | none |
| Dockerfile builds the RAG service | yes, same `app.main:app` entrypoint hosts both `/api/v1/sessions` and `/api/v1/rag/*` | the current single-image approach co-locates the agent and RAG; splitting into two images is the cleaner Path-A move |
| Helm chart | exists for the agent (`charts/agent-platform/`) | split into agent / rag / frontend subcharts per the shape above |
| `OPENAI_API_KEY` in secret + env | not in the chart | one-line template + values.yaml addition |
| `RAG_SERVICE_URL` env on agent | wired in code (`app/tools/rag_search.py`) | one env block on the agent deployment |
| Chroma indices | local `data/chroma_*` (gitignored, ~200 MB combined) | bake into RAG image at build time, OR build PVC + init-container that rebuilds from `data/raw/` |
| EKS cluster | not in this repo | provisioned by `dev-platform` |
| Argo CD app | not in this repo | one Application manifest pointing at this chart |
| Observability | OTel + Jaeger + Prometheus + Grafana exist for the agent | RAG service emits no spans yet; add the existing `setup_tracing()` + decorators to RAG routes |

## The honest read

The application is done. The deploy work is bounded:

- A weekend to bake-and-ship a single-replica EKS deployment for a live demo
  URL. That's a real differentiator: most candidate portfolios are GitHub repos
  with screenshots, not running URLs.
- A full week to deploy it *well*: HPA, NetworkPolicy, External Secrets, Argo
  CD pipeline, RAG observability wired, blue-green or canary rollout. Most of
  that work would duplicate skills already demonstrated in `dev-platform`.

Path B (this document) is the close-out for the RAG project specifically: the
portfolio value of the measured-iteration arc lives in
[`findings.md`](app/rag/eval/findings.md), the [README](README.md), the
[scorecard template](app/rag/eval/scorecard.md), and the
[routing agent](app/rag/agent/graph.py). The deploy is a real next step but
not where the measured-RAG story's value sits.

## Local demo (the reproducible cut)

The demo cut that actually runs today, without EKS:

```bash
# 1. one-time setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. build indices (~5 min total: scrape Fandom+Wikipedia, embed)
python -m app.rag.ingestion.fetch
python -m app.rag.ingestion.build_index
python -m app.rag.ingestion.build_semantic_index   # optional: for semantic route
python -m app.rag.ingestion.build_pdr_index        # optional: PDR alternative

# 3. set keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...

# 4. run the service (agent + RAG + agent-tool path all in one process)
uvicorn app.main:app --port 8000

# 5. demo the routing agent
curl -X POST localhost:8000/api/v1/rag/agent_query_debug \
  -H 'content-type: application/json' \
  -d '{"question": "Who killed Smaug?"}'
```

The `agent_query_debug` response includes the classifier's reasoning, the
chosen route, the grade, and the trace, which is what makes the demo
substantively different from a generic "type a question, get an answer"
chatbot.
