# Agent Platform

A self-hosted platform for running and observing Claude-powered AI agents. Built with FastAPI, deployed on Kubernetes via Helm, with full observability through OpenTelemetry, Jaeger, Prometheus, and Grafana.

## What this demonstrates

This project is a portfolio demonstration of production-grade platform engineering applied to an agentic AI workload.

**Infrastructure ownership**
- Kubernetes deployment via Helm with HPA (2→5 replicas), NetworkPolicy (default-deny), resource limits, and liveness/readiness probes
- Multi-environment config: `values-staging.yaml` / `values-prod.yaml` with env-appropriate replicas, resource limits, and persistence sizes
- One-command cluster provisioning and deployment via `deploy.sh` using kind
- IaC-first approach — all config in code, nothing applied manually

**Agentic AI platform design**
- Multi-turn Claude agent loop with native tool use: real web search (Tavily), code execution, file I/O
- Every agent step — LLM calls, tool invocations, token usage — captured as OpenTelemetry spans
- Async session execution with background task queue; API returns immediately with session ID
- SQLite-backed session persistence — sessions survive pod restarts, mounted via PVC in Kubernetes

**Security**
- API key middleware (`X-API-Key` header) guards all `/api/*` routes; health and metrics endpoints exempt
- Trivy security scan in CI blocks on CRITICAL/HIGH CVEs

**Observability**
- Distributed tracing: session → LLM call → tool call spans visible in Jaeger
- Prometheus metrics: session counters, tool call rates, p95 duration histograms, active session gauge
- Grafana dashboard provisioned as code via Helm values

**CI/CD**
- Three GitHub Actions workflows: Python + frontend lint/typecheck, Docker build + Trivy security scan (blocks on CRITICAL/HIGH), Helm lint + dry-run validation on every PR

## Screenshots

**Agent running — live tool call inspector**
![Agent running](docs/screenshot-prompt.png)

**Agent completed — tool call input/output + result**
![Agent result](docs/screenshot-prompt-result.png)

**Jaeger — distributed trace waterfall (session → llm_call → tool_call spans)**
![Jaeger traces](docs/screenshot-jaeger.png)

**Grafana — sessions, tool call rates, p95 duration, completion rate**
![Grafana dashboard](docs/screenshot-grafana.png)

## Architecture

```mermaid
flowchart LR
    User(["User / React UI"])

    subgraph K8s["Kubernetes (kind)"]
        direction LR
        API["FastAPI App\n2 replicas · HPA 2→5\nNetworkPolicy · probes"]
        Jaeger["Jaeger\nTraces"]
        Prometheus["Prometheus\nMetrics"]
        Grafana["Grafana\nDashboards"]
    end

    Anthropic["Anthropic API\n(Claude)"]

    User -->|HTTP + X-API-Key| API
    API -->|OTLP/gRPC| Jaeger
    API -->|scrape| Prometheus
    Prometheus -->|datasource| Grafana
    API -->|LLM + tool calls| Anthropic
```

**Stack:** FastAPI · Anthropic Claude SDK · React · Vite · TypeScript · OpenTelemetry · Jaeger · Prometheus · Grafana · Docker · Kubernetes (kind) · Helm

## Features

- **React UI** — session list with status indicators, task input, tool call inspector, live polling while agent runs
- **Real web search** — Tavily API integration; set `TAVILY_API_KEY` to activate
- **SQLite persistence** — sessions survive restarts; data volume mounted via Docker/Kubernetes
- **API key auth** — `X-API-Key` middleware; disabled when `PLATFORM_API_KEY` is unset (local dev friendly)
- **Distributed tracing** — every request and agent turn traced via OTLP → Jaeger
- **Prometheus metrics** — sessions created, tool call rates, p95 duration, active session gauge
- **Kubernetes-native** — Helm chart with HPA, NetworkPolicy, PVC, resource limits, health probes
- **Multi-environment** — `values-staging.yaml` / `values-prod.yaml` with per-env resource and scaling config
- **CI/CD** — GitHub Actions for lint/test, Docker build + Trivy security scan, Helm dry-run validation

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/sessions` | Create and run an agent session |
| `GET` | `/api/v1/sessions` | List all sessions |
| `GET` | `/api/v1/sessions/{id}` | Get session by ID |
| `DELETE` | `/api/v1/sessions/{id}` | Delete a session |
| `GET` | `/health` | Liveness check (auth exempt) |
| `GET` | `/metrics` | Prometheus metrics (auth exempt) |

### Example

```bash
# Create a session
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"task": "Search the web for the latest Claude model releases and summarise them."}'

# Poll for result
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/sessions/<session_id>
```

## Local Development

### Prerequisites

- Python 3.11+
- Docker + Docker Compose

### Setup

```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env
# Optionally set TAVILY_API_KEY for real web search
# Optionally set PLATFORM_API_KEY to enable API auth

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend (React UI)

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173 — proxies /api to port 8000
```

### Docker Compose (full stack)

```bash
docker compose up --build
```

Sessions are persisted to a named Docker volume (`agent-data`). All four services start together:

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Jaeger UI | http://localhost:16686 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin123) |

## Kubernetes Deployment

### Prerequisites

- [kind](https://kind.sigs.k8s.io/) · [kubectl](https://kubernetes.io/docs/tasks/tools/) · [helm](https://helm.sh/) · Docker

### Deploy

```bash
export ANTHROPIC_API_KEY=<your_key>
chmod +x deploy.sh
./deploy.sh
```

### Multi-environment

```bash
# Staging — 2 replicas, 512Mi memory, 1Gi persistence
helm upgrade --install agent-platform charts/agent-platform \
  -f charts/agent-platform/values-staging.yaml \
  --set anthropicApiKey=$ANTHROPIC_API_KEY

# Production — 3 replicas, 1Gi memory, 10Gi persistence, auth enabled
helm upgrade --install agent-platform charts/agent-platform \
  -f charts/agent-platform/values-prod.yaml \
  --set anthropicApiKey=$ANTHROPIC_API_KEY \
  --set platformApiKey=$PLATFORM_API_KEY
```

### Helm chart

```
charts/agent-platform/
├── Chart.yaml              # chart metadata, Prometheus + Grafana as dependencies
├── values.yaml             # base defaults
├── values-staging.yaml     # staging overrides
├── values-prod.yaml        # production overrides
└── templates/
    ├── deployment.yaml     # replicas, probes, resource limits, volume mount
    ├── service.yaml        # NodePort
    ├── hpa.yaml            # scale 2→5 on CPU/memory thresholds
    ├── networkpolicy.yaml  # ingress :8000, egress DNS + OTLP + HTTPS
    ├── secret.yaml         # ANTHROPIC_API_KEY + PLATFORM_API_KEY
    ├── pvc.yaml            # SQLite data volume (when persistence.enabled)
    └── jaeger.yaml         # Jaeger all-in-one
```

## Project Structure

```
agent-platform/
├── app/
│   ├── api/routes.py           # FastAPI routes
│   ├── agent/
│   │   ├── models.py           # AgentSession pydantic model
│   │   ├── runner.py           # Claude agent execution loop
│   │   └── store.py            # SQLite session store
│   ├── middleware/
│   │   └── auth.py             # X-API-Key middleware
│   ├── observability/
│   │   ├── tracing.py          # OTLP tracing setup
│   │   └── metrics.py          # Prometheus metrics
│   └── main.py                 # App entrypoint
├── frontend/                   # React + Vite + TypeScript UI
│   └── src/
│       ├── api.ts              # Fetch wrappers (X-API-Key aware)
│       ├── types.ts            # AgentSession, ToolCall types
│       ├── hooks/              # useSessions, useSession (polling)
│       └── components/         # SessionList, SessionDetail, StatusBadge, NewSessionForm
├── .github/workflows/
│   ├── ci.yml                  # Python lint/test + frontend lint + typecheck
│   ├── docker.yml              # Docker build + Trivy CRITICAL/HIGH scan
│   └── helm.yml                # Helm lint + template dry-run + kubeval
├── charts/agent-platform/      # Helm chart
├── k8s/manifests/              # Raw Kubernetes manifests
├── kind-config.yaml            # kind cluster definition
├── docker-compose.yml          # Local full-stack compose
├── Dockerfile
├── deploy.sh                   # One-shot k8s deploy script
└── requirements.txt
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | **Required.** Anthropic API key |
| `TAVILY_API_KEY` | — | Optional. Enables real web search via Tavily |
| `PLATFORM_API_KEY` | — | Optional. Enables `X-API-Key` auth on all `/api/*` routes |
| `DB_PATH` | `/data/sessions.db` | SQLite database path |
| `ENVIRONMENT` | `development` | Deployment environment tag |
| `OTLP_ENDPOINT` | `http://jaeger:4317` | OpenTelemetry collector endpoint |
| `MAX_TOKENS` | `4096` | Max tokens per agent turn |
