#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

CLUSTER_NAME="agent-platform"
NAMESPACE="agent-platform"
RELEASE="agent-platform"
CHART="./charts/agent-platform"

# ── 1. Cluster ──────────────────────────────────────────────────────────────
if ! kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
  echo ">> Creating kind cluster..."
  kind create cluster --config kind-config.yaml
else
  echo ">> Cluster '${CLUSTER_NAME}' already exists, skipping."
fi

kubectl cluster-info --context "kind-${CLUSTER_NAME}"

# ── 2. Build & load image ───────────────────────────────────────────────────
echo ">> Building Docker image..."
docker build -t agent-platform:latest .

echo ">> Loading image into kind..."
kind load docker-image agent-platform:latest --name "${CLUSTER_NAME}"

# ── 3. Namespace ────────────────────────────────────────────────────────────
kubectl apply -f k8s/manifests/namespace.yaml

# ── 4. Helm deps & deploy ───────────────────────────────────────────────────
echo ">> Updating Helm dependencies..."
helm dependency update "${CHART}"

ANTHROPIC_KEY="${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY env var is required}"

echo ">> Deploying Helm release..."
helm upgrade --install "${RELEASE}" "${CHART}" \
  --namespace "${NAMESPACE}" \
  --set anthropicApiKey="${ANTHROPIC_KEY}" \
  --wait \
  --timeout 5m

# ── 5. Status ───────────────────────────────────────────────────────────────
echo ""
echo ">> Deployment complete. Pod status:"
kubectl get pods -n "${NAMESPACE}"

echo ""
echo ">> Services:"
kubectl get svc -n "${NAMESPACE}"

echo ""
echo "Endpoints:"
echo "  API:      http://localhost:8000"
echo "  Jaeger:   http://localhost:16686"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:  http://localhost:3000  (admin / admin123)"
