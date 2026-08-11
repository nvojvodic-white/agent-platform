#!/usr/bin/env bash
# One-command local demo (macOS / Linux): creates the venv, installs deps,
# then starts the API and UI via scripts/demo.py.
#
#   ./demo.sh              start everything
#   ./demo.sh --check      preflight only
#   ./demo.sh --no-ui      API only
set -euo pipefail
cd "$(dirname "$0")"

PY="venv/bin/python"
STAMP="venv/.requirements.sha"

# --- venv --------------------------------------------------------------------
if [ ! -x "$PY" ]; then
  echo "Creating venv..."
  if command -v uv >/dev/null 2>&1; then
    uv venv venv --python 3.11
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m venv venv
  else
    echo "No Python found. Install Python 3.11+ or uv (https://astral.sh/uv)." >&2
    exit 1
  fi
fi

# --- dependencies (only when requirements.txt changed) -----------------------
if command -v sha256sum >/dev/null 2>&1; then
  WANT=$(sha256sum requirements.txt | cut -d' ' -f1)
else
  WANT=$(shasum -a 256 requirements.txt | cut -d' ' -f1)   # macOS
fi
HAVE=$(cat "$STAMP" 2>/dev/null || echo "")

if [ "$WANT" != "$HAVE" ]; then
  echo "Installing dependencies (first run, or requirements.txt changed)..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" -r requirements.txt
  else
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt
  fi
  echo "$WANT" > "$STAMP"
fi

exec "$PY" scripts/demo.py "$@"
