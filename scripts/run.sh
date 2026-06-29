#!/usr/bin/env bash
# scripts/run.sh — one-command launcher for Cloak
# Usage: ./scripts/run.sh
set -euo pipefail

# Resolve repo root relative to this script
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Colours ──────────────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

info()    { echo -e "${GREEN}[cloak]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[cloak]${RESET} $*"; }
error()   { echo -e "${RED}[cloak]${RESET} $*" >&2; }
heading() { echo -e "\n${BOLD}$*${RESET}"; }

# ── Step 1: Check Ollama ──────────────────────────────────────────────────────
heading "1/5  Checking Ollama"

if ! command -v ollama &>/dev/null; then
    error "Ollama is not installed."
    echo    "    Install it from: https://ollama.com/download"
    exit 1
fi

if ! ollama list &>/dev/null; then
    error "Ollama is installed but not reachable (is the daemon running?)."
    echo    "    Start it with: ollama serve"
    echo    "    Or open the Ollama app on macOS."
    exit 1
fi

info "Ollama OK"

# ── Step 2: Set up models ─────────────────────────────────────────────────────
heading "2/5  Setting up models (first run: ~3.8 GB download, subsequent runs: instant)"

# Download the GGUF from HuggingFace and build the Ollama model locally. We build
# locally (rather than `ollama pull`) so the relevance model gets its chat template
# from the bundled Modelfile — no Ollama account required.
HF_BASE="https://huggingface.co/praxis-nation"
MODELS_DIR="$REPO_ROOT/models"

ensure_model() {
    local name="$1" repo="$2" gguf="$3" modelfile="$4"
    if ollama list 2>/dev/null | grep -q "^${name}"; then
        info "  ${name} already present — skipping"
        return
    fi
    if [ ! -f "$MODELS_DIR/$gguf" ]; then
        info "  Downloading ${gguf} from HuggingFace ..."
        curl -fL --progress-bar "$HF_BASE/$repo/resolve/main/$gguf" -o "$MODELS_DIR/$gguf"
    fi
    info "  Building ${name} into Ollama ..."
    # Build from MODELS_DIR so the Modelfile's relative `FROM ./*.gguf` resolves.
    ( cd "$MODELS_DIR" && ollama create "$name" -f "$modelfile" )
}

ensure_model "praxis/spanfinder-3b" "spanfinder-3b" "spanfinder-3b-q4_k_m.gguf" "$MODELS_DIR/Modelfile.spanfinder"
ensure_model "praxis/relevance-3b"  "relevance-3b"  "relevance-3b-q4_k_m.gguf"  "$MODELS_DIR/Modelfile.relevance"

# ── Step 3: Python virtual environment ───────────────────────────────────────
heading "3/5  Setting up Python environment"

PYTHON_BIN="python3"
VENV_DIR="$REPO_ROOT/.venv"

# Require Python 3.10+
if ! "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,10), "requires Python 3.10+"' 2>/dev/null; then
    error "Python 3.10 or higher is required."
    echo    "    Found: $("$PYTHON_BIN" --version 2>&1 || echo 'not found')"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment at .venv ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

info "Installing/updating Python dependencies ..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -e ".[server]"

# ── Step 4: Build frontend ────────────────────────────────────────────────────
heading "4/5  Frontend"

FRONTEND_DIR="$REPO_ROOT/app/frontend"
DIST_DIR="$FRONTEND_DIR/dist"

if [ ! -d "$DIST_DIR" ]; then
    if command -v npm &>/dev/null; then
        info "Building frontend (first run only) ..."
        (cd "$FRONTEND_DIR" && npm install && npm run build)
    else
        warn "npm not found — skipping frontend build."
        warn "The UI will not be available. Install Node.js from https://nodejs.org/"
    fi
else
    info "Frontend dist already built — skipping"
fi

# ── Step 5: Start server ──────────────────────────────────────────────────────
heading "5/5  Starting Cloak"

HOST="127.0.0.1"
PORT="8765"
URL="http://${HOST}:${PORT}"

info "Server starting at ${URL}"
info "Press Ctrl+C to stop."

# Open browser after a short delay (best-effort; ignore errors)
(sleep 2 && (
    if command -v open &>/dev/null; then
        open "$URL"
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$URL"
    fi
)) &>/dev/null &

exec "$VENV_DIR/bin/python" -m uvicorn backend.main:app \
    --app-dir "$REPO_ROOT/app" \
    --host "$HOST" \
    --port "$PORT"
