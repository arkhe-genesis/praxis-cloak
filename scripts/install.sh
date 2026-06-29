#!/usr/bin/env bash
# scripts/install.sh — install Python dependencies only (no server start)
#
# External prerequisites (not installed by this script):
#   - Ollama:  https://ollama.com/download
#   - Node.js: https://nodejs.org/  (only needed to build the frontend UI)
#
# Usage: ./scripts/install.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
RESET='\033[0m'

info()  { echo -e "${GREEN}[cloak]${RESET} $*"; }
error() { echo -e "${RED}[cloak]${RESET} $*" >&2; }

# ── Python ────────────────────────────────────────────────────────────────────
PYTHON_BIN="python3"
VENV_DIR="$REPO_ROOT/.venv"

echo -e "\n${BOLD}Checking Python${RESET}"

if ! "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,10), "requires 3.10+"' 2>/dev/null; then
    error "Python 3.10 or higher is required."
    echo    "    Found: $("$PYTHON_BIN" --version 2>&1 || echo 'not found')"
    exit 1
fi

info "Python $("$PYTHON_BIN" --version) OK"

# ── Virtual environment ───────────────────────────────────────────────────────
echo -e "\n${BOLD}Setting up virtual environment${RESET}"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating .venv ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    info ".venv already exists — skipping creation"
fi

# ── Dependencies ──────────────────────────────────────────────────────────────
echo -e "\n${BOLD}Installing dependencies${RESET}"

info "Upgrading pip ..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip

info "Installing package (pip install -e \".[server]\") ..."
"$VENV_DIR/bin/pip" install -e ".[server]"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "Installation complete."
echo ""
echo "  To start the server (downloads + builds the on-device models on first run):"
echo "    ./scripts/run.sh"
echo ""
echo "  External prerequisites (install manually if not already present):"
echo "    Ollama:   https://ollama.com/download"
echo "    Node.js:  https://nodejs.org/  (optional; needed to build the frontend)"
