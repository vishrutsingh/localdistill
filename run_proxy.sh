#!/usr/bin/env bash
# localdistill — Start the LiteLLM proxy with localdistill logging.
#
# Usage:
#   cp .env.example .env   # Fill in your API keys
#   ./run_proxy.sh          # Start proxy on :8787
#   ./run_proxy.sh 9000     # Custom port
#
# Then point your tools to: http://localhost:8787/v1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8787}"

cd "$SCRIPT_DIR"

# Load API keys from .env if it exists
if [ -f ".env" ]; then
    echo "[localdistill] Loading API keys from .env..."
    set -a
    source .env
    set +a
else
    echo "[localdistill] WARNING: No .env file found. Copy .env.example to .env and add your keys."
    echo "[localdistill] The proxy will start but won't be able to route to API models."
fi

# Initialize database
echo "[localdistill] Initializing database..."
python3 -c "from db import init_db; init_db(); print('  ✓ Database ready')"

# Ensure localdistill module is importable by LiteLLM
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║  localdistill proxy starting on :$PORT       ║"
echo "  ║                                              ║"
echo "  ║  Set your tools to use:                      ║"
echo "  ║    http://localhost:$PORT/v1                   ║"
echo "  ║                                              ║"
echo "  ║  Available models: gpt-4o, gpt-4o-mini,      ║"
echo "  ║                    claude-sonnet-4, etc.      ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

exec litellm \
    --config "$SCRIPT_DIR/proxy_config.yaml" \
    --port "$PORT" \
    --num_workers 1