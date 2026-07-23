#!/usr/bin/env bash
# localdistill — Train your model
# Runs the trainer container with GPU support.
#
# Usage:
#   ./train.sh                        # Train with defaults
#   ./train.sh --min-score 0.7        # Only high-quality conversations
#   ./train.sh --base mistralai/Mistral-7B-Instruct-v0.3
#   ./train.sh --dry-run              # Export dataset only, no training

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║         LOCALDISTILL TRAINING            ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# Check for API keys
source .env 2>/dev/null || true

if ! nvidia-smi &>/dev/null; then
  echo "  No GPU detected. Use cloud training:"
  echo "    ./scripts/cloud_train.sh runpod"
  exit 1
fi

echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Available for training: $(python3 -c "from proxy.db import get_db; db=get_db(); r=db.execute('SELECT COUNT(*) as cnt FROM curated_training WHERE used_in_training=0').fetchone(); print(r['cnt']); db.close()" 2>/dev/null || echo "?") conversations"
echo ""

docker compose --profile training run --rm trainer "$@"

echo ""
echo "  Training complete. Adapter saved to ./adapters/"