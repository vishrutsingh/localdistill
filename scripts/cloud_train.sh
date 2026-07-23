#!/usr/bin/env bash
# localdistill — Cloud GPU training
#
# Usage:
#   ./scripts/cloud_train.sh runpod    # Train on RunPod
#   ./scripts/cloud_train.sh vastai    # Train on Vast.ai
#   ./scripts/cloud_train.sh lambda    # Train on Lambda Labs
#
# Prerequisites:
#   - Provider API key set in .env
#   - Provider CLI installed (runpodctl, vastai, etc.)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

PROVIDER="${1:-}"
source .env 2>/dev/null || true

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║      LOCALDISTILL CLOUD TRAINING         ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

if [[ -z "$PROVIDER" ]]; then
  echo "  Usage: ./scripts/cloud_train.sh <provider>"
  echo ""
  echo "  Supported providers:"
  echo "    runpod  — RunPod GPU cloud"
  echo "    vastai  — Vast.ai GPU marketplace"
  echo "    lambda  — Lambda Labs GPU cloud"
  exit 1
fi

# ── Export dataset ──
echo "  [1/3] Exporting training dataset..."
mkdir -p /tmp/localdistill-cloud
curl -s -o /tmp/localdistill-cloud/train.jsonl "http://localhost:8000/api/export?format=chatml"
count=$(wc -l < /tmp/localdistill-cloud/train.jsonl)
echo "  ✓ Exported $count examples"

# ── Upload to cloud ──
echo ""
echo "  [2/3] Launching $PROVIDER GPU instance..."

case "$PROVIDER" in
  runpod)
    echo "  → Install runpodctl: https://github.com/runpod/runpodctl"
    echo "  → Upload dataset: runpodctl send train.jsonl"
    echo "  → Start pod with image: unsloth/unsloth:latest"
    echo "  → Run: python train.py /workspace/train.jsonl"
    echo "  → Download adapter: runpodctl receive adapters/"
    echo ""
    echo "  ⚠  Cloud training requires provider setup. See docs for details."
    ;;
  vastai)
    echo "  → Install vastai CLI: pip install vastai"
    echo "  → Search GPU: vastai search offers 'reliability>0.99 rented=False'"
    echo "  → Create instance with unsloth image"
    echo "  → SCP dataset and run train.py"
    echo ""
    echo "  ⚠  Cloud training requires provider setup. See docs for details."
    ;;
  lambda)
    echo "  → Install lambda CLI: pip install lambda-cli"
    echo "  → Launch instance: lambda-cli launch --gpu a100 --image unsloth/unsloth"
    echo "  → SCP dataset and run train.py"
    echo ""
    echo "  ⚠  Cloud training requires provider setup. See docs for details."
    ;;
  *)
    echo "  Unknown provider: $PROVIDER"
    exit 1
    ;;
esac

echo "  [3/3] Done. Pull adapter back to ./adapters/ after training completes."