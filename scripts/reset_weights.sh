#!/usr/bin/env bash
# Delete the WAN 2.2 video weights (and HF cache) and re-download them cleanly,
# with hf_transfer for max speed.
#
#   bash scripts/reset_weights.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"

MODELS_DIR="${STUDIO_MODELS_DIR:-$ROOT/models}"

echo "==> removing WAN 2.2 weights"
rm -rf "$MODELS_DIR/Wan2.2-T2V-A14B-weights"
echo "==> clearing HuggingFace hub cache for Wan2.2-T2V-A14B"
rm -rf "$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B" 2>/dev/null || true

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "⚠️  HF_TOKEN not set — set it for faster, un-throttled downloads: export HF_TOKEN=***"
fi

echo "==> re-downloading WAN 2.2 weights"
"$PYBIN" -m studio.models
echo "==> done ✓  disk usage:"
du -sh "$MODELS_DIR"/* 2>/dev/null || true
