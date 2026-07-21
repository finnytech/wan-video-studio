#!/usr/bin/env bash
# Launch WAN Video Studio. Prints a PRIVATE public share link + access token.
#
#   bash scripts/run.sh                      # public *.gradio.live link, token-gated
#   bash scripts/run.sh --auth-only-local    # local only (127.0.0.1), no public link
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"

# RTX PRO 6000 (96 GB) friendly runtime defaults. 96 GB fits the 14B model in
# full, so offload stays OFF for maximum speed. Override before calling run.sh.
export WAN_OFFLOAD_MODEL="${WAN_OFFLOAD_MODEL:-0}"
export WAN_CONVERT_DTYPE="${WAN_CONVERT_DTYPE:-0}"
export WAN_T5_CPU="${WAN_T5_CPU:-0}"
export WAN_SAMPLE_STEPS="${WAN_SAMPLE_STEPS:-30}"
export ENABLE_TF32="${ENABLE_TF32:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "⚠️  HF_TOKEN not set — HF downloads will be rate-limited. export HF_TOKEN=***"
fi

if [ "${1:-}" = "--auth-only-local" ]; then
  export STUDIO_SHARE=0
  export STUDIO_HOST=127.0.0.1
  echo "==> local-only mode (no public link). Use an SSH tunnel to reach it."
fi

echo "==> launching WAN Video Studio (env: $STUDIO_ENV_KIND; watch for link + token below)"
exec "$PYBIN" -m studio.app
