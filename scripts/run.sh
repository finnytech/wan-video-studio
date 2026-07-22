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
# Auto-OOM seatbelt ON by default: probe VRAM, start resident, escalate host-RAM
# offload rung-by-rung on any CUDA OOM instead of crashing. Set WAN_AUTO_OOM=0 to
# force a single fixed config (the old behaviour).
export WAN_AUTO_OOM="${WAN_AUTO_OOM:-1}"
# expandable_segments kills fragmentation OOMs; garbage_collection_threshold lets
# the allocator reclaim early instead of dying at a hard ceiling.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"
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

# --- guard WAN's optional-model imports (every launch; instant + idempotent) --
# WAN's wan/__init__.py eagerly imports i2v/s2v/ti2v/animate, several of which
# need heavy deps (decord/peft/...) that aren't in its requirements.txt. We only
# use t2v, so we wrap those imports in try/except so a missing optional dep can
# never break the render. Done here too (not just in setup) so it applies even
# when the setup sentinel would otherwise skip a reinstall.
WAN_INIT="${STUDIO_MODELS_DIR:-$ROOT/models}/Wan2.2/wan/__init__.py"
if [ -f "$WAN_INIT" ]; then
  "$PYBIN" "$ROOT/scripts/patch_wan.py" "$WAN_INIT" || true
fi

# --- give WAN's flash_attention() a real SDPA fallback (every launch) ---------
# WAN's model.py calls flash_attention() DIRECTLY, and that fn hard-asserts
# `FLASH_ATTN_2_AVAILABLE`. flash-attn has no cp312 wheel and its source build
# fails on many boxes -> without this patch EVERY render dies instantly with a
# bare AssertionError, regardless of free VRAM. This injects a
# scaled_dot_product_attention fallback so flash-attn becomes truly optional.
WAN_ATTN="${STUDIO_MODELS_DIR:-$ROOT/models}/Wan2.2/wan/modules/attention.py"
if [ -f "$WAN_ATTN" ]; then
  "$PYBIN" "$ROOT/scripts/patch_wan_attention.py" "$WAN_ATTN" || true
fi

# --- self-healing preflight -----------------------------------------------
# preflight exit codes: 0 = all present -> launch; 1 = CORE missing -> setup;
# 2 = only OPTIONAL missing (e.g. sound stack) -> setup once (sentinel), else
# launch anyway so a stubborn optional build can't block the app forever.
PF_RC=0
"$PYBIN" "$ROOT/scripts/preflight.py" || PF_RC=$?
if [ "${STUDIO_NO_AUTOSETUP:-0}" = "1" ]; then
  [ "$PF_RC" != "0" ] && echo "!! missing pieces but STUDIO_NO_AUTOSETUP=1 -> skipping. Run: bash scripts/setup.sh"
elif [ "$PF_RC" = "1" ]; then
  echo "==> core pieces missing -> running setup (installs only what's missing)"
  bash "$ROOT/scripts/setup.sh"
elif [ "$PF_RC" = "2" ]; then
  if [ ! -f "$ROOT/.setup-complete" ]; then
    echo "==> optional pieces missing (first run) -> running setup once"
    bash "$ROOT/scripts/setup.sh"
  else
    echo "==> optional pieces missing but setup already ran -> launching anyway (audio may be off)"
  fi
fi

echo "==> launching WAN Video Studio (env: $STUDIO_ENV_KIND; watch for link + token below)"
exec "$PYBIN" -m studio.app
