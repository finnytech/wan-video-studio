#!/usr/bin/env bash
# Smart, idempotent installer for WAN Video Studio.
# Works on Lightning.ai Studios (single conda env, no venvs) AND on plain VMs.
# Safe to re-run: only does the work that's actually missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "==> WAN Video Studio setup in $ROOT"

PY="${PYTHON:-python3}"
VENV="$ROOT/.venv"
MODELS_DIR="${STUDIO_MODELS_DIR:-$ROOT/models}"

# --- 0. environment: venv vs current (Lightning forbids venvs) ------------
USE_VENV=1
if [ -n "${STUDIO_NO_VENV:-}" ] || [ -d /teamspace/studios ] || [ -n "${CONDA_PREFIX:-}" ]; then
  USE_VENV=0
  echo "==> detected managed/conda environment -> using the ACTIVE env (no venv)"
fi

# --- 1. system deps -------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "==> installing ffmpeg + git-lfs"
  (sudo apt-get update -y && sudo apt-get install -y ffmpeg git-lfs) \
    || echo "!! could not apt-get ffmpeg; ensure it's available"
fi
command -v git-lfs >/dev/null 2>&1 && git lfs install || true

# --- 2. python environment ------------------------------------------------
if [ "$USE_VENV" = 1 ]; then
  if [ ! -d "$VENV" ]; then
    echo "==> creating venv"
    if ! "$PY" -m venv "$VENV" 2>/dev/null; then
      echo "!! venv creation not allowed -> falling back to ACTIVE env"
    fi
  fi
fi
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"
echo "==> using python: $PYBIN  (env kind: $STUDIO_ENV_KIND)"
"$PYBIN" -m pip install --upgrade pip wheel setuptools
PIP() { "$PYBIN" -m pip "$@"; }

# --- 3. torch (CUDA 12.4) — only if missing -------------------------------
if ! "$PYBIN" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "==> installing torch (cu124)"
  PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
  echo "==> torch + CUDA already present ✓"
fi

# --- 4. studio (UI) requirements -----------------------------------------
echo "==> installing studio requirements"
PIP install -r requirements.txt

# --- 5. clone WAN 2.2 + download T2V-A14B weights (idempotent) ------------
echo "==> ensuring WAN 2.2 code + weights"
"$PYBIN" -m studio.models || true

# --- 6. install WAN's own requirements ------------------------------------
if [ -f "$MODELS_DIR/Wan2.2/requirements.txt" ]; then
  echo "==> installing WAN 2.2 requirements"
  PIP install -r "$MODELS_DIR/Wan2.2/requirements.txt" || echo "!! some WAN deps failed; check logs"
fi

# --- 7. AudioCraft (Meta AudioGen + MusicGen) -----------------------------
echo "==> installing AudioCraft"
PIP install -U audiocraft || echo "!! audiocraft install failed; check logs"

# --- 8. speed kernels (best-effort; failures never block) -----------------
echo "==> installing FlashAttention (best-effort, big speedup on Blackwell)"
PIP install flash-attn --no-build-isolation 2>/dev/null || echo "   (flash-attn optional — skipped)"

echo ""
echo "==> setup complete ✓ (env: $STUDIO_ENV_KIND)   run:  bash scripts/run.sh"
