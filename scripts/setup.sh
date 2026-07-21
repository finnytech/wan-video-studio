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
# ffmpeg binary (mux) + git-lfs + ffmpeg DEV headers & pkg-config. The dev
# headers are required to build PyAV (av==11.0.0), a hard dep of audiocraft;
# without them the audiocraft wheel build fails with "libavdevice not found".
APT_PKGS=""
command -v ffmpeg   >/dev/null 2>&1 || APT_PKGS="$APT_PKGS ffmpeg"
command -v git-lfs  >/dev/null 2>&1 || APT_PKGS="$APT_PKGS git-lfs"
command -v pkg-config >/dev/null 2>&1 || APT_PKGS="$APT_PKGS pkg-config"
# Always ensure the -dev headers (cheap if already installed).
APT_PKGS="$APT_PKGS libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev"
if [ -n "$APT_PKGS" ]; then
  echo "==> installing system deps:$APT_PKGS"
  # shellcheck disable=SC2086
  (sudo apt-get update -y && sudo apt-get install -y $APT_PKGS) \
    || echo "!! apt-get failed for some packages; audiocraft build may fail (video still works)"
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
# IMPORTANT: WAN's requirements.txt pins flash_attn, which builds from source and
# needs torch already importable in the build env. Under pip build isolation it
# fails ("No module named 'torch'") and that ONE failure aborts the whole -r
# install, leaving diffusers/transformers/opencv uninstalled. So we strip
# flash_attn out here and install it separately, best-effort, in step 8.
if [ -f "$MODELS_DIR/Wan2.2/requirements.txt" ]; then
  echo "==> installing WAN 2.2 requirements (flash_attn handled separately)"
  WAN_REQS_FILTERED="$(mktemp)"
  grep -viE '^\s*flash[-_]attn' "$MODELS_DIR/Wan2.2/requirements.txt" > "$WAN_REQS_FILTERED" || true
  PIP install -r "$WAN_REQS_FILTERED" || echo "!! some WAN deps failed; check logs"
  rm -f "$WAN_REQS_FILTERED"
fi

# --- 6b. guard WAN imports + smoke-test the t2v chain ---------------------
# WAN's __init__ eagerly imports i2v/s2v/ti2v/animate (needing decord/peft/...
# which aren't in its requirements.txt and often lack cp312 wheels). We only use
# t2v, so guard those imports, then ACTUALLY try `import wan` here -- catching
# any remaining missing dep now, at setup time, instead of at expensive GPU
# render time. Whatever is still missing gets auto-installed (bounded loop).
WAN_INIT="$MODELS_DIR/Wan2.2/wan/__init__.py"
if [ -f "$WAN_INIT" ]; then
  echo "==> guarding WAN optional-model imports (t2v-only; skips decord/peft/etc.)"
  "$PYBIN" "$ROOT/scripts/patch_wan.py" "$WAN_INIT" || true

  echo "==> smoke-testing 'import wan' + auto-healing any missing deps"
  _wan_missing() {
    ( cd "$MODELS_DIR/Wan2.2" && "$PYBIN" - <<'PY' 2>/dev/null
import importlib, sys
try:
    wan = importlib.import_module("wan")
    if getattr(wan, "WanT2V", None) is None:
        importlib.import_module("wan.text2video")  # force the hard t2v import
    print("OK")
except ModuleNotFoundError as e:
    print((e.name or "").split(".")[0])
except Exception:
    print("OK")  # non-missing import-time error: stop looping, don't mask it
PY
)
  }
  for _i in 1 2 3 4 5 6 7 8; do
    miss="$(_wan_missing)"
    [ "$miss" = "OK" ] && { echo "   import wan OK ✓"; break; }
    [ -z "$miss" ] && { echo "   import wan: unknown import error (see run log)"; break; }
    case "$miss" in
      cv2)     pkg="opencv-python" ;;
      PIL)     pkg="Pillow" ;;
      sklearn) pkg="scikit-learn" ;;
      decord)  pkg="eva-decord" ;;   # decord has no cp312 wheel; eva-decord fork does
      yaml)    pkg="PyYAML" ;;
      *)       pkg="$miss" ;;
    esac
    echo "   'import wan' needs '$miss' -> installing $pkg"
    PIP install "$pkg" || { echo "   !! could not install $pkg (video may fail; see logs)"; break; }
  done
fi

# --- 7. AudioCraft (Meta AudioGen + MusicGen) -----------------------------
# Optional sound stage. Video always works without it.
#
# WHY --no-deps + a pinned version: audiocraft's loose pins make pip BACKTRACK
# from 1.3.0 down to 1.1.0, which requires spacy==3.5.2 — a version with NO
# cp312 wheel that then fails to compile (thinc / Cython<3 vs modern NumPy).
# Pinning 1.3.0 + --no-deps sidesteps that entirely; we then install only the
# runtime deps AudioGen/MusicGen actually need, all of which ship cp312 wheels
# (no source builds, no compile hell).
if "$PYBIN" -c "import audiocraft" 2>/dev/null; then
  echo "==> AudioCraft already present ✓"
else
  echo "==> installing AudioCraft (pinned 1.3.0, --no-deps + wheel-only runtime deps)"
  if PIP install "audiocraft==1.3.0" --no-deps; then
    PIP install \
      "av>=12.0.0" einops encodec julius num2words omegaconf \
      "hydra-core>=1.1" hydra_colorlog "spacy>=3.7,<3.9" sentencepiece flashy \
      librosa soundfile torchmetrics \
      || echo "!! some audiocraft runtime deps failed (sound stage may be off)"
  else
    echo "!! audiocraft install failed (video still works; sound stage disabled)"
  fi
  "$PYBIN" -c "import audiocraft; print('   audiocraft import OK ✓')" 2>/dev/null \
    || echo "   audiocraft not importable yet — video still works, audio off"
fi

# --- 8. speed kernels (best-effort; failures never block) -----------------
# flash-attn needs torch visible during build -> --no-build-isolation. Big
# speedup on Blackwell but 100% optional; WAN falls back to SDPA without it.
if "$PYBIN" -c "import flash_attn" 2>/dev/null; then
  echo "==> FlashAttention already present ✓"
else
  echo "==> installing FlashAttention (best-effort, big speedup on Blackwell)"
  PIP install flash-attn --no-build-isolation 2>/dev/null || echo "   (flash-attn optional — skipped)"
fi

# Sentinel so run.sh won't loop re-running setup for a stubborn optional dep.
touch "$ROOT/.setup-complete"

echo ""
echo "==> setup complete ✓ (env: $STUDIO_ENV_KIND)   run:  bash scripts/run.sh"
