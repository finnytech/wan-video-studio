"""Central configuration for WAN Video Studio.

Text -> WAN 2.2 (T2V-A14B) video -> Stable Audio Open (SFX + music) sound
-> muxed MP4 with a download button, behind a one-time token gate.

Tuned for a single RTX PRO 6000 (Blackwell, 96 GB VRAM, 48 vCPU, ~500 TFLOPS).
With 96 GB the 14B model fits WITHOUT offload, so we keep offload off for max speed.

Everything is env-overridable so you never touch code on the VM.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_str(name: str):
    return os.environ.get(name)


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("STUDIO_MODELS_DIR", ROOT / "models"))
OUTPUT_DIR = Path(os.environ.get("STUDIO_OUTPUT_DIR", ROOT / "outputs"))
CACHE_DIR = Path(os.environ.get("STUDIO_CACHE_DIR", ROOT / ".cache"))
VIDEOS_DIR = Path(os.environ.get("STUDIO_VIDEOS_DIR", ROOT / "outputs" / "videos"))
LOG_DIR = Path(os.environ.get("STUDIO_LOG_DIR", ROOT / "outputs" / "logs"))
for _d in (MODELS_DIR, OUTPUT_DIR, CACHE_DIR, VIDEOS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- WAN 2.2 (video) -------------------------------------------------------
VIDEO_REPO_URL = os.environ.get("VIDEO_REPO_URL", "https://github.com/Wan-Video/Wan2.2.git")
VIDEO_CODE_DIR = MODELS_DIR / "Wan2.2"                       # code checkout (generate.py)
# Default = official T2V-A14B. For an UNCENSORED community finetune, just point
# this at the HF repo id (or drop weights in models/Wan2.2-T2V-A14B-weights).
VIDEO_MODEL_REPO = os.environ.get("VIDEO_MODEL_REPO", "Wan-AI/Wan2.2-T2V-A14B")
VIDEO_WEIGHTS_DIR = Path(os.environ.get("VIDEO_WEIGHTS_DIR", MODELS_DIR / "Wan2.2-T2V-A14B-weights"))
VIDEO_TASK = os.environ.get("WAN_TASK", "t2v-A14B")

# Faster HF downloads (Rust multi-threaded transfer).
HF_FAST = _flag("STUDIO_HF_FAST", True)


# --- Generation defaults ---------------------------------------------------
# WAN 2.2 renders at 16 fps. Sweet spot is 5-10 s (81-161 frames); beyond ~10 s
# motion coherence degrades (prompt drift / looping), so we cap there.
FPS = _int("STUDIO_FPS", 16)
MIN_SECONDS = _int("STUDIO_MIN_SECONDS", 5)
MAX_SECONDS = _int("STUDIO_MAX_SECONDS", 10)
DEFAULT_SECONDS = _int("STUDIO_DEFAULT_SECONDS", 5)

# WAN's SIZE_CONFIGS uses "W*H" strings. T2V-A14B supports 480p and 720p natively.
RESOLUTION_MAP = {
    "480p": "832*480",
    "720p": "1280*720",
}
# 720p is the realism sweet spot for T2V-A14B and fits easily in 96 GB.
DEFAULT_RESOLUTION = os.environ.get("STUDIO_DEFAULT_RES", "720p")

# Sampling. Official default is 40-50 steps. On the RTX PRO 6000 the 96 GB budget
# lets us run high step counts at full precision -> we default to 40 for maximum
# realism/detail (film-action look). Drop to 30 for faster drafts.
DEFAULT_STEPS = _int("WAN_SAMPLE_STEPS", 40)
# sample_shift shapes the noise schedule; 5.0 is WAN's filmic default. Guidance 5.0
# keeps strong prompt adherence without the over-sharpened/plastic look.
SAMPLE_SHIFT = _float("WAN_SAMPLE_SHIFT", 5.0)
SAMPLE_GUIDE_SCALE = _float("WAN_GUIDE_SCALE", 5.0)
SAMPLE_SOLVER = os.environ.get("WAN_SAMPLE_SOLVER", "unipc")

# Cinematic prompt enhancement (photoreal / film-action). See prompt_enhance.py.
ENHANCE_PROMPT = _flag("STUDIO_ENHANCE", True)


# --- Speed / VRAM (RTX PRO 6000, 96 GB) ------------------------------------
# 96 GB fits the 14B model without offload -> keep offload OFF for max speed.
OFFLOAD_MODEL = _flag("WAN_OFFLOAD_MODEL", False)
CONVERT_MODEL_DTYPE = _flag("WAN_CONVERT_DTYPE", False)  # only needed when VRAM tight
T5_CPU = _flag("WAN_T5_CPU", False)                      # keep T5 on GPU (fast) at 96GB
# TF32 + cuDNN autotune: free throughput on Blackwell, no quality hit.
ENABLE_TF32 = _flag("ENABLE_TF32", True)
# 48 vCPU -> give data/VAE work plenty of threads, leave a few for OS/UI.
CPU_THREADS = _int("STUDIO_CPU_THREADS", 40)
# Optional prompt extension (needs DashScope key or local Qwen) -> off, self-contained.
USE_PROMPT_EXTEND = _flag("WAN_PROMPT_EXTEND", False)

# --- Auto-OOM recovery (never hard-crash on CUDA OOM) ----------------------
# When on, we probe free VRAM at launch, start on the fastest rung that fits, and
# escalate host-RAM offload one rung at a time on any CUDA-OOM instead of dying.
# The RTX PRO 6000 normally never leaves rung 0 (resident); this is the seatbelt.
AUTO_OOM = _flag("WAN_AUTO_OOM", True)
# Free-VRAM thresholds (GiB) for picking the *starting* rung. Generous defaults:
#   >= RESIDENT_GB -> rung 0 (all on GPU, fastest)
#   >= OFFLOAD_GB  -> rung 1 (stream idle expert to pinned host RAM)
#   >= T5CPU_GB    -> rung 2 (+ T5 text encoder on CPU)
#   below that     -> rung 3 (+ cast weights to bf16/fp16 on load)
VRAM_RESIDENT_GB = _float("WAN_VRAM_RESIDENT_GB", 55.0)
VRAM_OFFLOAD_GB = _float("WAN_VRAM_OFFLOAD_GB", 32.0)
VRAM_T5CPU_GB = _float("WAN_VRAM_T5CPU_GB", 20.0)


# --- Stable Audio Open (sound) ---------------------------------------------
# ONE diffusers pipeline (StableAudioPipeline) turns TEXT into SFX *and* music,
# stereo 44.1 kHz, up to 47 s. We generate audio of the video's exact length and
# mux it onto the timeline. No torch-version conflicts (unlike audiocraft).
# GATED model: accept the license once + set HF_TOKEN.
#   https://huggingface.co/stabilityai/stable-audio-open-1.0
AUDIO_MODEL = os.environ.get("AUDIO_MODEL", "stabilityai/stable-audio-open-1.0")
# Default sound mode: "sfx" (effects/ambience), "music" (score), "both" (mixed).
DEFAULT_AUDIO_MODE = os.environ.get("STUDIO_AUDIO_MODE", "sfx")
# Diffusion steps: higher = better/slower. 100 is a solid default for 5-10 s.
AUDIO_STEPS = _int("STUDIO_AUDIO_STEPS", 100)
# Guidance scale: prompt adherence. 7.0 is Stable Audio's sweet spot.
AUDIO_GUIDANCE = _float("STUDIO_AUDIO_GUIDANCE", 7.0)
# Negative prompt markedly improves quality (per Stable Audio docs).
AUDIO_NEG_PROMPT = os.environ.get(
    "STUDIO_AUDIO_NEG", "low quality, average quality, distorted, harsh noise, clipping"
)
# When mixing "both", the music sits under the SFX by this gain (dB).
MUSIC_UNDER_DB = _float("STUDIO_MUSIC_UNDER_DB", -8.0)
AUDIO_TIMEOUT = _int("STUDIO_AUDIO_TIMEOUT", 900)


# --- Robustness ------------------------------------------------------------
VIDEO_TIMEOUT = _int("STUDIO_VIDEO_TIMEOUT", 3600)
STAGE_RETRIES = _int("STUDIO_STAGE_RETRIES", 1)


# --- Server / auth ---------------------------------------------------------
SERVER_NAME = os.environ.get("STUDIO_HOST", "0.0.0.0")
SERVER_PORT = _int("STUDIO_PORT", 7860)
PUBLIC_SHARE = _flag("STUDIO_SHARE", True)
PINNED_TOKEN = _env_str("STUDIO_TOKEN") or None
MAX_CONCURRENCY = _int("STUDIO_MAX_CONCURRENCY", 1)


def available_resolutions():
    return list(RESOLUTION_MAP.keys())
