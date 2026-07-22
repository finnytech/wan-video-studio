"""GPU probing + OOM-recovery ladder for WAN 2.2 rendering.

The RTX PRO 6000 (96 GB) fits the 14B T2V model fully resident, so the fast path
keeps everything on-GPU. But VRAM can still get tight when:

  * another process shares the card (shared VMs / notebooks),
  * fragmentation eats usable VRAM on long sessions,
  * someone renders 720p at high step counts + long clips.

Instead of letting WAN's generate.py hard-crash with `CUDA out of memory`, we:

  1. probe free VRAM at launch and pick a sensible *starting* offload rung, and
  2. hand the runner an escalation *ladder*. When it sees an OOM in the log it
     retries the exact same render one rung stronger — WAN streams the idle
     expert / T5 encoder to pinned host RAM (block-swap over PCIe) instead of
     crashing. Each rung trades a little speed for a lot of headroom.

This is real: every flag maps to an actual, tested WAN generate.py option.
"""
from __future__ import annotations

from typing import List

from . import config


# --- VRAM probe ------------------------------------------------------------
def free_vram_gb() -> float:
    """Best-effort free VRAM in GiB. Returns a large number if we can't tell
    (so we don't needlessly force offload on a healthy 96 GB card)."""
    # Prefer NVML (accounts for *other* processes on the card, not just torch).
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        pynvml.nvmlShutdown()
        return info.free / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        if torch.cuda.is_available():
            free_b, _total_b = torch.cuda.mem_get_info(0)
            return free_b / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass
    return 1e9  # unknown -> assume plenty, let the OOM ladder catch real trouble


def total_vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            _free_b, total_b = torch.cuda.mem_get_info(0)
            return total_b / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


# --- OOM escalation ladder -------------------------------------------------
# Each rung is a set of extra flags appended to WAN's generate.py. Rungs are
# ordered from fastest (fully resident) to most VRAM-frugal (everything that can
# live on CPU does, weights converted to a smaller dtype). Monotonic: each rung
# only *adds* savings over the previous one.
#
# Rung meaning:
#   0  resident            fastest; whole 14B model on GPU
#   1  offload_model       WAN streams the idle noise-expert to pinned host RAM
#   2  + t5_cpu            T5 text encoder stays on CPU (frees ~11 GB of encoder)
#   3  + convert_dtype     model weights cast to bf16/fp16 on load (~half VRAM)
_RUNGS: List[List[str]] = [
    [],
    ["--offload_model", "True"],
    ["--offload_model", "True", "--t5_cpu"],
    ["--offload_model", "True", "--t5_cpu", "--convert_model_dtype"],
]


def _rung_from_config() -> int:
    """Lowest rung implied by the user's explicit env config, so we never start
    *weaker* than what they asked for."""
    rung = 0
    if config.OFFLOAD_MODEL:
        rung = max(rung, 1)
    if config.T5_CPU:
        rung = max(rung, 2)
    if config.CONVERT_MODEL_DTYPE:
        rung = max(rung, 3)
    return rung


def _rung_from_vram(free_gb: float) -> int:
    """Pick a starting rung from measured free VRAM. Thresholds are generous:
    the 14B model needs ~40-48 GB fully resident at 720p, T5 adds ~11 GB."""
    if free_gb >= config.VRAM_RESIDENT_GB:      # default 55 GB -> stay fully resident
        return 0
    if free_gb >= config.VRAM_OFFLOAD_GB:       # default 32 GB -> swap idle expert
        return 1
    if free_gb >= config.VRAM_T5CPU_GB:         # default 20 GB -> + T5 on CPU
        return 2
    return 3                                    # very tight -> also shrink dtype


def start_rung(log=lambda *_: None) -> int:
    """Choose the starting rung: max(user-config rung, VRAM-implied rung)."""
    if not config.AUTO_OOM:
        return _rung_from_config()
    free_gb = free_vram_gb()
    total_gb = total_vram_gb()
    vram_rung = _rung_from_vram(free_gb)
    cfg_rung = _rung_from_config()
    rung = max(vram_rung, cfg_rung)
    if free_gb < 1e8:
        log(f"VRAM probe: {free_gb:.1f} GB free / {total_gb:.0f} GB total "
            f"-> starting offload rung {rung} ({rung_name(rung)})")
    return rung


def ladder(start: int) -> List[List[str]]:
    """Escalation variants for the runner: one per attempt, from `start` down to
    the most frugal rung. The runner tries variant[i] on attempt i, climbing a
    rung whenever the previous attempt OOM'd."""
    start = max(0, min(start, len(_RUNGS) - 1))
    if not config.AUTO_OOM:
        return [_RUNGS[start]]
    return _RUNGS[start:]


def rung_name(i: int) -> str:
    return ["resident", "offload-expert", "offload+t5-cpu", "offload+t5+dtype"][
        max(0, min(i, 3))
    ]
