"""Robust subprocess runner shared by the video and Foley stages.

Features:
  * hard timeout (a wedged stage can't pin the GPU forever)
  * full stdout/stderr streamed to a per-run log file for post-mortem
  * one automatic retry on transient failure (OOM hiccup, kernel autotune)
  * a tuned environment (TF32, CPU threads, expandable CUDA allocator)
  * a fresh, empty CUDA cache is guaranteed because each stage is its own
    process — when it exits, the OS reclaims 100% of its VRAM.
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from . import config


def _log(msg: str) -> None:
    print(f"[runner] {msg}", flush=True)


# CUDA OOM / cuDNN-alloc failure signatures we auto-recover from.
_OOM_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "cuda_error_out_of_memory",
    "failed to allocate",
    "torch.cuda.outofmemoryerror",
)


def _looks_like_oom(tail: str) -> bool:
    low = tail.lower()
    return any(m in low for m in _OOM_MARKERS)


def child_env(extra: Optional[dict] = None) -> dict:
    """Environment for GPU subprocesses: perf knobs + threading + allocator."""
    env = dict(os.environ)
    # expandable_segments crushes fragmentation OOMs; garbage_collection_threshold
    # lets the allocator reclaim early instead of dying at a hard ceiling.
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.9",
    )
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["OMP_NUM_THREADS"] = str(config.CPU_THREADS)
    env["MKL_NUM_THREADS"] = str(config.CPU_THREADS)
    if config.ENABLE_TF32:
        # Honoured by frameworks that read these; harmless otherwise.
        env.setdefault("NVIDIA_TF32_OVERRIDE", "1")
        env.setdefault("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def run(
    cmd: Sequence[str],
    cwd: Path | str,
    stage: str,
    timeout: int,
    retries: Optional[int] = None,
    env_extra: Optional[dict] = None,
    oom_variants: Optional[Sequence[Sequence[str]]] = None,
    on_variant=None,
) -> Path:
    """Run `cmd`, tee output to a log file, enforce `timeout`, retry on failure.

    OOM auto-recovery: if `oom_variants` is given, attempt *i* runs
    `cmd + oom_variants[i]`. On a CUDA-OOM failure we climb to the next, stronger
    variant (more host-RAM offload) instead of crashing. Non-OOM failures use the
    ordinary transient retry within the current variant.

    Returns the log file path. Raises RuntimeError with a readable tail on failure.
    """
    retries = config.STAGE_RETRIES if retries is None else retries
    env = child_env(env_extra)
    variants = [list(v) for v in (oom_variants or [[]])]
    last_tail = ""
    attempt = 0

    vi = 0
    transient_left = retries
    while vi < len(variants):
        extra = variants[vi]
        full_cmd = list(map(str, cmd)) + list(map(str, extra))
        if on_variant is not None:
            try:
                on_variant(vi, extra)
            except Exception:  # noqa: BLE001
                pass
        attempt += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = config.LOG_DIR / f"{stage}_{stamp}_v{vi}.log"
        tag = f"variant {vi + 1}/{len(variants)}" + (f" ({' '.join(extra)})" if extra else " (resident)")
        _log(f"{stage}: {tag} -> log {log_path.name}")
        t0 = time.time()
        try:
            with open(log_path, "w") as lf:
                lf.write("CMD: " + " ".join(full_cmd) + "\n\n")
                lf.flush()
                subprocess.run(
                    full_cmd,
                    cwd=str(cwd),
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=True,
                )
            _log(f"{stage}: OK in {time.time() - t0:.1f}s ({tag})")
            return log_path
        except subprocess.TimeoutExpired:
            last_tail = _tail(log_path)
            _log(f"{stage}: TIMEOUT after {timeout}s")
            vi += 1
            continue
        except subprocess.CalledProcessError as e:
            last_tail = _tail(log_path)
            _log(f"{stage}: exit {e.returncode} after {time.time() - t0:.1f}s")

        if _looks_like_oom(last_tail) and vi + 1 < len(variants):
            nxt = variants[vi + 1]
            _log(f"{stage}: CUDA OOM -> escalating offload to variant {vi + 2} "
                 f"({' '.join(nxt) or 'resident'}) and retrying (spilling to host RAM)")
            _clear_cuda()
            time.sleep(3)
            vi += 1
            continue

        if transient_left > 0:
            transient_left -= 1
            _log(f"{stage}: transient failure -> retry same variant after cooldown "
                 f"({transient_left} left)")
            _clear_cuda()
            time.sleep(3)
            continue

        # exhausted transient retries on this variant; try a stronger one if any
        if vi + 1 < len(variants):
            _log(f"{stage}: escalating to next offload variant")
            vi += 1
            continue
        break

    raise RuntimeError(
        f"{stage} failed after {attempt} attempt(s) across {len(variants)} "
        f"offload variant(s).\n--- log tail ---\n{last_tail}"
    )


def _clear_cuda() -> None:
    """Best-effort VRAM sweep in *our* process between attempts (the child already
    freed its own on exit, but this trims our allocator too)."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _tail(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:  # noqa: BLE001
        return "(no log captured)"
