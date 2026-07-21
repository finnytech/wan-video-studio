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


def child_env(extra: Optional[dict] = None) -> dict:
    """Environment for GPU subprocesses: perf knobs + threading + allocator."""
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
) -> Path:
    """Run `cmd`, tee output to a log file, enforce `timeout`, retry on failure.

    Returns the log file path. Raises RuntimeError with a readable tail on failure.
    """
    retries = config.STAGE_RETRIES if retries is None else retries
    env = child_env(env_extra)
    attempt = 0
    last_tail = ""

    while attempt <= retries:
        attempt += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = config.LOG_DIR / f"{stage}_{stamp}_try{attempt}.log"
        _log(f"{stage}: attempt {attempt}/{retries + 1} -> log {log_path.name}")
        t0 = time.time()
        try:
            with open(log_path, "w") as lf:
                lf.write("CMD: " + " ".join(map(str, cmd)) + "\n\n")
                lf.flush()
                proc = subprocess.run(
                    list(map(str, cmd)),
                    cwd=str(cwd),
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=True,
                )
            _log(f"{stage}: OK in {time.time() - t0:.1f}s")
            return log_path
        except subprocess.TimeoutExpired:
            last_tail = _tail(log_path)
            _log(f"{stage}: TIMEOUT after {timeout}s")
        except subprocess.CalledProcessError as e:
            last_tail = _tail(log_path)
            _log(f"{stage}: exit {e.returncode} after {time.time() - t0:.1f}s")
        if attempt <= retries:
            _log(f"{stage}: retrying after cooldown ...")
            time.sleep(3)

    raise RuntimeError(f"{stage} failed after {retries + 1} attempt(s).\n--- log tail ---\n{last_tail}")


def _tail(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:  # noqa: BLE001
        return "(no log captured)"
