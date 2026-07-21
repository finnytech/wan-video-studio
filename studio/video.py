"""WAN 2.2 text-to-video via the official generate.py.

Driven as a subprocess so every flag is real and tested upstream:

  * --task t2v-A14B --size WxH --frame_num N --sample_steps S
  * --sample_shift / --sample_guide_scale / --sample_solver   quality knobs
  * --offload_model / --convert_model_dtype / --t5_cpu        VRAM knobs
    (all OFF by default: the RTX PRO 6000's 96 GB fits the 14B model in full,
     so we keep everything resident for maximum speed)

Frame count is clamped to WAN's 4n+1 rule and to 5-10 s (81-161 frames at 16 fps),
the coherent range for WAN 2.2. Each render is its own process, so its VRAM is
fully reclaimed by the OS before the AudioCraft stage runs — no contention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import config, runner


def _log(msg: str) -> None:
    print(f"[video] {msg}", flush=True)


def seconds_to_frames(seconds: float) -> int:
    """Map seconds to a valid WAN frame count (4n+1) at FPS, clamped to 5-10 s."""
    seconds = max(config.MIN_SECONDS, min(float(seconds), config.MAX_SECONDS))
    frames = int(round(seconds * config.FPS))
    n = round((frames - 1) / 4)
    frames = int(4 * n + 1)
    lo = 4 * round((config.MIN_SECONDS * config.FPS - 1) / 4) + 1
    hi = 4 * round((config.MAX_SECONDS * config.FPS - 1) / 4) + 1
    return max(lo, min(frames, hi))


class VideoGenerator:
    """Stateless driver around WAN 2.2 generate.py."""

    def generate(
        self,
        prompt: str,
        seconds: float,
        resolution: str,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        out_path: Optional[Path] = None,
    ) -> Path:
        from .models import ensure_video

        code_dir, weights_dir = ensure_video()

        resolution = resolution if resolution in config.RESOLUTION_MAP else config.DEFAULT_RESOLUTION
        size = config.RESOLUTION_MAP[resolution]
        num_frames = seconds_to_frames(seconds)
        steps = int(steps or config.DEFAULT_STEPS)
        seed_val = int(seed) if seed is not None else -1

        out_path = Path(out_path) if out_path else (config.OUTPUT_DIR / "video_silent.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "generate.py",
            "--task", config.VIDEO_TASK,
            "--size", size,
            "--frame_num", str(num_frames),
            "--ckpt_dir", str(weights_dir),
            "--sample_steps", str(steps),
            "--sample_shift", str(config.SAMPLE_SHIFT),
            "--sample_guide_scale", str(config.SAMPLE_GUIDE_SCALE),
            "--sample_solver", config.SAMPLE_SOLVER,
            "--base_seed", str(seed_val),
            "--save_file", str(out_path),
            "--prompt", prompt,
        ]
        if config.OFFLOAD_MODEL:
            cmd += ["--offload_model", "True"]
        if config.CONVERT_MODEL_DTYPE:
            cmd += ["--convert_model_dtype"]
        if config.T5_CPU:
            cmd += ["--t5_cpu"]
        if config.USE_PROMPT_EXTEND:
            cmd += ["--use_prompt_extend"]

        _log(f"render: {num_frames}f ({num_frames/config.FPS:.1f}s @ {config.FPS}fps), "
             f"{size}, {steps} steps, seed={seed_val}")
        runner.run(
            cmd, cwd=code_dir, stage="video",
            timeout=config.VIDEO_TIMEOUT, retries=config.STAGE_RETRIES,
        )
        return self._resolve_output(out_path, code_dir)

    def _resolve_output(self, out_path: Path, code_dir: Path) -> Path:
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        produced = []
        for d in (out_path.parent, Path(code_dir)):
            if d.exists():
                produced += [p for p in d.rglob("*.mp4") if p.stat().st_size > 0]
        if not produced:
            raise RuntimeError("generate.py produced no output video (see outputs/logs)")
        newest = max(produced, key=lambda p: p.stat().st_mtime)
        try:
            newest.replace(out_path)
            return out_path
        except Exception:  # noqa: BLE001
            return newest

    def unload(self) -> None:
        """generate.py runs as its own process; its VRAM is already released on
        exit. We sweep our own tiny process for good measure."""
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
