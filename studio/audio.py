"""Stable Audio Open stage — generate sound of the video's exact length, then
mux it onto the timeline.

Runs AFTER the video process has exited (clean GPU). Uses the standalone
audio_worker as a subprocess (timeout + logs + retry via the shared runner), so
all audio-model VRAM is freed on exit too.

Note: Stable Audio Open is text-conditioned (not video-conditioned like a foley
model), so the audio is generated to match the PROMPT and the exact DURATION, then
aligned to the timeline by muxing at t=0 with length == video length.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from . import config, mux, runner


def _log(msg: str) -> None:
    print(f"[audio] {msg}", flush=True)


# --- prompt shaping --------------------------------------------------------
# Stable Audio responds best to descriptive, quality-tagged prompts. We turn the
# VIDEO prompt into an audio-oriented one so effects/ambience actually match the
# scene instead of sounding generic.
_SFX_TMPL = (
    "High quality, realistic ambient sound and detailed sound effects of {p}. "
    "Immersive field recording, natural, clear stereo, no music, no speech."
)
_MUSIC_TMPL = (
    "High quality cinematic ambient score for a scene of {p}. "
    "Warm, atmospheric, emotional film soundtrack, instrumental, no vocals."
)


def _shape_prompts(video_prompt: str, mode: str, music_prompt: Optional[str]):
    """Return (sfx_prompt, music_prompt) tuned for Stable Audio Open."""
    p = video_prompt.strip().rstrip(".")
    sfx = _SFX_TMPL.format(p=p)
    if music_prompt and music_prompt.strip():
        music = f"High quality, {music_prompt.strip()}"
    else:
        music = _MUSIC_TMPL.format(p=p)
    return sfx, music


def _probe_duration(path: Path) -> float:
    import shutil
    import subprocess

    ff = shutil.which("ffprobe")
    if not ff:
        return float(config.DEFAULT_SECONDS)
    out = subprocess.run(
        [ff, "-i", str(path), "-show_entries", "format=duration",
         "-v", "quiet", "-of", "csv=p=0"],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except (TypeError, ValueError):
        return float(config.DEFAULT_SECONDS)


def add_sound(
    video_in: Path,
    prompt: str,
    mode: str,
    music_prompt: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """Generate matching audio and mux it onto `video_in`. Returns video WITH sound."""
    out_dir = Path(out_dir) if out_dir else (config.OUTPUT_DIR / "audio")
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = _probe_duration(video_in)
    wav_path = out_dir / "sound.wav"
    sfx_prompt, music_p = _shape_prompts(prompt, mode, music_prompt)

    cmd = [
        sys.executable, "-m", "studio.audio_worker",
        "--mode", mode,
        "--prompt", sfx_prompt,
        "--music_prompt", music_p,
        "--duration", f"{duration:.3f}",
        "--out", str(wav_path),
        "--model", config.AUDIO_MODEL,
        "--steps", str(config.AUDIO_STEPS),
        "--guidance", str(config.AUDIO_GUIDANCE),
        "--negative", config.AUDIO_NEG_PROMPT,
        "--music_under_db", str(config.MUSIC_UNDER_DB),
    ]

    _log(f"generating {mode} audio for {duration:.2f}s (Stable Audio Open)")
    runner.run(
        cmd, cwd=config.ROOT, stage="audio",
        timeout=config.AUDIO_TIMEOUT, retries=config.STAGE_RETRIES,
    )
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise RuntimeError("audio worker produced no wav (see outputs/logs)")

    final = mux.mux(video_in, wav_path, out_dir / "with_sound.mp4")
    _log(f"muxed: {final}")
    return final
