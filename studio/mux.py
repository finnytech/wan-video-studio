"""ffmpeg mux helpers.

Foley's infer.py normally returns a video that already contains the generated
audio. But we keep an explicit, robust mux path here so that:
  * if Foley ever returns audio-only, we still combine cleanly, and
  * we can guarantee the audio is trimmed/padded to exactly the video length
    (timeline alignment -> feels like native sound).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import config


def _log(msg: str) -> None:
    print(f"[mux] {msg}", flush=True)


def mux(video_no_audio: Path, audio: Path, out_path: Path | None = None) -> Path:
    out_path = out_path or (config.OUTPUT_DIR / "final.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_no_audio),
        "-i", str(audio),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        # -shortest keeps A/V aligned to the shorter stream (the video length).
        "-shortest",
        "-map", "0:v:0", "-map", "1:a:0",
        str(out_path),
    ]
    _log(" ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def normalize_final(src: Path, out_path: Path | None = None) -> Path:
    """Re-encode to a widely-compatible MP4 (H.264 + AAC, faststart) for the
    browser download button."""
    out_path = out_path or (config.OUTPUT_DIR / "download.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    _log("normalize -> " + str(out_path))
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
