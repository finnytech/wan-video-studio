"""Model presence checks + downloads.

Video = WAN 2.2 official code (generate.py) + T2V-A14B weights from HF.
Sound = Stable Audio Open via diffusers StableAudioPipeline; weights auto-download
        on first use (gated: needs HF_TOKEN + accepted license), cached under HF cache.

Idempotent: nothing is re-downloaded if it's already on disk.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import config


def _log(msg: str) -> None:
    print(f"[models] {msg}", flush=True)


def _hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _enable_fast_transfer() -> None:
    if not config.HF_FAST:
        return
    try:
        import hf_transfer  # noqa: F401

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        _log("hf_transfer enabled (fast downloads) ✓")
    except Exception:  # noqa: BLE001
        _log("hf_transfer not installed; `pip install hf_transfer` for max speed")


def _warn_token() -> None:
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        _log("⚠️  HF_TOKEN not set — HF downloads are rate-limited. export HF_TOKEN=***")


def _has_weights(path: Path) -> bool:
    if not path.exists():
        return False
    for pat in ("*.safetensors", "*.pth", "*.bin", "*.ckpt"):
        if any(path.rglob(pat)):
            return True
    return False


def _clone(url: str, dest: Path, marker: str) -> None:
    if (dest / marker).exists():
        _log(f"code present: {dest.name} ✓")
        return
    _log(f"cloning {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def ensure_video() -> tuple[Path, Path]:
    """Ensure WAN 2.2 code + T2V-A14B weights. Returns (code_dir, weights_dir)."""
    _enable_fast_transfer()
    _warn_token()
    _clone(config.VIDEO_REPO_URL, config.VIDEO_CODE_DIR, "generate.py")

    wdir = config.VIDEO_WEIGHTS_DIR
    if _has_weights(wdir):
        _log(f"video weights present: {wdir.name} ✓")
        return config.VIDEO_CODE_DIR, wdir

    from huggingface_hub import snapshot_download

    _log(f"downloading video weights {config.VIDEO_MODEL_REPO} -> {wdir}")
    snapshot_download(
        repo_id=config.VIDEO_MODEL_REPO,
        local_dir=str(wdir),
        token=_hf_token(),
        max_workers=8,
    )
    _log("video weights ready ✓")
    return config.VIDEO_CODE_DIR, wdir


def ensure_audio() -> None:
    """Verify the Stable Audio Open pipeline is importable. Weights lazy-download
    on first use (gated model -> needs HF_TOKEN + accepted license)."""
    try:
        from diffusers import StableAudioPipeline  # noqa: F401

        _log("Stable Audio Open pipeline OK ✓ (weights auto-download on first run)")
    except Exception as e:  # noqa: BLE001
        _log(f"⚠️  StableAudioPipeline unavailable ({e}) — run setup.sh; video still works")


def ensure_all() -> None:
    ensure_video()
    ensure_audio()
    _log("all models ready ✓")


if __name__ == "__main__":
    try:
        ensure_all()
    except Exception as e:  # noqa: BLE001
        _log(f"ERROR: {e}")
        sys.exit(1)
