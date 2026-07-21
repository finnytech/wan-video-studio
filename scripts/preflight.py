"""Preflight dependency + model check.

Exit codes:
  0  -> everything present (core + optional); launch directly.
  1  -> something CORE is missing; caller must run setup.sh.
  2  -> core OK but an OPTIONAL piece is missing (e.g. audiocraft); caller may
        run setup once, but can also launch (video works, audio may be off).

"Core" = python deps to import the app + WAN code + WAN weights.
AudioCraft is treated as OPTIONAL: the video pipeline runs without it, so a
missing audio stack only prints a warning (it never forces a full reinstall).

Run:  python scripts/preflight.py            # core check
      python scripts/preflight.py --verbose  # list every component
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Ensure the repo root (parent of scripts/) is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio import config


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:  # noqa: BLE001
        return False


def _has_weights(path) -> bool:
    if not path.exists():
        return False
    for pat in ("*.safetensors", "*.pth", "*.bin", "*.ckpt"):
        if any(path.rglob(pat)):
            return True
    return False


def _wan_import_error() -> str | None:
    """Real t2v health check: does ``import wan`` succeed on a MISSING module?

    WAN's ``wan/__init__.py`` eagerly imports several model families. run.sh
    guards the non-t2v ones, so this should pass; if a module is still missing,
    a CORE t2v dep is absent and setup must run. This runs in preflight's own
    process (separate from the app), so the heavy import can't pollute the app.
    Non-missing import errors are ignored here -- we don't force reinstalls for
    unrelated runtime issues.
    """
    code_dir = config.VIDEO_CODE_DIR
    if not (code_dir / "generate.py").exists():
        return None  # already reported separately as wan2.2-code
    cwd0 = os.getcwd()
    sys.path.insert(0, str(code_dir))
    try:
        os.chdir(str(code_dir))
        import importlib

        importlib.import_module("wan")
        return None
    except ModuleNotFoundError as e:  # noqa: BLE001
        return f"missing {e.name}"
    except Exception:  # noqa: BLE001
        return None  # non-missing import error: don't trigger an endless reinstall
    finally:
        os.chdir(cwd0)


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    core_missing: list[str] = []
    optional_missing: list[str] = []

    # --- python deps needed to even import + run the UI + video model ---
    # einops is imported directly by WAN's model code, so it's CORE (video won't
    # run without it). flash_attn is NOT listed: WAN falls back to torch SDPA.
    for mod in ("gradio", "torch", "huggingface_hub", "numpy", "einops"):
        if not _have(mod):
            core_missing.append(f"pip:{mod}")

    # --- torch must see CUDA ---
    if _have("torch"):
        try:
            import torch

            if not torch.cuda.is_available():
                core_missing.append("torch-cuda (no GPU visible)")
        except Exception as e:  # noqa: BLE001
            core_missing.append(f"torch-import ({e})")

    # --- WAN 2.2 code checkout ---
    if not (config.VIDEO_CODE_DIR / "generate.py").exists():
        core_missing.append("wan2.2-code")

    # --- WAN 2.2 T2V-A14B weights ---
    if not _has_weights(config.VIDEO_WEIGHTS_DIR):
        core_missing.append("wan2.2-weights")

    # --- WAN import chain actually works (only worth checking if basics are OK) ---
    if not core_missing:
        wan_err = _wan_import_error()
        if wan_err:
            core_missing.append(f"wan-import ({wan_err})")

    # --- optional: AudioCraft (sound stage) ---
    if not _have("audiocraft"):
        optional_missing.append("audiocraft (sound stage)")

    # --- optional: ffmpeg binary (mux) ---
    import shutil

    if not shutil.which("ffmpeg"):
        core_missing.append("ffmpeg-binary")

    if verbose or core_missing or optional_missing:
        print("== preflight ==")
        print(f"  core missing     : {core_missing or 'none ✓'}")
        print(f"  optional missing : {optional_missing or 'none ✓'}")

    if core_missing:
        print("[preflight] core deps missing -> setup required", flush=True)
        return 1
    if optional_missing:
        print("[preflight] core OK ✓ (optional missing; video works, audio may be off)", flush=True)
        return 2
    print("[preflight] all deps + models present ✓ -> launching directly", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
