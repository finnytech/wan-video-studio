#!/usr/bin/env python3
"""Guard WAN's optional-model imports so text-to-video never breaks on a missing
heavy dependency (decord, peft, librosa, ...).

WHY THIS EXISTS
---------------
WAN 2.2's ``wan/__init__.py`` eagerly imports *every* model family::

    from .image2video     import WanI2V
    from .speech2video    import WanS2V      # needs decord + s2v audio deps
    from .text2video      import WanT2V      # <-- the ONLY one this studio uses
    from .textimage2video import WanTI2V
    from .animate         import WanAnimate  # needs decord + peft

Several of those pull in packages that are NOT listed in WAN's requirements.txt
and often have no cp312 wheel (notably ``decord``). Because the import is eager,
a single missing optional dep makes ``import wan`` crash -> the whole t2v render
dies with e.g. ``ModuleNotFoundError: No module named 'decord'``.

This studio only does text-to-video (WanT2V), so we wrap the non-t2v imports in
try/except: if their deps are absent the symbol becomes ``None`` instead of
blowing up ``import wan``. ``text2video`` stays a hard import -- if THAT breaks we
want to know loudly.

Idempotent: a marker line makes re-runs a no-op. Safe to re-run after WAN code
updates (setup + run both call it).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = "# [studio] optional-model imports guarded"
# Model families we do NOT use -> tolerate missing deps. text2video stays hard.
OPTIONAL = ("image2video", "speech2video", "textimage2video", "animate")


def guard(path: Path) -> int:
    if not path.is_file():
        print(f"   patch_wan: {path} not found (skip)")
        return 0
    src = path.read_text()
    if MARKER in src:
        print("   wan/__init__.py already guarded \u2713")
        return 0

    out: list[str] = [MARKER]
    changed = 0
    for line in src.splitlines():
        m = re.match(r"^\s*from\s+\.(\w+)\s+import\s+(.+?)\s*$", line)
        if m and m.group(1) in OPTIONAL:
            names = m.group(2)
            out.append("try:")
            out.append(f"    {line.strip()}")
            out.append(
                "except Exception:  # optional model deps (decord/peft/...) may be absent"
            )
            for sym in (s.strip() for s in names.split(",")):
                alias = sym.split(" as ")[-1].strip()
                if alias:
                    out.append(f"    {alias} = None")
            changed += 1
        else:
            out.append(line)

    if not changed:
        # Upstream layout changed; don't guess. The setup smoke-test will still
        # catch any real import breakage and auto-heal it.
        print("   patch_wan: no optional import lines matched (upstream changed?) -> left as-is")
        return 0

    path.write_text("\n".join(out) + "\n")
    print(f"   wan/__init__.py guarded \u2713 ({changed} optional imports wrapped)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: patch_wan.py <path-to-wan/__init__.py>", file=sys.stderr)
        sys.exit(2)
    sys.exit(guard(Path(sys.argv[1])))
