#!/usr/bin/env python3
"""Give WAN's ``flash_attention`` a real PyTorch-SDPA fallback.

WHY THIS EXISTS
---------------
WAN 2.2's ``wan/modules/model.py`` calls ``flash_attention(...)`` **directly**
for every self/cross-attention. That function (in ``wan/modules/attention.py``)
starts with::

    assert FLASH_ATTN_2_AVAILABLE

so if the ``flash-attn`` wheel isn't installed (it builds from source and has no
cp312 prebuilt wheel -> the build fails on many boxes), EVERY render dies
instantly with a bare ``AssertionError`` — no matter how much VRAM is free.

The upstream code has an SDPA path only inside the higher-level ``attention()``
wrapper, which ``model.py`` never calls. So we inject an SDPA fallback at the top
of ``flash_attention`` itself: when neither FlashAttention 2 nor 3 is available,
route the same q/k/v through ``torch.nn.functional.scaled_dot_product_attention``
(which on Blackwell still uses an efficient fused kernel). Result: identical math,
no crash, no flash-attn dependency.

Idempotent: a marker comment makes re-runs a no-op. Safe to run every launch.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = "# [studio] sdpa-fallback injected"

FALLBACK = '''\
    {MARKER}
    if not (FLASH_ATTN_3_AVAILABLE or FLASH_ATTN_2_AVAILABLE):
        import torch.nn.functional as _F
        _out_dtype = q.dtype
        # WAN passes q/k/v as [B, L, num_heads, head_dim]; SDPA wants [B, H, L, D].
        _q = q.transpose(1, 2).to(dtype)
        _k = k.transpose(1, 2).to(dtype)
        _v = v.transpose(1, 2).to(dtype)
        if q_scale is not None:
            _q = _q * q_scale
        _o = _F.scaled_dot_product_attention(
            _q, _k, _v, attn_mask=None, is_causal=causal,
            dropout_p=dropout_p, scale=softmax_scale,
        )
        return _o.transpose(1, 2).contiguous().to(_out_dtype)
'''


def _find_def_end(lines: list[str], start: int) -> int:
    """Return the index of the line that closes the ``def flash_attention(...)``
    signature (the line ending with ``):``), scanning from ``start``."""
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("(") - lines[i].count(")")
        if depth <= 0 and lines[i].rstrip().endswith(":"):
            return i
    return -1


def patch(path: Path) -> int:
    if not path.is_file():
        print(f"   patch_wan_attention: {path} not found (skip)")
        return 0
    src = path.read_text()
    if MARKER in src:
        print("   wan attention.py already has SDPA fallback \u2713")
        return 0

    lines = src.splitlines()
    def_idx = -1
    for i, ln in enumerate(lines):
        if re.match(r"^\s*def\s+flash_attention\s*\(", ln):
            def_idx = i
            break
    if def_idx < 0:
        print("   patch_wan_attention: no flash_attention() def found (upstream changed?) -> left as-is")
        return 0

    end_idx = _find_def_end(lines, def_idx)
    if end_idx < 0:
        print("   patch_wan_attention: could not locate end of signature -> left as-is")
        return 0

    block = FALLBACK.format(MARKER=MARKER).rstrip("\n").split("\n")
    out = lines[: end_idx + 1] + block + lines[end_idx + 1 :]
    path.write_text("\n".join(out) + "\n")
    print("   wan attention.py SDPA fallback injected \u2713 (flash-attn now optional)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: patch_wan_attention.py <path-to-wan/modules/attention.py>", file=sys.stderr)
        sys.exit(2)
    sys.exit(patch(Path(sys.argv[1])))
