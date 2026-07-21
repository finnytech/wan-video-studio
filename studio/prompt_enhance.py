"""Cinematic prompt enhancement for maximum realism / film-action look.

This is a text-only, model-agnostic quality lever: we append proven cinematic
descriptors to the user's prompt (unless they already wrote their own style) so
WAN 2.2 leans toward photoreal, filmic motion instead of flat/CGI output.

All strings are env-overridable and the whole thing can be turned off with
STUDIO_ENHANCE=0.
"""
from __future__ import annotations

import os

# Filmic look: shot on real cameras, natural light, real physics, high detail.
CINEMATIC_SUFFIX = os.environ.get(
    "STUDIO_CINEMATIC_SUFFIX",
    "cinematic film still, shot on ARRI Alexa, 35mm anamorphic lens, shallow depth "
    "of field, natural volumetric lighting, photorealistic, ultra-detailed textures, "
    "realistic physics and motion, dynamic action, high dynamic range, film grain, "
    "professional color grading, 8k, masterpiece",
)

# Kill the usual generative artifacts (kept as reference; WAN's CLI has no negative
# flag, but this string is reused by the muxer-free realism notes and any future
# pipeline that supports it).
NEGATIVE_PROMPT = os.environ.get(
    "STUDIO_NEGATIVE_PROMPT",
    "blurry, low quality, low resolution, jpeg artifacts, deformed, distorted, "
    "disfigured, bad anatomy, extra limbs, mutated hands, watermark, text, "
    "oversaturated, flat lighting, cartoon, cgi, plastic skin, flickering, "
    "morphing, warping, duplicate frames, static, still image",
)

# Heuristic: if the user already wrote cinematic language, don't double up.
_STYLE_HINTS = (
    "cinematic", "film", "35mm", "anamorphic", "photoreal", "arri", "8k", "4k",
    "depth of field", "bokeh", "color grade", "hdr", "grain",
)


def _enabled() -> bool:
    v = os.environ.get("STUDIO_ENHANCE", "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def enhance(prompt: str) -> str:
    """Return the prompt with a cinematic suffix appended (idempotent-ish)."""
    prompt = (prompt or "").strip()
    if not prompt or not _enabled():
        return prompt
    low = prompt.lower()
    if any(h in low for h in _STYLE_HINTS):
        return prompt  # user already set a look; respect it
    sep = "" if prompt.endswith((".", ",", ";")) else ","
    return f"{prompt}{sep} {CINEMATIC_SUFFIX}"


def negative() -> str:
    return NEGATIVE_PROMPT
