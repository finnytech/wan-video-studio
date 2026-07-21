"""One-time OAuth-style token gate.

Goal: the public *.gradio.live link is reachable by anyone, but only a holder of
the freshly generated access token can actually use the app or spend GPU time.
This stops bots from spamming the endpoint and draining hourly GPU credits.

We use Gradio's native `auth` callback (username can be anything, password must
equal the token) so the gate sits in front of the whole app — no request reaches
the generation code without a valid token.
"""
from __future__ import annotations

import hmac
import secrets

from . import config


def generate_token() -> str:
    """Return the pinned token if set, else a fresh URL-safe token."""
    if config.PINNED_TOKEN:
        return config.PINNED_TOKEN
    return secrets.token_urlsafe(18)


def make_auth_callback(token: str):
    """Gradio auth callback: constant-time compare against the token.

    Any username is accepted; the password field must carry the token. Using
    hmac.compare_digest avoids timing side-channels.
    """

    def _check(username: str, password: str) -> bool:
        if not password:
            return False
        return hmac.compare_digest(password.strip(), token)

    return _check
