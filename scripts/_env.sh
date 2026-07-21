# Shared environment activation, sourced by setup.sh and run.sh.
# Uses a local .venv when one exists/was created; otherwise falls back to the
# currently active environment (e.g. Lightning.ai's single conda env, which
# forbids extra venvs). Sets $PYBIN to the python to use.

STUDIO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$STUDIO_ROOT/.venv"

if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  STUDIO_ENV_KIND="venv"
else
  STUDIO_ENV_KIND="system"
fi

PYBIN="$(command -v python3 || command -v python)"
export STUDIO_ROOT VENV_DIR PYBIN STUDIO_ENV_KIND
