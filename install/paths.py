"""Immutable program files and writable per-user data, for source or release.

SPLASH_DATA, or `splash serve --data-dir`, names one directory everything
Splash writes goes under: models, runtime state and agent profiles (Layout),
and unless set otherwise the prepared weights, the Hub cache, the thinking
key and crash traces too (launcher.data_environment). Without it, a release
writes under ~/Library/Application Support/Splash and a source checkout under
itself.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = (ROOT / "release.json").is_file()
# The frozen server build_server.sh builds: one executable, no interpreter.
FROZEN = bool(getattr(sys, "frozen", False))
DATA_VARIABLE = "SPLASH_DATA"


@dataclass(frozen=True)
class Layout:
    """The writable directories of one data directory."""

    data: Path
    models: Path
    runtime: Path
    # Hermes's sessions and profile: kept out of build/, which `make clean`
    # removes.
    profiles: Path

    @property
    def weights(self) -> Path:
        """Where the engine prepares weights unless SPLASH_WEIGHT_CACHE says."""
        return self.data / "weights"

    @property
    def hub_cache(self) -> Path:
        """Where Hub downloads go unless HF_HUB_CACHE or HF_HOME says."""
        return self.data / "huggingface"


def custom_data_dir():
    """The data directory SPLASH_DATA names, or None."""
    value = os.environ.get(DATA_VARIABLE)
    return Path(value).expanduser().absolute() if value else None


def layout(data=None) -> Layout:
    """The layout of data, a custom data directory, or without one of
    SPLASH_DATA's, or else the default: a release's per-user directory, or
    the source checkout's own folders."""
    if data is None:
        data = custom_data_dir()
    if data is not None:
        data = Path(data).expanduser().absolute()
        return Layout(data, data / "models", data / "runtime", data / "runtime")
    if PACKAGED:
        data = Path.home() / "Library/Application Support/Splash"
        return Layout(data, data / "models", data / "runtime", data / "runtime")
    return Layout(
        ROOT, ROOT / "install/models", ROOT / "build/runtime", ROOT / "install/profiles"
    )


_DEFAULT = layout()
DATA = _DEFAULT.data
MODELS = _DEFAULT.models
RUNTIME = _DEFAULT.runtime
PROFILES = _DEFAULT.profiles
PYTHON = ROOT / ("python/bin/python3" if PACKAGED else ".venv/bin/python")
BINARY = ROOT / ("engine/splash" if PACKAGED else "build/splash")


def program(subcommand, script, *options):
    """The command that runs one of Splash's Python programs: script under
    PYTHON with these interpreter options, or in the frozen server, which
    has no interpreter, this executable's subcommand of that name
    (dev/tools/frozen_server.py)."""
    if FROZEN:
        return [sys.executable, subcommand]
    return [str(PYTHON), *options, str(script)]
