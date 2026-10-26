"""Locating the workspace without passing ``-w`` on every command.

Resolution order: ``-w`` argument, ``CHUNKTRIAGER_WORKSPACE`` environment
variable, the default saved by ``chunktriager use``, then the current
directory if it contains a config.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .config import CONFIG_NAME, ConfigError

ENV_WORKSPACE = "CHUNKTRIAGER_WORKSPACE"
# Overrides where the user settings file lives; used by tests.
ENV_HOME = "CHUNKTRIAGER_HOME"


def settings_path() -> Path:
    if os.environ.get(ENV_HOME):
        base = Path(os.environ[ENV_HOME])
    elif sys.platform == "win32" and os.environ.get("APPDATA"):
        base = Path(os.environ["APPDATA"]) / "chunktriager"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "chunktriager"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "chunktriager"
    return base / "settings.json"


def saved_default() -> Path | None:
    path = settings_path()
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("workspace")
    except (OSError, ValueError):
        return None
    return Path(value) if value else None


def save_default(workspace: Path) -> Path:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"workspace": str(workspace.resolve())}, indent=2), encoding="utf-8")
    return path


def resolve(explicit: Path | None, *, for_init: bool = False) -> Path:
    """Pick the workspace directory, or raise ConfigError explaining how to set one."""
    if explicit is not None:
        return explicit
    if for_init:
        # init creates a new workspace, so it never falls back to the default.
        raise ConfigError("give the new workspace folder: chunktriager -w <folder> init --world <world copy>")
    if os.environ.get(ENV_WORKSPACE):
        return Path(os.environ[ENV_WORKSPACE])
    saved = saved_default()
    if saved is not None:
        return saved
    if (Path.cwd() / CONFIG_NAME).is_file():
        return Path.cwd()
    raise ConfigError(
        "no workspace configured; run `chunktriager use <folder>` once, "
        "or pass -w <folder>"
    )
