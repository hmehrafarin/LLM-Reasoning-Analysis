"""Locations of the repository's data, prompt and config directories."""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "TRANSITIVE_REASONING_ROOT"


def repo_root() -> Path:
    """Return the repository root: `$TRANSITIVE_REASONING_ROOT` or the directory holding
    pyproject.toml.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(f"No pyproject.toml above {here}; set {ENV_VAR} to the repository root")


def data_dir() -> Path:
    return repo_root() / "data"


def prompts_dir() -> Path:
    return repo_root() / "prompts"


def configs_dir() -> Path:
    return repo_root() / "configs"
