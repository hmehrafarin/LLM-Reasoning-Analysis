from __future__ import annotations

from pathlib import Path

import pytest

from transitive_reasoning import paths


def test_repo_root_holds_pyproject() -> None:
    assert (paths.repo_root() / "pyproject.toml").is_file()


def test_env_var_overrides_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    assert paths.repo_root() == tmp_path.resolve()
    assert paths.data_dir() == tmp_path.resolve() / "data"
    assert paths.prompts_dir() == tmp_path.resolve() / "prompts"
    assert paths.configs_dir() == tmp_path.resolve() / "configs"
