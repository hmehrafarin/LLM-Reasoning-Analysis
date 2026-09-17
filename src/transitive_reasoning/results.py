"""Run outputs: predictions.jsonl, metrics.json and run.json with reproducibility metadata."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from transitive_reasoning.evaluate import Metrics
from transitive_reasoning.paths import repo_root

PACKAGE = "transitive-reasoning"
TRACKED_LIBRARIES = ("torch", "transformers")


class JsonRow(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Prediction:
    """One scored instance: the (manipulated) inputs, the generation and what was parsed from it."""

    id: str
    question: str
    choices: str | None
    fact1: str | None
    fact2: str | None
    deduction: str | None
    answer: str
    removed_words: list[str]
    generation: str
    predicted_deduction: str | None
    predicted_answer: str
    score: float
    prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunInfo:
    """What produced a run: configs, seed, code version and library versions."""

    experiment: dict[str, Any]
    model: dict[str, Any]
    seed: int
    git_commit: str | None
    timestamp: str
    package_version: str
    library_versions: dict[str, str]

    @classmethod
    def create(cls, experiment: dict[str, Any], model: dict[str, Any], seed: int) -> RunInfo:
        return cls(
            experiment=experiment,
            model=model,
            seed=seed,
            git_commit=git_commit(),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            package_version=_version(PACKAGE),
            library_versions=library_versions(),
        )


def write_run(out_dir: Path, rows: Sequence[JsonRow], metrics: Metrics, run_info: RunInfo) -> None:
    """Write predictions.jsonl, metrics.json and run.json into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    summary = {
        "metric": metrics.metric,
        "score": metrics.score,
        "n": metrics.n,
        "n_unparsed": metrics.n_unparsed,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "run.json").write_text(
        json.dumps(asdict(run_info), indent=2) + "\n", encoding="utf-8"
    )


def read_predictions(path: Path) -> list[dict[str, Any]]:
    """Read a predictions.jsonl file back into dictionaries."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def git_commit() -> str | None:
    """The current commit hash, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, RuntimeError):
        return None
    return result.stdout.strip() or None


def library_versions() -> dict[str, str]:
    return {name: _version(name) for name in TRACKED_LIBRARIES}


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"
