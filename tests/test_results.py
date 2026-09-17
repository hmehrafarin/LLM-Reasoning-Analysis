from __future__ import annotations

import json
import re
from pathlib import Path

from transitive_reasoning.evaluate import Metrics
from transitive_reasoning.results import (
    Prediction,
    RunInfo,
    git_commit,
    library_versions,
    read_predictions,
    write_run,
)


def make_prediction(index: int) -> Prediction:
    return Prediction(
        id=f"qasc-{index:04d}",
        question="q",
        choices="(A) a (B) b",
        fact1="f1",
        fact2="f2",
        deduction="d",
        answer="(A) a",
        removed_words=["x"],
        generation="Answer: (A) a",
        predicted_deduction=None,
        predicted_answer="(A) a",
        score=1.0,
    )


def test_write_run_creates_three_files(tmp_path: Path) -> None:
    out = tmp_path / "results" / "qasc" / "full" / "m" / "seed42"
    metrics = Metrics(metric="mc_accuracy", score=1.0, n=2, n_unparsed=0, per_instance=[1.0, 1.0])
    info = RunInfo.create(experiment={"name": "qasc/full"}, model={"name": "m"}, seed=42)
    write_run(out, [make_prediction(0), make_prediction(1)], metrics, info)

    rows = read_predictions(out / "predictions.jsonl")
    assert [row["id"] for row in rows] == ["qasc-0000", "qasc-0001"]
    assert rows[0]["removed_words"] == ["x"]
    assert rows[0]["prompt"] is None

    assert json.loads((out / "metrics.json").read_text()) == {
        "metric": "mc_accuracy",
        "score": 1.0,
        "n": 2,
        "n_unparsed": 0,
    }
    run = json.loads((out / "run.json").read_text())
    assert run["experiment"] == {"name": "qasc/full"}
    assert run["seed"] == 42
    assert set(run["library_versions"]) == {"torch", "transformers"}
    assert run["timestamp"].endswith("+00:00")


def test_git_commit_is_a_sha_or_none() -> None:
    commit = git_commit()
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_library_versions_are_strings() -> None:
    versions = library_versions()
    assert all(isinstance(v, str) and v for v in versions.values())
