from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.fakes import FakeBackend
from transitive_reasoning import cli
from transitive_reasoning.data import load_dataset

runner = CliRunner()


def reply(answer: str) -> str:
    return f"\nDeduce: d\nAnswer: {answer}\n"


def use_fake(monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> FakeBackend:
    backend = FakeBackend(replies)
    monkeypatch.setattr(cli, "build_backend", lambda model, cache_dir: backend)
    return backend


def test_list_shows_every_experiment_with_its_paper_reference() -> None:
    result = runner.invoke(cli.app, ["list"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 23
    assert any(line.startswith("qasc/f1q_ablation") and "Table 2" in line for line in lines)


def test_list_markdown_has_table_header() -> None:
    result = runner.invoke(cli.app, ["list", "--markdown"])
    assert result.exit_code == 0
    assert result.output.splitlines()[0].startswith("| Experiment |")
    assert "`qasc/full_shuffled`" in result.output


def test_run_writes_results_and_prints_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gold = load_dataset("qasc")[0].answer
    use_fake(monkeypatch, [reply(gold), reply("(Z) wrong")])
    result = runner.invoke(
        cli.app,
        [
            "run",
            "--experiment",
            "qasc/full",
            "--model",
            "flan_t5_xxl",
            "--limit",
            "2",
            "--seed",
            "5",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "qasc" / "full" / "flan_t5_xxl" / "seed5"
    assert (out_dir / "predictions.jsonl").is_file()
    assert json.loads((out_dir / "metrics.json").read_text())["score"] == 0.5
    assert json.loads((out_dir / "run.json").read_text())["seed"] == 5
    assert "mc_accuracy: 50.0" in result.output


def test_run_unknown_experiment_exits_with_2() -> None:
    result = runner.invoke(cli.app, ["run", "--experiment", "qasc/nope", "--model", "flan_t5_xxl"])
    assert result.exit_code == 2
    assert "qasc/full" in result.output


def test_evaluate_rescores_a_predictions_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gold = load_dataset("bamboogle")[0].answer
    use_fake(monkeypatch, [reply(gold)])
    runner.invoke(
        cli.app,
        [
            "run",
            "--experiment",
            "bamboogle/full",
            "--model",
            "flan_t5_xxl",
            "--limit",
            "1",
            "--out",
            str(tmp_path),
        ],
    )
    predictions = tmp_path / "bamboogle" / "full" / "flan_t5_xxl" / "seed42" / "predictions.jsonl"
    result = runner.invoke(cli.app, ["evaluate", str(predictions)])
    assert result.exit_code == 0, result.output
    assert "rouge1: 100.0" in result.output
    result = runner.invoke(cli.app, ["evaluate", str(predictions), "--metric", "exact_match"])
    assert "exact_match: 100.0" in result.output


def test_restore_word_order_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_fake(monkeypatch, ["Original sentence: nonsense", "Original sentence: nonsense"])
    result = runner.invoke(
        cli.app,
        ["restore-word-order", "--model", "flan_t5_xxl", "--limit", "2", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "restore_word_order" / "flan_t5_xxl" / "seed42"
    assert json.loads((out_dir / "metrics.json").read_text())["metric"] == "exact_match"
    assert "exact_match: 0.0" in result.output


def test_prepare_qasc_command(tmp_path: Path) -> None:
    record = {
        "question": {
            "stem": "Q?",
            "choices": [{"text": "a", "label": "A"}, {"text": "b", "label": "B"}],
        },
        "answerKey": "B",
        "fact1": "F1.",
        "fact2": "F2.",
        "combinedfact": "Combined.",
    }
    source = tmp_path / "dev.jsonl"
    source.write_text(json.dumps(record) + "\n")
    target = tmp_path / "dev.json"
    result = runner.invoke(cli.app, ["prepare-qasc", str(source), str(target)])
    assert result.exit_code == 0, result.output
    assert json.loads(target.read_text())[0]["answer"] == "(B) b"
