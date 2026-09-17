"""Command-line interface: run experiments, re-score results, list the experiment set."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, NoReturn, TypeVar

import typer
from pydantic import ValidationError

from transitive_reasoning.config import (
    ManipulationSpec,
    ModelConfig,
    RemoveSharedWords,
    ShuffleWords,
    list_experiments,
    load_experiment,
    load_model,
)
from transitive_reasoning.data import convert_qasc
from transitive_reasoning.evaluate import Metrics, score
from transitive_reasoning.restore_word_order import run_restore_word_order
from transitive_reasoning.results import RunInfo, read_predictions, write_run
from transitive_reasoning.run import ModelBackend, run_experiment

app = typer.Typer(
    help="Diagnostic experiments from 'Reasoning or a Semblance of it?' (EMNLP 2024).",
    no_args_is_help=True,
    add_completion=False,
)

ExperimentOption = Annotated[
    str,
    typer.Option("--experiment", "-e", help="Experiment name such as qasc/full, or a YAML path."),
]
ModelOption = Annotated[
    str, typer.Option("--model", "-m", help="Model name such as llama2_13b_chat, or a YAML path.")
]
SeedOption = Annotated[int, typer.Option(help="Seed for shuffling and sampling.")]
LimitOption = Annotated[int | None, typer.Option(help="Only run the first N instances.")]
OutOption = Annotated[Path, typer.Option(help="Root directory for results.")]
CacheOption = Annotated[Path | None, typer.Option(help="Hugging Face cache directory.")]

ConfigT = TypeVar("ConfigT")


def build_backend(model: ModelConfig, cache_dir: Path | None) -> ModelBackend:
    """Load the Hugging Face model; imported lazily because torch is slow to import."""
    from transitive_reasoning.models import HuggingFaceBackend

    return HuggingFaceBackend.from_config(model, cache_dir=cache_dir)


@app.command()
def run(
    experiment: ExperimentOption,
    model: ModelOption,
    seed: SeedOption = 42,
    limit: LimitOption = None,
    out: OutOption = Path("results"),
    cache_dir: CacheOption = None,
    save_prompts: Annotated[bool, typer.Option(help="Store the full prompt on every row.")] = False,
) -> None:
    """Generate answers for one experiment and score them."""
    experiment_config = _load(load_experiment, experiment)
    model_config = _load(load_model, model)
    backend = build_backend(model_config, cache_dir)
    result = run_experiment(
        experiment_config, backend, seed=seed, limit=limit, save_prompts=save_prompts
    )
    out_dir = out / experiment_config.name / model_config.name / f"seed{seed}"
    info = RunInfo.create(experiment_config.model_dump(), model_config.model_dump(), seed)
    write_run(out_dir, result.predictions, result.metrics, info)
    typer.echo(_summary(result.metrics, out_dir))


@app.command("evaluate")
def evaluate_command(
    predictions: Annotated[Path, typer.Argument(help="A predictions.jsonl file from `run`.")],
    metric: Annotated[
        str | None, typer.Option(help="Metric to use; default: the run's own.")
    ] = None,
) -> None:
    """Re-score an existing predictions file."""
    rows = read_predictions(predictions)
    if not rows:
        _fail(f"{predictions} has no rows")
    if "predicted_answer" in rows[0]:
        predicted_key, reference_key = "predicted_answer", "answer"
    elif "predicted" in rows[0]:
        predicted_key, reference_key = "predicted", "original"
    else:
        _fail(f"{predictions}: rows must carry predicted_answer/answer or predicted/original")
    if metric is not None:
        chosen = metric
    else:
        try:
            metrics_json = json.loads((predictions.parent / "metrics.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            _fail(f"no readable metrics.json next to {predictions}; pass --metric")
        chosen = metrics_json["metric"]
    try:
        metrics = score([r[predicted_key] for r in rows], [r[reference_key] for r in rows], chosen)
    except ValueError as error:
        _fail(str(error))
    typer.echo(_summary(metrics, None))


@app.command("restore-word-order")
def restore_word_order_command(
    model: ModelOption,
    seed: SeedOption = 42,
    limit: LimitOption = None,
    out: OutOption = Path("results"),
    cache_dir: CacheOption = None,
) -> None:
    """Section 5.1: ask the model to restore the word order of shuffled QASC facts."""
    model_config = _load(load_model, model)
    backend = build_backend(model_config, cache_dir)
    result = run_restore_word_order(backend, seed=seed, limit=limit)
    out_dir = out / "restore_word_order" / model_config.name / f"seed{seed}"
    info = RunInfo.create({"name": "restore_word_order"}, model_config.model_dump(), seed)
    write_run(out_dir, result.predictions, result.metrics, info)
    typer.echo(_summary(result.metrics, out_dir))


@app.command("list")
def list_command(
    markdown: Annotated[bool, typer.Option(help="Print a Markdown table for the README.")] = False,
) -> None:
    """List every experiment with the paper table it reproduces."""
    experiments = list_experiments()
    if markdown:
        typer.echo("| Experiment | Dataset | Prompt | Manipulations | Metric | Paper |")
        typer.echo("|---|---|---|---|---|---|")
    for experiment in experiments:
        manipulations = ", ".join(_describe(m) for m in experiment.manipulations) or "none"
        if markdown:
            typer.echo(
                f"| `{experiment.name}` | {experiment.dataset} | `{experiment.prompt}` | "
                f"{manipulations} | {experiment.metric} | {experiment.paper} |"
            )
        else:
            typer.echo(f"{experiment.name:36} {experiment.metric:12} {experiment.paper}")


@app.command("prepare-qasc")
def prepare_qasc_command(
    source: Annotated[Path, typer.Argument(help="dev.jsonl from the official QASC release.")],
    target: Annotated[
        Path, typer.Argument(help="Output path, normally data/qasc/dev.json.")
    ] = Path("data/qasc/dev.json"),
) -> None:
    """Convert the official QASC release into this repository's record format."""
    count = convert_qasc(source, target)
    typer.echo(f"wrote {count} records to {target}")


def _load(loader: Callable[[str], ConfigT], name: str) -> ConfigT:
    try:
        return loader(name)
    except (FileNotFoundError, ValidationError, ValueError) as error:
        _fail(str(error))


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=2)


def _summary(metrics: Metrics, out_dir: Path | None) -> str:
    text = f"{metrics.metric}: {metrics.score * 100:.1f}"
    text += f" (n={metrics.n}, unparsed={metrics.n_unparsed})"
    return f"{text} -> {out_dir}" if out_dir else text


def _describe(spec: ManipulationSpec) -> str:
    if isinstance(spec, ShuffleWords):
        return f"shuffle_words({', '.join(spec.fields)})"
    if isinstance(spec, RemoveSharedWords):
        between = f"{spec.between[0]}, {spec.between[1]}"
        return f"remove_shared_words({between}) on {', '.join(spec.modify)}"
    return f"remove_answer_keywords({', '.join(spec.fields)})"
