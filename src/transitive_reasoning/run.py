"""Run one experiment end to end: load, manipulate, prompt, generate, parse, score."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from transformers import set_seed

from transitive_reasoning.config import ExperimentConfig
from transitive_reasoning.data import Instance, load_dataset
from transitive_reasoning.evaluate import Metrics, score
from transitive_reasoning.manipulations import apply_manipulations
from transitive_reasoning.prompts import PromptTemplate
from transitive_reasoning.results import Prediction


class ModelBackend(Protocol):
    """Anything that turns a batch of prompts into a batch of generations."""

    def generate(self, prompts: Sequence[str]) -> list[str]: ...


@dataclass(frozen=True)
class RunResult:
    predictions: list[Prediction]
    metrics: Metrics


def seed_everything(seed: int) -> random.Random:
    """Seed Python, NumPy and torch, and return the generator used for manipulations."""
    set_seed(seed)
    return random.Random(seed)


def run_experiment(
    experiment: ExperimentConfig,
    backend: ModelBackend,
    *,
    seed: int = 42,
    limit: int | None = None,
    save_prompts: bool = False,
) -> RunResult:
    """Run `experiment` on `backend` and return scored predictions."""
    rng = seed_everything(seed)
    template = PromptTemplate.load(experiment.prompt)
    instances = load_dataset(experiment.dataset)
    if limit is not None:
        instances = instances[:limit]

    manipulated: list[Instance] = []
    removed_words: list[list[str]] = []
    for instance in instances:
        new_instance, removed = apply_manipulations(instance, experiment.manipulations, rng)
        manipulated.append(new_instance)
        removed_words.append(removed)

    prompts = [template.render(asdict(instance)) for instance in manipulated]
    generations = backend.generate(prompts)
    if len(generations) != len(prompts):
        raise RuntimeError(
            f"backend returned {len(generations)} generations for {len(prompts)} prompts"
        )
    parsed = [template.parse(generation) for generation in generations]
    metrics = score([p.answer for p in parsed], [i.answer for i in manipulated], experiment.metric)

    predictions = [
        Prediction(
            id=instance.id,
            question=instance.question,
            choices=instance.choices,
            fact1=instance.fact1,
            fact2=instance.fact2,
            deduction=instance.deduction,
            answer=instance.answer,
            removed_words=removed,
            generation=generation,
            predicted_deduction=response.deduction,
            predicted_answer=response.answer,
            score=instance_score,
            prompt=prompt if save_prompts else None,
        )
        for instance, removed, prompt, generation, response, instance_score in zip(
            manipulated,
            removed_words,
            prompts,
            generations,
            parsed,
            metrics.per_instance,
            strict=True,
        )
    ]
    return RunResult(predictions=predictions, metrics=metrics)
