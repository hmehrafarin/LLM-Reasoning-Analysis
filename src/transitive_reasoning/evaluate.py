"""Metrics from Section 3.4 of the paper, as pure functions over strings."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rouge_score import rouge_scorer

_CHOICE_LETTER = re.compile(r"^\s*\(([A-Ha-h])\)")
_rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)


@dataclass(frozen=True)
class Metrics:
    """An aggregate score in [0, 1] plus the per-instance scores it was averaged from."""

    metric: str
    score: float
    n: int
    n_unparsed: int
    per_instance: list[float]


def choice_letter(text: str) -> str | None:
    """The multiple-choice letter at the start of an answer like "(F) beads", upper-cased."""
    match = _CHOICE_LETTER.match(text)
    return match.group(1).upper() if match else None


def mc_accuracy_scores(predictions: Sequence[str], references: Sequence[str]) -> list[float]:
    """QASC accuracy: exact match on the choice letter only."""
    return [
        float(choice_letter(pred) is not None and choice_letter(pred) == choice_letter(ref))
        for pred, ref in zip(predictions, references, strict=True)
    ]


def rouge1_scores(predictions: Sequence[str], references: Sequence[str]) -> list[float]:
    """Bamboogle ROUGE-1: unigram F1 between the prediction and the gold answer."""
    return [
        float(_rouge.score(ref, pred)["rouge1"].fmeasure)
        for pred, ref in zip(predictions, references, strict=True)
    ]


def exact_match_scores(predictions: Sequence[str], references: Sequence[str]) -> list[float]:
    """Restore-word-order accuracy: equality ignoring case, outer whitespace and a final period."""
    return [
        float(_normalise(pred) == _normalise(ref))
        for pred, ref in zip(predictions, references, strict=True)
    ]


METRICS: dict[str, Callable[[Sequence[str], Sequence[str]], list[float]]] = {
    "mc_accuracy": mc_accuracy_scores,
    "rouge1": rouge1_scores,
    "exact_match": exact_match_scores,
}


def score(predictions: Sequence[str], references: Sequence[str], metric: str) -> Metrics:
    """Score predictions against references with the named metric."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    try:
        scorer = METRICS[metric]
    except KeyError:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(METRICS)}") from None
    per_instance = scorer(predictions, references)
    mean = sum(per_instance) / len(per_instance) if per_instance else 0.0
    return Metrics(
        metric=metric,
        score=mean,
        n=len(per_instance),
        n_unparsed=sum(1 for pred in predictions if not pred.strip()),
        per_instance=per_instance,
    )


def _normalise(text: str) -> str:
    return text.strip().removesuffix(".").strip().lower()
