"""The restore-word-order probe (Section 5.1): can a model unscramble a shuffled fact?"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from transitive_reasoning.data import Instance, load_dataset
from transitive_reasoning.evaluate import Metrics, score
from transitive_reasoning.manipulations import shuffle_text
from transitive_reasoning.prompts import PromptTemplate
from transitive_reasoning.run import ModelBackend, seed_everything

PROMPT_NAME = "restore_word_order"
ORIGINAL_LABEL = "Original sentence:"


@dataclass(frozen=True)
class RestoreWordOrderPrediction:
    id: str
    shuffled: str
    original: str
    generation: str
    predicted: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RestoreWordOrderResult:
    predictions: list[RestoreWordOrderPrediction]
    metrics: Metrics


def unique_facts(instances: Sequence[Instance]) -> list[str]:
    """Every distinct fact (lower-cased) across fact1 and fact2, in first-seen order."""
    seen: dict[str, None] = {}
    for instance in instances:
        for fact in (instance.fact1, instance.fact2):
            if fact:
                seen.setdefault(fact.lower(), None)
    return list(seen)


def parse_restored_sentence(generation: str) -> str:
    """The text after the first `Original sentence:` label, or "" if the model wrote none."""
    for raw_line in generation.splitlines():
        line = raw_line.strip()
        if line.startswith(ORIGINAL_LABEL):
            return line[len(ORIGINAL_LABEL) :].strip()
    return ""


def run_restore_word_order(
    backend: ModelBackend, *, seed: int = 42, limit: int | None = None
) -> RestoreWordOrderResult:
    """Shuffle each unique QASC fact, ask the model to restore it, and score by exact match."""
    rng = seed_everything(seed)
    template = PromptTemplate.load(PROMPT_NAME)
    originals = unique_facts(load_dataset("qasc"))
    if limit is not None:
        originals = originals[:limit]
    shuffled = [shuffle_text(sentence, rng) for sentence in originals]
    prompts = [template.render({"sentence": sentence}) for sentence in shuffled]
    generations = backend.generate(prompts)
    predicted = [parse_restored_sentence(generation) for generation in generations]
    metrics = score(predicted, originals, "exact_match")
    predictions = [
        RestoreWordOrderPrediction(
            id=f"qasc-fact-{index:04d}",
            shuffled=shuffled_sentence,
            original=original,
            generation=generation,
            predicted=prediction,
            score=instance_score,
        )
        for index, (
            shuffled_sentence,
            original,
            generation,
            prediction,
            instance_score,
        ) in enumerate(
            zip(shuffled, originals, generations, predicted, metrics.per_instance, strict=True)
        )
    ]
    return RestoreWordOrderResult(predictions=predictions, metrics=metrics)
