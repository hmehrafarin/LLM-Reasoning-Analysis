"""The paper's datasets: QASC (dev split) and the two Bamboogle re-annotations."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transitive_reasoning.paths import data_dir

DATASET_FILES: dict[str, str] = {
    "qasc": "qasc/dev.json",
    "bamboogle": "bamboogle/with_facts.json",
    "bamboogle_gibberish": "bamboogle/gibberish.json",
}


@dataclass(frozen=True)
class Instance:
    """One test query: a question, its two supporting facts, the deduction and the gold answer."""

    id: str
    question: str
    answer: str
    choices: str | None = None
    fact1: str | None = None
    fact2: str | None = None
    deduction: str | None = None
    original_answer: str | None = None


INSTANCE_FIELDS: frozenset[str] = frozenset(f.name for f in dataclasses.fields(Instance)) - {"id"}
_REQUIRED_KEYS: frozenset[str] = frozenset({"question", "answer"})

DATASET_FIELDS: dict[str, frozenset[str]] = {
    "qasc": frozenset({"question", "choices", "fact1", "fact2", "deduction", "answer"}),
    "bamboogle": frozenset({"question", "fact1", "fact2", "deduction", "answer"}),
    "bamboogle_gibberish": frozenset(
        {"question", "fact1", "fact2", "deduction", "answer", "original_answer"}
    ),
}


def load_instances(path: Path, prefix: str) -> list[Instance]:
    """Read a JSON array of records and give each one the id `<prefix>-<index>`."""
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON array of records")
    instances: list[Instance] = []
    for index, record in enumerate(records):
        unknown = set(record) - INSTANCE_FIELDS
        missing = _REQUIRED_KEYS - set(record)
        if unknown or missing:
            raise ValueError(
                f"{path} record {index}: unknown keys {sorted(unknown)}, "
                f"missing keys {sorted(missing)}"
            )
        instances.append(Instance(id=f"{prefix}-{index:04d}", **record))
    return instances


def load_dataset(name: str) -> list[Instance]:
    """Load one of the paper's datasets by name."""
    try:
        relative = DATASET_FILES[name]
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASET_FILES)}") from None
    return load_instances(data_dir() / relative, prefix=name)


def convert_qasc_record(record: dict[str, Any]) -> dict[str, str]:
    """Turn one line of the official QASC jsonl release into this repository's record format."""
    question = record["question"]
    choices = " ".join(f"({choice['label']}) {choice['text']}" for choice in question["choices"])
    key = record["answerKey"]
    answer_text = next(c["text"] for c in question["choices"] if c["label"] == key)
    combined = record["combinedfact"].strip()
    combined = combined[:1].lower() + combined[1:]
    if not combined.endswith("."):
        combined += "."
    return {
        "question": question["stem"],
        "choices": choices,
        "fact1": record["fact1"],
        "fact2": record["fact2"],
        "deduction": f"Therefore, {combined}",
        "answer": f"({key}) {answer_text}",
    }


def convert_qasc(source: Path, target: Path) -> int:
    """Convert an official QASC jsonl file to a JSON array at `target`; return the row count."""
    records = [
        convert_qasc_record(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(records)
