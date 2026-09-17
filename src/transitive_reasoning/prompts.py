"""Prompt templates (Appendix C of the paper) and parsing of model responses."""

from __future__ import annotations

import json
import string
from collections.abc import Mapping
from dataclasses import dataclass

from transitive_reasoning.paths import prompts_dir

DEDUCTION_LABELS: tuple[str, ...] = ("Deduce:", "Deduction:")
ANSWER_LABEL = "Answer:"
BLOCK_TERMINATORS: tuple[str, ...] = ("######", "Context:")
_KEYS = ("preamble", "query_template", "response_split")


@dataclass(frozen=True)
class ParsedResponse:
    """The deduction (if the model wrote one) and the final answer extracted from a generation."""

    deduction: str | None
    answer: str


@dataclass(frozen=True)
class PromptTemplate:
    """An instruction plus three demonstrations (`preamble`) and a query with placeholders."""

    name: str
    preamble: str
    query_template: str
    response_split: str

    @classmethod
    def load(cls, name: str) -> PromptTemplate:
        path = prompts_dir() / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"No prompt file at {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        missing = [key for key in _KEYS if key not in raw]
        if missing:
            raise ValueError(f"{path} is missing keys {missing}")
        return cls(
            name=name,
            preamble=raw["preamble"],
            query_template=raw["query_template"],
            response_split=raw["response_split"],
        )

    @property
    def fields(self) -> frozenset[str]:
        """Placeholder names used by the query template."""
        parsed = string.Formatter().parse(self.query_template)
        return frozenset(field for _, field, _, _ in parsed if field)

    def render(self, values: Mapping[str, object]) -> str:
        """Build the full prompt for one query; every placeholder must have a value (None is
        rejected).
        """
        missing = sorted(field for field in self.fields if values.get(field) is None)
        if missing:
            raise ValueError(f"Prompt {self.name!r} needs {missing} but they are missing or empty")
        return self.preamble + self.query_template.format_map(values)

    def parse(self, generation: str) -> ParsedResponse:
        return parse_response(generation, response_split=self.response_split)


def parse_response(generation: str, response_split: str = "Steps:") -> ParsedResponse:
    """Extract the deduction and answer from the first "Steps" block of a generation.

    Only newly generated text is expected. Anything after a demonstration separator
    (`######`) or a new `Context:` is a hallucinated demonstration and is ignored.
    """
    block = _first_block(generation)
    if response_split in block:
        block = block.split(response_split, 1)[1]
    deduction: str | None = None
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if deduction is None and line.startswith(DEDUCTION_LABELS):
            deduction = _after_label(line, DEDUCTION_LABELS)
        elif line.startswith(ANSWER_LABEL):
            return ParsedResponse(deduction=deduction, answer=line[len(ANSWER_LABEL) :].strip())
    return ParsedResponse(deduction=deduction, answer="")


def _first_block(text: str) -> str:
    cut = len(text)
    for marker in BLOCK_TERMINATORS:
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    return text[:cut]


def _after_label(line: str, labels: tuple[str, ...]) -> str:
    for label in labels:
        if line.startswith(label):
            return line[len(label) :].strip()
    raise ValueError(f"{line!r} has none of the labels {labels}")
