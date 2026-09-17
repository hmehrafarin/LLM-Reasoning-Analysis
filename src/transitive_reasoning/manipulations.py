"""Manipulations of the test query (Sections 5 and 6 of the paper).

Every function is pure: it returns a new `Instance` and never touches the
demonstrations, which stay exactly as in the prompt file.
"""

from __future__ import annotations

import random
import re
import string
from collections.abc import Sequence
from dataclasses import replace

from transitive_reasoning.data import Instance

_PUNCTUATION = str.maketrans("", "", string.punctuation)
_CHOICE = re.compile(r"\(([A-Z])\)\s*(.*?)\s*(?=\([A-Z]\)|$)")


def shuffle_text(text: str, rng: random.Random) -> str:
    """Randomly reorder the space-separated words of `text` after dropping one trailing period."""
    words = text.removesuffix(".").split(" ")
    return " ".join(rng.sample(words, len(words)))


def shuffle_words(instance: Instance, fields: Sequence[str], rng: random.Random) -> Instance:
    """The "Shuffled Facts" manipulation: shuffle the words inside each named field."""
    return replace(
        instance, **{field: shuffle_text(_text(instance, field), rng) for field in fields}
    )


def remove_shared_words(
    instance: Instance, between: tuple[str, str], modify: Sequence[str]
) -> tuple[Instance, list[str]]:
    """The "Connecting Words Ablation": delete words the two `between` fields have in common.

    Words are compared case-sensitively after stripping punctuation. Only the
    fields listed in `modify` are rewritten (the question is never changed in the
    paper's F1Q and F2Q experiments). Returns the new instance and the removed words.
    """
    tokens = {field: _tokens(_text(instance, field)) for field in between}
    shared = set(tokens[between[0]]) & set(tokens[between[1]])
    updates: dict[str, str] = {}
    for field in modify:
        if field not in tokens:
            raise ValueError(f"{field!r} is not one of the compared fields {between}")
        updates[field] = " ".join(word for word in tokens[field] if word not in shared) or " "
    return replace(instance, **updates), sorted(shared)


def parse_choices(choices: str) -> list[str]:
    """Split a multiple-choice string like "(A) sand (B) forests" into its choice texts."""
    return [text for _, text in _CHOICE.findall(choices) if text]


def remove_answer_keywords(instance: Instance, fields: Sequence[str]) -> Instance:
    """The "Keyword Ablation": delete every answer choice's text from the named fields."""
    keywords = parse_choices(_text(instance, "choices"))
    updates: dict[str, str] = {}
    for field in fields:
        text = _text(instance, field)
        for keyword in keywords:
            pattern = re.escape(keyword)
            text = re.sub(pattern + " ", "", text, flags=re.IGNORECASE)
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        updates[field] = text
    return replace(instance, **updates)


def _text(instance: Instance, field: str) -> str:
    value = getattr(instance, field)
    if not isinstance(value, str):
        raise ValueError(f"Instance {instance.id} has no {field!r} to manipulate")
    return value


def _tokens(text: str) -> list[str]:
    return text.translate(_PUNCTUATION).split(" ")
