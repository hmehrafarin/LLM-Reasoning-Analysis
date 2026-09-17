from __future__ import annotations

import random
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from transitive_reasoning.data import Instance
from transitive_reasoning.manipulations import (
    parse_choices,
    remove_answer_keywords,
    remove_shared_words,
    shuffle_text,
    shuffle_words,
)

words = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8), min_size=1, max_size=12
)


@given(words)
def test_shuffle_text_keeps_the_same_words(word_list: list[str]) -> None:
    sentence = " ".join(word_list)
    assert sorted(shuffle_text(sentence, random.Random(0)).split(" ")) == sorted(word_list)


@given(words, st.integers(min_value=0, max_value=10_000))
def test_shuffle_text_is_deterministic_for_a_seed(word_list: list[str], seed: int) -> None:
    sentence = " ".join(word_list)
    assert shuffle_text(sentence, random.Random(seed)) == shuffle_text(
        sentence, random.Random(seed)
    )


def test_shuffle_text_drops_one_trailing_period() -> None:
    assert shuffle_text("water.", random.Random(1)) == "water"


def test_shuffle_words_touches_only_named_fields(climate_instance: Instance) -> None:
    shuffled = shuffle_words(climate_instance, ["fact1", "fact2"], random.Random(3))
    assert shuffled.question == climate_instance.question
    assert shuffled.choices == climate_instance.choices
    assert shuffled.answer == climate_instance.answer
    assert sorted(shuffled.fact1.split(" ")) == sorted(
        climate_instance.fact1.removesuffix(".").split(" ")
    )
    assert sorted(shuffled.fact2.split(" ")) == sorted(
        climate_instance.fact2.removesuffix(".").split(" ")
    )


def test_remove_shared_words_paper_example(climate_instance: Instance) -> None:
    """Table 6 of the paper: fact 1 becomes 'Climate generally moisture'."""
    ablated, removed = remove_shared_words(climate_instance, ("fact1", "question"), ["fact1"])
    assert ablated.fact1 == "Climate generally moisture"
    assert ablated.question == climate_instance.question
    assert removed == ["and", "described", "in", "is", "of", "temperature", "terms"]


def test_remove_shared_words_between_facts_modifies_both(climate_instance: Instance) -> None:
    ablated, removed = remove_shared_words(climate_instance, ("fact1", "fact2"), ["fact1", "fact2"])
    for word in removed:
        assert word not in ablated.fact1.split(" ")
        assert word not in ablated.fact2.split(" ")
    assert "moisture" in removed


def test_remove_shared_words_leaves_a_space_when_nothing_remains(
    climate_instance: Instance,
) -> None:
    instance = replace(climate_instance, fact1="temperature", fact2="temperature")
    ablated, removed = remove_shared_words(instance, ("fact1", "fact2"), ["fact1"])
    assert ablated.fact1 == " "
    assert removed == ["temperature"]


def test_remove_shared_words_is_case_sensitive(climate_instance: Instance) -> None:
    instance = replace(climate_instance, fact1="Climate climate", question="climate?")
    ablated, removed = remove_shared_words(instance, ("fact1", "question"), ["fact1"])
    assert ablated.fact1 == "Climate"
    assert removed == ["climate"]


def test_parse_choices() -> None:
    assert parse_choices("(A) sand (B) occurs over a wide range (C) forests") == [
        "sand",
        "occurs over a wide range",
        "forests",
    ]


def test_remove_answer_keywords_paper_example(climate_instance: Instance) -> None:
    """Table 6 of the paper: 'Climate ' is removed from fact 1, case-insensitively."""
    ablated = remove_answer_keywords(climate_instance, ["fact1", "fact2"])
    assert ablated.fact1 == "is generally described in terms of temperature and moisture."
    assert ablated.fact2 == climate_instance.fact2
    assert ablated.question == climate_instance.question


def test_remove_answer_keywords_removes_every_occurrence(climate_instance: Instance) -> None:
    instance = replace(climate_instance, fact2="Rain and rain and RAIN.")
    assert remove_answer_keywords(instance, ["fact2"]).fact2 == "and and ."


def test_manipulating_a_missing_field_raises(climate_instance: Instance) -> None:
    instance = replace(climate_instance, fact2=None)
    with pytest.raises(ValueError, match="fact2"):
        shuffle_words(instance, ["fact2"], random.Random(0))
