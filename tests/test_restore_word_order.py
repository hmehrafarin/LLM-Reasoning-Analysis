from __future__ import annotations

from dataclasses import replace

from tests.fakes import FakeBackend
from transitive_reasoning.data import Instance, load_dataset
from transitive_reasoning.restore_word_order import (
    parse_restored_sentence,
    run_restore_word_order,
    unique_facts,
)


def test_unique_facts_lowercases_dedupes_and_keeps_order(climate_instance: Instance) -> None:
    twin = replace(climate_instance, id="twin", fact1=climate_instance.fact2, fact2="Water is wet.")
    assert unique_facts([climate_instance, twin]) == [
        climate_instance.fact1.lower(),
        climate_instance.fact2.lower(),
        "water is wet.",
    ]


def test_qasc_has_many_unique_facts() -> None:
    facts = unique_facts(load_dataset("qasc"))
    assert len(facts) == 856


def test_parse_restored_sentence() -> None:
    assert (
        parse_restored_sentence("Original sentence: beads of water\nContext:") == "beads of water"
    )
    assert parse_restored_sentence("  Original sentence:   x  ") == "x"
    assert parse_restored_sentence("no label here") == ""


def test_run_restore_word_order_scores_exact_matches() -> None:
    originals = unique_facts(load_dataset("qasc"))[:2]
    backend = FakeBackend([f"Original sentence: {originals[0]}", "Original sentence: nonsense"])
    result = run_restore_word_order(backend, seed=3, limit=2)

    assert result.metrics.metric == "exact_match"
    assert result.metrics.score == 0.5
    assert result.predictions[0].original == originals[0]
    assert result.predictions[0].score == 1.0
    assert result.predictions[1].predicted == "nonsense"
    assert sorted(result.predictions[0].shuffled.split(" ")) == sorted(
        originals[0].removesuffix(".").split(" ")
    )
    assert backend.prompts[0].endswith(f" Shuffled sentence: {result.predictions[0].shuffled}")
    assert backend.prompts[0].startswith("Given the shuffled sentence below")
