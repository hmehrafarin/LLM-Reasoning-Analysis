from __future__ import annotations

import pytest

from transitive_reasoning.evaluate import (
    METRICS,
    choice_letter,
    exact_match_scores,
    mc_accuracy_scores,
    rouge1_scores,
    score,
)


def test_choice_letter() -> None:
    assert choice_letter("(F) beads") == "F"
    assert choice_letter("  (b) climate") == "B"
    assert choice_letter("beads") is None
    assert choice_letter("") is None


def test_mc_accuracy_compares_letters_only() -> None:
    predictions = ["(A) sand", "(B) wrong text", "(C) x", ""]
    references = ["(A) sand", "(B) right text", "(D) x", "(E) y"]
    assert mc_accuracy_scores(predictions, references) == [1.0, 1.0, 0.0, 0.0]


def test_rouge1_is_unigram_f1() -> None:
    assert rouge1_scores(["April 22, 1994"], ["22 April 1994"]) == [1.0]
    assert rouge1_scores(["1953"], ["July 27, 1953"]) == [pytest.approx(0.5)]
    assert rouge1_scores([""], ["1953"]) == [0.0]


def test_exact_match_ignores_case_whitespace_and_trailing_period() -> None:
    assert exact_match_scores([" Beads of water. "], ["beads of water"]) == [1.0]
    assert exact_match_scores(["beads of water vapor"], ["beads of water"]) == [0.0]


def test_score_aggregates_and_counts_unparsed() -> None:
    metrics = score(["(A) x", "", "(C) z"], ["(A) x", "(B) y", "(D) z"], "mc_accuracy")
    assert metrics.metric == "mc_accuracy"
    assert metrics.score == pytest.approx(1 / 3)
    assert metrics.n == 3
    assert metrics.n_unparsed == 1
    assert metrics.per_instance == [1.0, 0.0, 0.0]


def test_score_on_empty_input_is_zero() -> None:
    assert score([], [], "rouge1").score == 0.0


def test_score_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        score(["a"], [], "exact_match")


def test_score_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        score(["a"], ["a"], "bleu")


def test_metric_registry_names() -> None:
    assert set(METRICS) == {"mc_accuracy", "rouge1", "exact_match"}
