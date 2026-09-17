from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from transitive_reasoning.data import Instance
from transitive_reasoning.paths import prompts_dir
from transitive_reasoning.prompts import ParsedResponse, PromptTemplate, parse_response

EXPECTED_PROMPTS = {
    f"{dataset}/{variant}"
    for dataset in ("qasc", "bamboogle")
    for variant in ("full", "qa", "qa_step_by_step", "qaf", "qaf_fact1_only", "qaf_fact2_only")
} | {"restore_word_order"}


def test_all_expected_prompt_files_exist() -> None:
    found = {
        str(p.relative_to(prompts_dir()).with_suffix("")) for p in prompts_dir().rglob("*.json")
    }
    assert found == EXPECTED_PROMPTS


@pytest.mark.parametrize("name", sorted(EXPECTED_PROMPTS))
def test_every_prompt_loads_with_known_placeholders(name: str) -> None:
    template = PromptTemplate.load(name)
    assert template.preamble.startswith(
        ("Follow the demonstrations", "Please think", "Given the shuffled")
    )
    assert template.fields <= {"question", "choices", "fact1", "fact2", "sentence"}


def test_qasc_full_uses_all_four_fields() -> None:
    assert PromptTemplate.load("qasc/full").fields == {"question", "choices", "fact1", "fact2"}


def test_bamboogle_prompts_never_use_choices() -> None:
    for variant in ("full", "qa", "qa_step_by_step", "qaf", "qaf_fact1_only", "qaf_fact2_only"):
        assert "choices" not in PromptTemplate.load(f"bamboogle/{variant}").fields


def test_render_appends_query_to_preamble(climate_instance: Instance) -> None:
    template = PromptTemplate.load("qasc/full")
    rendered = template.render(asdict(climate_instance))
    assert rendered.startswith(template.preamble)
    assert rendered.endswith(
        f"Question: {climate_instance.question}\nAnswers: {climate_instance.choices}\n"
        f"Fact 1: {climate_instance.fact1}\nFact 2: {climate_instance.fact2}\nSteps:"
    )


def test_render_rejects_missing_field(climate_instance: Instance) -> None:
    template = PromptTemplate.load("qasc/full")
    with pytest.raises(ValueError, match="choices"):
        template.render(asdict(replace(climate_instance, choices=None)))


def test_load_unknown_prompt_raises() -> None:
    with pytest.raises(FileNotFoundError):
        PromptTemplate.load("qasc/nonexistent")


def test_parse_deduction_and_answer() -> None:
    generation = "\nDeduce: Therefore, climate is described by water.\nAnswer: (B) climate\n"
    assert parse_response(generation) == ParsedResponse(
        deduction="Therefore, climate is described by water.", answer="(B) climate"
    )


def test_parse_accepts_deduction_label() -> None:
    assert parse_response("Deduction: foo\nAnswer: bar") == ParsedResponse("foo", "bar")


def test_parse_answer_only() -> None:
    assert parse_response("\nAnswer: 1999\n") == ParsedResponse(deduction=None, answer="1999")


def test_parse_scores_first_block_not_hallucinated_demonstration() -> None:
    generation = (
        "\nDeduce: first\nAnswer: (B) climate\n######\nContext:\nQuestion: made up\n"
        "Steps:\nDeduce: second\nAnswer: (C) rain\n"
    )
    assert parse_response(generation) == ParsedResponse("first", "(B) climate")


def test_parse_stops_at_new_context_without_separator() -> None:
    generation = "Answer: James Madison\nContext:\nQuestion: another\nSteps:\nAnswer: wrong"
    assert parse_response(generation).answer == "James Madison"


def test_parse_strips_leading_response_split() -> None:
    assert parse_response("Steps:\nAnswer: 1999") == ParsedResponse(None, "1999")


def test_parse_empty_generation() -> None:
    assert parse_response("") == ParsedResponse(deduction=None, answer="")


def test_template_parse_uses_its_own_split_marker() -> None:
    template = PromptTemplate.load("qasc/full")
    assert template.parse("Steps: \nAnswer: (A) x").answer == "(A) x"
