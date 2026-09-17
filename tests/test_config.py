from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from transitive_reasoning import config as config_module
from transitive_reasoning.config import (
    ExperimentConfig,
    GenerationParams,
    ModelConfig,
    RemoveAnswerKeywords,
    RemoveSharedWords,
    ShuffleWords,
    list_experiments,
    list_models,
    load_experiment,
    load_model,
)

# name -> (dataset, prompt, metric, manipulation types, paper)
EXPECTED: dict[str, tuple[str, str, str, list[str], str]] = {}
for variant in ("full", "qa", "qa_step_by_step", "qaf", "qaf_fact1_only", "qaf_fact2_only"):
    EXPECTED[f"qasc/{variant}"] = ("qasc", f"qasc/{variant}", "mc_accuracy", [], "Table 1")
    EXPECTED[f"bamboogle/{variant}"] = (
        "bamboogle",
        f"bamboogle/{variant}",
        "rouge1",
        [],
        "Table 3",
    )
EXPECTED["qasc/full_shuffled"] = ("qasc", "qasc/full", "mc_accuracy", ["shuffle_words"], "Figure 2")
for ablation in ("f1q_ablation", "f2q_ablation", "f1f2_ablation"):
    EXPECTED[f"qasc/{ablation}"] = (
        "qasc",
        "qasc/full",
        "mc_accuracy",
        ["remove_shared_words"],
        "Table 2",
    )
    EXPECTED[f"bamboogle/{ablation}"] = (
        "bamboogle",
        "bamboogle/full",
        "rouge1",
        ["remove_shared_words"],
        "Table 3",
    )
EXPECTED["qasc/f1f2a_keyword_ablation"] = (
    "qasc",
    "qasc/full",
    "mc_accuracy",
    ["remove_answer_keywords"],
    "Table 2",
)
EXPECTED["bamboogle/full_shuffled"] = (
    "bamboogle",
    "bamboogle/full",
    "rouge1",
    ["shuffle_words"],
    "Table 3",
)
EXPECTED["bamboogle_gibberish/full"] = (
    "bamboogle_gibberish",
    "bamboogle/full",
    "rouge1",
    [],
    "Table 4",
)
EXPECTED["bamboogle_gibberish/full_shuffled"] = (
    "bamboogle_gibberish",
    "bamboogle/full",
    "rouge1",
    ["shuffle_words"],
    "Table 4",
)


def test_experiment_set_matches_the_paper() -> None:
    experiments = {e.name: e for e in list_experiments()}
    assert set(experiments) == set(EXPECTED)
    for name, (dataset, prompt, metric, types, paper) in EXPECTED.items():
        experiment = experiments[name]
        assert (experiment.dataset, experiment.prompt, experiment.metric, experiment.paper) == (
            dataset,
            prompt,
            metric,
            paper,
        ), name
        assert [m.type for m in experiment.manipulations] == types, name


def test_f1q_ablation_modifies_only_fact1() -> None:
    manipulation = load_experiment("qasc/f1q_ablation").manipulations[0]
    assert isinstance(manipulation, RemoveSharedWords)
    assert manipulation.between == ("fact1", "question")
    assert manipulation.modify == ["fact1"]


def test_f1f2_ablation_modifies_both_facts() -> None:
    manipulation = load_experiment("bamboogle/f1f2_ablation").manipulations[0]
    assert isinstance(manipulation, RemoveSharedWords)
    assert manipulation.modify == ["fact1", "fact2"]


def test_model_configs_carry_paper_settings() -> None:
    models = {m.name: m for m in list_models()}
    assert set(models) == {"llama2_13b_chat", "llama2_7b_chat", "flan_t5_xxl"}
    assert models["llama2_13b_chat"].batch_size == 3
    assert models["llama2_7b_chat"].batch_size == 5
    assert models["flan_t5_xxl"].batch_size == 2
    for model in models.values():
        assert model.quantization == "8bit"
        assert model.generation == GenerationParams(
            do_sample=True, temperature=0.7, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=128
        )


def test_load_by_path(tmp_path: Path) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text(
        "name: custom\npaper: none\ndataset: qasc\nprompt: qasc/qa\n"
        "manipulations: []\nmetric: mc_accuracy\n"
    )
    assert load_experiment(str(path)).name == "custom"


def test_unknown_experiment_name_lists_available() -> None:
    with pytest.raises(FileNotFoundError, match="qasc/full"):
        load_experiment("qasc/nope")


def test_name_must_match_file_location(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    experiments = tmp_path / "configs" / "experiments" / "qasc"
    experiments.mkdir(parents=True)
    (experiments / "typo.yaml").write_text(
        "name: qasc/other\npaper: none\ndataset: qasc\nprompt: qasc/qa\nmetric: mc_accuracy\n"
    )
    monkeypatch.setattr(config_module, "configs_dir", lambda: tmp_path / "configs")
    with pytest.raises(ValueError, match="name"):
        load_experiment("qasc/typo")


def test_unknown_dataset_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown dataset"):
        ExperimentConfig(
            name="x", paper="", dataset="hotpotqa", prompt="qasc/qa", metric="mc_accuracy"
        )


def test_prompt_needing_fields_the_dataset_lacks_is_rejected() -> None:
    with pytest.raises(ValidationError, match="choices"):
        ExperimentConfig(
            name="x", paper="", dataset="bamboogle", prompt="qasc/full", metric="rouge1"
        )


def test_missing_prompt_file_rejected() -> None:
    with pytest.raises(ValidationError, match="prompt"):
        ExperimentConfig(
            name="x", paper="", dataset="qasc", prompt="qasc/nope", metric="mc_accuracy"
        )


def test_manipulation_field_names_are_checked() -> None:
    with pytest.raises(ValidationError):
        ShuffleWords(type="shuffle_words", fields=["fact 1"])
    with pytest.raises(ValidationError, match="modify"):
        RemoveSharedWords(
            type="remove_shared_words", between=("fact1", "question"), modify=["fact2"]
        )


def test_unknown_manipulation_type_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "name": "x",
                "paper": "",
                "dataset": "qasc",
                "prompt": "qasc/full",
                "manipulations": [{"type": "random_tokens", "fields": ["fact1"]}],
                "metric": "mc_accuracy",
            }
        )


def test_extra_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(name="m", hf_id="x/y", batch_size=1, load_8bit=True)


def test_keyword_ablation_spec_type() -> None:
    manipulation = load_experiment("qasc/f1f2a_keyword_ablation").manipulations[0]
    assert isinstance(manipulation, RemoveAnswerKeywords)
    assert manipulation.fields == ["fact1", "fact2"]


def test_load_model_by_name() -> None:
    assert load_model("flan_t5_xxl").hf_id == "google/flan-t5-xxl"
