from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from transitive_reasoning import config as config_module
from transitive_reasoning.config import (
    ExperimentConfig,
    GenerationParams,
    InstanceField,
    ModelConfig,
    RemoveAnswerKeywords,
    RemoveSharedWords,
    ShuffleWords,
    list_experiments,
    list_models,
    load_experiment,
    load_model,
)
from transitive_reasoning.data import INSTANCE_FIELDS

# name -> the exact `paper` row label each config carries
PAPER: dict[str, str] = {
    "qasc/full": "Table 1, Full",
    "qasc/qa": "Table 1, QA",
    "qasc/qa_step_by_step": "Table 1, QA (step-by-step)",
    "qasc/qaf": "Table 1, QAF",
    "qasc/qaf_fact1_only": "Table 1, QAF (fact 1 only)",
    "qasc/qaf_fact2_only": "Table 1, QAF (fact 2 only)",
    "qasc/full_shuffled": "Figure 2, both facts shuffled",
    "qasc/f1q_ablation": "Table 2, F1Q Connecting Words Ablation",
    "qasc/f2q_ablation": "Table 2, F2Q Connecting Words Ablation",
    "qasc/f1f2_ablation": "Table 2, F1F2 Connecting Words Ablation",
    "qasc/f1f2a_keyword_ablation": "Table 2, F1F2A Keyword Ablation",
    "bamboogle/full": "Table 3, Full",
    "bamboogle/qa": "Table 3, QA",
    "bamboogle/qa_step_by_step": "Table 3, QA (step-by-step)",
    "bamboogle/qaf": "Table 3, QAF",
    "bamboogle/qaf_fact1_only": "Table 3, QAF (fact 1 only)",
    "bamboogle/qaf_fact2_only": "Table 3, QAF (fact 2 only)",
    "bamboogle/full_shuffled": "Table 3, Full (both facts shuffled)",
    "bamboogle/f1q_ablation": "Table 3, F1Q Connecting Words Ablation",
    "bamboogle/f2q_ablation": "Table 3, F2Q Connecting Words Ablation",
    "bamboogle/f1f2_ablation": "Table 3, F1F2 Connecting Words Ablation",
    "bamboogle_gibberish/full": "Table 4, Gibberish Full",
    "bamboogle_gibberish/full_shuffled": "Table 4, Gibberish Both Facts Shuffled",
}

# name -> (dataset, prompt, metric, manipulation types)
EXPECTED: dict[str, tuple[str, str, str, list[str]]] = {}
for variant in ("full", "qa", "qa_step_by_step", "qaf", "qaf_fact1_only", "qaf_fact2_only"):
    EXPECTED[f"qasc/{variant}"] = ("qasc", f"qasc/{variant}", "mc_accuracy", [])
    EXPECTED[f"bamboogle/{variant}"] = ("bamboogle", f"bamboogle/{variant}", "rouge1", [])
EXPECTED["qasc/full_shuffled"] = ("qasc", "qasc/full", "mc_accuracy", ["shuffle_words"])
for ablation in ("f1q_ablation", "f2q_ablation", "f1f2_ablation"):
    EXPECTED[f"qasc/{ablation}"] = ("qasc", "qasc/full", "mc_accuracy", ["remove_shared_words"])
    EXPECTED[f"bamboogle/{ablation}"] = (
        "bamboogle",
        "bamboogle/full",
        "rouge1",
        ["remove_shared_words"],
    )
EXPECTED["qasc/f1f2a_keyword_ablation"] = (
    "qasc",
    "qasc/full",
    "mc_accuracy",
    ["remove_answer_keywords"],
)
EXPECTED["bamboogle/full_shuffled"] = ("bamboogle", "bamboogle/full", "rouge1", ["shuffle_words"])
EXPECTED["bamboogle_gibberish/full"] = ("bamboogle_gibberish", "bamboogle/full", "rouge1", [])
EXPECTED["bamboogle_gibberish/full_shuffled"] = (
    "bamboogle_gibberish",
    "bamboogle/full",
    "rouge1",
    ["shuffle_words"],
)


def test_experiment_set_matches_the_paper() -> None:
    experiments = {e.name: e for e in list_experiments()}
    assert set(experiments) == set(EXPECTED) == set(PAPER)
    for name, (dataset, prompt, metric, types) in EXPECTED.items():
        experiment = experiments[name]
        assert (experiment.dataset, experiment.prompt, experiment.metric, experiment.paper) == (
            dataset,
            prompt,
            metric,
            PAPER[name],
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


def test_model_name_must_match_file_location(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models = tmp_path / "configs" / "models"
    models.mkdir(parents=True)
    (models / "typo.yaml").write_text("name: other\nhf_id: x/y\nbatch_size: 1\n")
    monkeypatch.setattr(config_module, "configs_dir", lambda: tmp_path / "configs")
    with pytest.raises(ValueError, match="name"):
        load_model("typo")


def test_instance_field_literal_matches_the_dataclass() -> None:
    assert set(get_args(InstanceField)) <= INSTANCE_FIELDS


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
