"""Experiment and model configuration, loaded from YAML and validated with Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from transitive_reasoning.data import DATASET_FIELDS, DATASET_FILES
from transitive_reasoning.paths import configs_dir
from transitive_reasoning.prompts import PromptTemplate

InstanceField = Literal["question", "choices", "fact1", "fact2", "deduction", "answer"]
Metric = Literal["mc_accuracy", "rouge1", "exact_match"]


class ShuffleWords(BaseModel):
    """Shuffle the words of each field (the paper's "Shuffled Facts")."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["shuffle_words"]
    fields: list[InstanceField] = Field(min_length=1)


class RemoveSharedWords(BaseModel):
    """Remove words shared by the two `between` fields from the `modify` fields."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["remove_shared_words"]
    between: tuple[InstanceField, InstanceField]
    modify: list[InstanceField] = Field(min_length=1)

    @model_validator(mode="after")
    def _modify_within_between(self) -> RemoveSharedWords:
        outside = [field for field in self.modify if field not in self.between]
        if outside:
            raise ValueError(f"modify fields {outside} are not among between={list(self.between)}")
        return self


class RemoveAnswerKeywords(BaseModel):
    """Remove every answer choice's text from the fields (the paper's "Keyword Ablation")."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["remove_answer_keywords"]
    fields: list[InstanceField] = Field(min_length=1)


ManipulationSpec = Annotated[
    ShuffleWords | RemoveSharedWords | RemoveAnswerKeywords, Field(discriminator="type")
]


class ExperimentConfig(BaseModel):
    """One diagnostic experiment: a dataset, a prompt, optional manipulations and a metric."""

    model_config = ConfigDict(extra="forbid")
    name: str
    paper: str
    dataset: str
    prompt: str
    manipulations: list[ManipulationSpec] = Field(default_factory=list)
    metric: Metric

    @model_validator(mode="after")
    def _check_references(self) -> ExperimentConfig:
        if self.dataset not in DATASET_FILES:
            raise ValueError(
                f"Unknown dataset {self.dataset!r}; choose from {sorted(DATASET_FILES)}"
            )
        try:
            template = PromptTemplate.load(self.prompt)
        except FileNotFoundError as error:
            raise ValueError(f"prompt {self.prompt!r} has no file: {error}") from None
        missing = template.fields - DATASET_FIELDS[self.dataset]
        if missing:
            raise ValueError(
                f"prompt {self.prompt!r} needs fields {sorted(missing)} "
                f"that dataset {self.dataset!r} does not provide"
            )
        return self


class GenerationParams(BaseModel):
    """Decoding settings; the defaults are the paper's (Appendix A)."""

    model_config = ConfigDict(extra="forbid")
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.75
    top_k: int = 40
    num_beams: int = 4
    max_new_tokens: int = 128


class ModelConfig(BaseModel):
    """A Hugging Face model and how to load and decode from it."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    name: str
    hf_id: str
    quantization: Literal["8bit", "none"] = "none"
    batch_size: int = Field(ge=1)
    generation: GenerationParams = Field(default_factory=GenerationParams)


def load_experiment(name_or_path: str) -> ExperimentConfig:
    """Load an experiment by name (`qasc/full`) from `configs/experiments/` or from a YAML path."""
    root = configs_dir() / "experiments"
    path = _resolve(name_or_path, root, kind="experiment")
    config = ExperimentConfig.model_validate(_read_yaml(path))
    expected = _name_from_path(path, root)
    if expected is not None and config.name != expected:
        raise ValueError(
            f"{path}: name is {config.name!r} but the file location implies {expected!r}"
        )
    return config


def load_model(name_or_path: str) -> ModelConfig:
    """Load a model config by name (`flan_t5_xxl`) from `configs/models/` or from a YAML path."""
    root = configs_dir() / "models"
    path = _resolve(name_or_path, root, kind="model")
    config = ModelConfig.model_validate(_read_yaml(path))
    expected = _name_from_path(path, root)
    if expected is not None and config.name != expected:
        raise ValueError(
            f"{path}: name is {config.name!r} but the file location implies {expected!r}"
        )
    return config


def list_experiments() -> list[ExperimentConfig]:
    root = configs_dir() / "experiments"
    return [load_experiment(str(path)) for path in sorted(root.rglob("*.yaml"))]


def list_models() -> list[ModelConfig]:
    root = configs_dir() / "models"
    return [load_model(str(path)) for path in sorted(root.glob("*.yaml"))]


def _resolve(name_or_path: str, root: Path, kind: str) -> Path:
    direct = Path(name_or_path)
    if direct.suffix in {".yaml", ".yml"} and direct.is_file():
        return direct
    path = root / f"{name_or_path}.yaml"
    if path.is_file():
        return path
    available = sorted(str(p.relative_to(root).with_suffix("")) for p in root.rglob("*.yaml"))
    raise FileNotFoundError(f"No {kind} {name_or_path!r}. Available: {', '.join(available)}")


def _name_from_path(path: Path, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root.resolve()).with_suffix("")).replace("\\", "/")
    except ValueError:
        return None


def _read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
