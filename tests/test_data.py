from __future__ import annotations

import json
from pathlib import Path

import pytest

from transitive_reasoning import data
from transitive_reasoning.data import (
    DATASET_FIELDS,
    DATASET_FILES,
    INSTANCE_FIELDS,
    Instance,
    convert_qasc,
    convert_qasc_record,
    load_dataset,
    load_instances,
)
from transitive_reasoning.paths import data_dir


@pytest.mark.parametrize(
    ("name", "size"), [("qasc", 926), ("bamboogle", 112), ("bamboogle_gibberish", 112)]
)
def test_dataset_sizes(name: str, size: int) -> None:
    assert len(load_dataset(name)) == size


def test_ids_are_prefixed_zero_padded_and_unique() -> None:
    instances = load_dataset("bamboogle")
    assert instances[0].id == "bamboogle-0000"
    assert instances[-1].id == "bamboogle-0111"
    assert len({i.id for i in instances}) == len(instances)


@pytest.mark.parametrize("name", sorted(DATASET_FILES))
def test_files_provide_exactly_the_declared_fields(name: str) -> None:
    for instance in load_dataset(name):
        for field in INSTANCE_FIELDS:
            value = getattr(instance, field)
            if field in DATASET_FIELDS[name]:
                assert isinstance(value, str) and value.strip(), (instance.id, field)
            else:
                assert value is None, (instance.id, field)


def test_demonstrations_are_not_test_questions() -> None:
    demos = load_instances(data_dir() / "bamboogle" / "demonstrations.json", prefix="demo")
    questions = {i.question.strip().lower() for i in load_dataset("bamboogle")}
    assert len(demos) == 3
    assert not any(d.question.strip().lower() in questions for d in demos)


def test_original_bamboogle_has_125_questions() -> None:
    original = load_instances(data_dir() / "bamboogle" / "original.json", prefix="orig")
    assert len(original) == 125
    assert original[0].fact1 is None


def test_gibberish_answers_differ_from_originals() -> None:
    changed = [i for i in load_dataset("bamboogle_gibberish") if i.answer != i.original_answer]
    assert len(changed) >= 100


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("hotpotqa")


def test_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"question": "q", "answer": "a", "fact 1": "old style"}]))
    with pytest.raises(ValueError, match="unknown keys"):
        load_instances(path, prefix="bad")


def test_instance_is_frozen(climate_instance: Instance) -> None:
    with pytest.raises(AttributeError):
        climate_instance.answer = "(A) storm"


QASC_RECORD = {
    "id": "3E7TUJ2EGCLQNOV1WEAJ2NN9ROPU9K",
    "question": {
        "stem": "Climate is generally described in terms of what?",
        "choices": [
            {"text": "sand", "label": "A"},
            {"text": "occurs over a wide range", "label": "B"},
            {"text": "forests", "label": "C"},
            {"text": "Global warming", "label": "D"},
            {"text": "rapid changes occur", "label": "E"},
            {"text": "local weather conditions", "label": "F"},
            {"text": "measure of motion", "label": "G"},
            {"text": "city life", "label": "H"},
        ],
    },
    "answerKey": "F",
    "fact1": "Climate is generally described in terms of temperature and moisture.",
    "fact2": (
        "Fire behavior is driven by local weather conditions such as winds, "
        "temperature and moisture."
    ),
    "combinedfact": "Climate is generally described in terms of local weather conditions.",
    "formatted_question": "unused",
}


def test_convert_qasc_record_matches_dev_json_format() -> None:
    expected = load_dataset("qasc")[0]
    converted = convert_qasc_record(QASC_RECORD)
    assert converted == {
        "question": expected.question,
        "choices": expected.choices,
        "fact1": expected.fact1,
        "fact2": expected.fact2,
        "deduction": expected.deduction,
        "answer": expected.answer,
    }


def test_convert_qasc_record_adds_period_to_deduction() -> None:
    record = {**QASC_RECORD, "combinedfact": "Water is wet"}
    assert convert_qasc_record(record)["deduction"] == "Therefore, water is wet."


def test_convert_qasc_writes_json_array(tmp_path: Path) -> None:
    source = tmp_path / "dev.jsonl"
    source.write_text(json.dumps(QASC_RECORD) + "\n" + json.dumps(QASC_RECORD) + "\n")
    target = tmp_path / "out" / "dev.json"
    assert convert_qasc(source, target) == 2
    assert len(load_instances(target, prefix="x")) == 2


def test_module_exposes_three_datasets() -> None:
    assert set(data.DATASET_FILES) == {"qasc", "bamboogle", "bamboogle_gibberish"}
