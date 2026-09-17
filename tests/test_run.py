from __future__ import annotations

from tests.fakes import FakeBackend
from transitive_reasoning.config import load_experiment
from transitive_reasoning.data import load_dataset
from transitive_reasoning.run import run_experiment


def reply(answer: str) -> str:
    return f"\nDeduce: something.\nAnswer: {answer}\n"


def test_run_scores_predictions_in_file_order() -> None:
    instances = load_dataset("qasc")[:4]
    backend = FakeBackend(
        [
            reply(instances[0].answer),
            reply("(Z) wrong"),
            reply(instances[2].answer),
            reply("(Z) wrong"),
        ]
    )
    result = run_experiment(load_experiment("qasc/full"), backend, seed=1, limit=4)

    assert result.metrics.metric == "mc_accuracy"
    assert result.metrics.score == 0.5
    assert result.metrics.n == 4
    assert result.metrics.n_unparsed == 0
    assert [p.id for p in result.predictions] == [i.id for i in instances]
    assert result.predictions[1].predicted_answer == "(Z) wrong"
    assert result.predictions[1].score == 0.0
    assert result.predictions[0].predicted_deduction == "something."
    assert result.predictions[0].removed_words == []
    assert result.predictions[0].prompt is None
    assert len(backend.prompts) == 4
    assert backend.prompts[0].endswith(
        f"Question: {instances[0].question}\nAnswers: {instances[0].choices}\n"
        f"Fact 1: {instances[0].fact1}\nFact 2: {instances[0].fact2}\nSteps:"
    )


def test_run_manipulates_before_prompting_and_records_removed_words() -> None:
    original = load_dataset("qasc")[0]
    backend = FakeBackend([reply("(A) x")])
    result = run_experiment(load_experiment("qasc/f1q_ablation"), backend, limit=1)

    prediction = result.predictions[0]
    assert prediction.fact1 == "temperature and moisture"
    assert prediction.removed_words == [
        "Climate",
        "described",
        "generally",
        "in",
        "is",
        "of",
        "terms",
    ]
    assert prediction.fact1 in backend.prompts[0]
    assert original.fact1 not in backend.prompts[0]
    assert prediction.question == original.question


def test_run_counts_unparsed_generations_as_wrong() -> None:
    result = run_experiment(load_experiment("bamboogle/full"), FakeBackend(["garbage"]), limit=1)
    assert result.metrics.n_unparsed == 1
    assert result.metrics.score == 0.0
    assert result.predictions[0].predicted_answer == ""


def test_run_uses_gibberish_answer_as_reference() -> None:
    instance = load_dataset("bamboogle_gibberish")[0]
    backend = FakeBackend([reply(instance.answer)])
    result = run_experiment(load_experiment("bamboogle_gibberish/full"), backend, limit=1)
    assert result.metrics.score == 1.0
    assert result.predictions[0].answer == instance.answer


def test_run_is_deterministic_for_a_seed() -> None:
    experiment = load_experiment("qasc/full_shuffled")
    first = run_experiment(experiment, FakeBackend([reply("(A) x")] * 3), seed=7, limit=3)
    second = run_experiment(experiment, FakeBackend([reply("(A) x")] * 3), seed=7, limit=3)
    third = run_experiment(experiment, FakeBackend([reply("(A) x")] * 3), seed=8, limit=3)
    assert [p.fact1 for p in first.predictions] == [p.fact1 for p in second.predictions]
    assert [p.fact1 for p in first.predictions] != [p.fact1 for p in third.predictions]


def test_run_can_store_prompts() -> None:
    backend = FakeBackend([reply("(A) x")])
    result = run_experiment(load_experiment("qasc/qa"), backend, limit=1, save_prompts=True)
    assert result.predictions[0].prompt == backend.prompts[0]


def test_reference_answers_come_from_the_original_instances() -> None:
    """qasc/full_shuffled shuffles fact1/fact2 only, never answer, so this also holds for it."""
    instances = load_dataset("qasc")[:2]
    backend = FakeBackend([reply(instances[0].answer), reply(instances[1].answer)])
    result = run_experiment(load_experiment("qasc/full_shuffled"), backend, limit=2)
    assert result.metrics.score == 1.0
    assert [p.answer for p in result.predictions] == [i.answer for i in instances]
