from __future__ import annotations

import pytest

from transitive_reasoning.config import GenerationParams, ModelConfig
from transitive_reasoning.models import HuggingFaceBackend

pytestmark = pytest.mark.integration

PROMPTS = ["Question: a\nSteps:", "Question: bb bb bb\nSteps:", "Question: c\nSteps:"]


def tiny(hf_id: str) -> ModelConfig:
    return ModelConfig(
        name="tiny",
        hf_id=hf_id,
        quantization="none",
        batch_size=2,
        generation=GenerationParams(do_sample=False, num_beams=1, max_new_tokens=4),
    )


@pytest.mark.parametrize(
    ("hf_id", "is_causal"),
    [("hf-internal-testing/tiny-random-gpt2", True), ("hf-internal-testing/tiny-random-t5", False)],
)
def test_generate_returns_new_text_per_prompt(hf_id: str, is_causal: bool) -> None:
    backend = HuggingFaceBackend.from_config(tiny(hf_id))
    assert backend.is_causal is is_causal
    outputs = backend.generate(PROMPTS)
    assert len(outputs) == 3
    assert all(isinstance(output, str) for output in outputs)
    for prompt, output in zip(PROMPTS, outputs, strict=True):
        assert not output.startswith(prompt)
