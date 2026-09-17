from __future__ import annotations

import pytest
import torch

from transitive_reasoning.config import ModelConfig
from transitive_reasoning.models import HuggingFaceBackend, require_cuda_for_8bit


@pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful without a GPU")
def test_eight_bit_without_cuda_fails_clearly() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        require_cuda_for_8bit()
    config = ModelConfig(name="m", hf_id="google/flan-t5-xxl", quantization="8bit", batch_size=1)
    with pytest.raises(RuntimeError, match="CUDA"):
        HuggingFaceBackend.from_config(config)
