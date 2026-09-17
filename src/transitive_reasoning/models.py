"""Hugging Face model backend for causal (LLaMA 2) and seq2seq (Flan-T5) models."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
)

from transitive_reasoning.config import ModelConfig


def require_cuda_for_8bit() -> None:
    """8-bit loading needs a CUDA device and the bitsandbytes package."""
    if not torch.cuda.is_available():
        raise RuntimeError("8-bit quantization needs CUDA; set quantization: none to run on CPU")
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError("8-bit quantization needs bitsandbytes; install the 'gpu' extra")


class HuggingFaceBackend:
    """Batched generation with a Hugging Face model, decoding only the newly generated tokens."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        is_causal: bool,
        batch_size: int,
        generation: GenerationConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.is_causal = is_causal
        self.batch_size = batch_size
        self.generation = generation

    @classmethod
    def from_config(cls, config: ModelConfig, cache_dir: Path | None = None) -> HuggingFaceBackend:
        cache = str(cache_dir) if cache_dir else None
        hf_config = AutoConfig.from_pretrained(config.hf_id, cache_dir=cache)
        is_causal = not getattr(hf_config, "is_encoder_decoder", False)

        tokenizer = AutoTokenizer.from_pretrained(config.hf_id, cache_dir=cache)
        if is_causal:
            tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

        load_kwargs: dict[str, Any] = {"cache_dir": cache}
        if config.quantization == "8bit":
            require_cuda_for_8bit()
            from transformers import BitsAndBytesConfig

            load_kwargs.update(
                quantization_config=BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                    load_in_8bit=True
                ),
                device_map="auto",
                dtype=torch.float16,
            )
        elif torch.cuda.is_available():
            load_kwargs.update(device_map="auto", dtype=torch.float16)

        loader = AutoModelForCausalLM if is_causal else AutoModelForSeq2SeqLM
        model = loader.from_pretrained(config.hf_id, **load_kwargs)
        model.eval()

        generation = GenerationConfig(  # type: ignore[no-untyped-call]
            **config.generation.model_dump(), pad_token_id=tokenizer.pad_token_id
        )
        return cls(
            tokenizer,
            model,
            is_causal=is_causal,
            batch_size=config.batch_size,
            generation=generation,
        )

    def generate(self, prompts: Sequence[str]) -> list[str]:
        outputs: list[str] = []
        batches = range(0, len(prompts), self.batch_size)
        for start in tqdm(
            batches, desc="generating", unit="batch", disable=len(prompts) <= self.batch_size
        ):
            outputs.extend(self._generate_batch(prompts[start : start + self.batch_size]))
        return outputs

    def _generate_batch(self, batch: Sequence[str]) -> list[str]:
        inputs = self.tokenizer(list(batch), return_tensors="pt", padding=True).to(
            self.model.device
        )
        with torch.no_grad():
            generated = self.model.generate(**inputs, generation_config=self.generation)
        if self.is_causal:
            generated = generated[:, inputs["input_ids"].shape[1] :]
        decoded: list[str] = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return decoded
