from __future__ import annotations

from collections.abc import Sequence


class FakeBackend:
    """Stands in for a model: replays canned generations and records the prompts it saw."""

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompts: Sequence[str]) -> list[str]:
        self.prompts.extend(prompts)
        if len(prompts) > len(self.replies):
            raise AssertionError(
                f"FakeBackend has {len(self.replies)} replies left but got {len(prompts)} prompts"
            )
        batch, self.replies = self.replies[: len(prompts)], self.replies[len(prompts) :]
        return batch
