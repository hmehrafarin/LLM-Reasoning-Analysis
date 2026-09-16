from __future__ import annotations

import pytest

from transitive_reasoning.data import Instance


@pytest.fixture
def climate_instance() -> Instance:
    """The QASC example used in the paper's Table 6."""
    return Instance(
        id="qasc-test",
        question="What is described in terms of temperature and water in the air?",
        choices="(A) storm (B) climate (C) rain (D) wind (E) snow (F) heat (G) fog (H) ice",
        fact1="Climate is generally described in terms of temperature and moisture.",
        fact2="Clouds are made of moisture and the moisture is from the water evaporating.",
        deduction="Therefore, climate is generally described in terms of water in the air.",
        answer="(B) climate",
    )
