"""Baseline (C0) wrapper. The 'model' is just the choice of prompt template;
the encoder is the same frozen backbone for every condition. This module exists
so the experiment loop has a uniform import surface for all conditions.
"""
from __future__ import annotations

from typing import List
from prompt_variants import TaxonomyRecord, generate_prompts


def flat_prompts(records: List[TaxonomyRecord]) -> List[str]:
    """Return the C0 flat prompts. Provided as a thin alias for clarity."""
    return generate_prompts(records, "C0")
