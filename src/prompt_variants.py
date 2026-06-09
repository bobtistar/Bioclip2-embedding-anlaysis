"""Prompt variant generators for the 5 counterfactual conditions (C0..C4).

Conditions
----------
C0: flat species-only            -- "a photo of {species}"
C1: normal hierarchical 7-rank   -- "a photo of {kingdom} {phylum} {class} {order} {family} {genus} {species}"
C2: random-token hierarchical    -- "a photo of tax0 tax1 tax2 tax3 tax4 tax5 {species}" (structure preserved, content meaningless)
C3: shuffled hierarchical        -- C1 with upper-6 rank labels replaced by labels from random other species (structure broken)
C4: hierarchical word-bag        -- C1 labels randomly permuted each call (order info removed)

All generators are deterministic given a numpy.random.Generator (rng) so experiments are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np


RANKS: List[str] = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


@dataclass
class TaxonomyRecord:
    """A single record holding the 7-rank Linnaean taxonomy of one species."""

    kingdom: str
    phylum: str
    cls: str  # 'class' is a Python keyword; we store as 'cls'
    order: str
    family: str
    genus: str
    species: str

    def labels(self) -> List[str]:
        return [self.kingdom, self.phylum, self.cls, self.order, self.family, self.genus, self.species]


# Placeholder tokens for C2. Chosen to be unlikely to appear in BPE vocabulary as
# semantically meaningful subwords. They are stable across calls.
C2_PLACEHOLDERS: List[str] = ["taxK", "taxP", "taxC", "taxO", "taxF", "taxG"]


def prompt_C0_flat(record: TaxonomyRecord) -> str:
    return f"a photo of {record.species}"


def prompt_C1_hierarchical(record: TaxonomyRecord) -> str:
    return (
        f"a photo of {record.kingdom} {record.phylum} {record.cls} "
        f"{record.order} {record.family} {record.genus} {record.species}"
    )


def prompt_C2_random_tokens(record: TaxonomyRecord) -> str:
    """Structure preserved, content meaningless. Same token count as C1."""
    placeholders = C2_PLACEHOLDERS  # 6 placeholders for 6 upper ranks
    return f"a photo of {' '.join(placeholders)} {record.species}"


def prompt_C3_shuffled(
    record: TaxonomyRecord,
    all_records: List[TaxonomyRecord],
    rng: np.random.Generator,
) -> str:
    """Replace upper-6 rank labels with labels from random OTHER species.
    Each rank is sampled independently to maximally break hierarchical consistency.
    Species token is preserved (otherwise we cannot label it).
    """
    n = len(all_records)
    if n <= 1:
        # Fallback: just shuffle within the single record's own labels (degenerate)
        labels = record.labels()[:6]
        rng.shuffle(labels)
        upper = labels
    else:
        upper: List[str] = []
        for rank_idx in range(6):  # kingdom..genus
            # sample index != self
            idx = int(rng.integers(0, n))
            # ensure different than self by retry (cheap)
            tries = 0
            while all_records[idx] is record and tries < 5:
                idx = int(rng.integers(0, n))
                tries += 1
            upper.append(all_records[idx].labels()[rank_idx])
    return f"a photo of {' '.join(upper)} {record.species}"


def prompt_C4_word_bag(record: TaxonomyRecord, rng: np.random.Generator) -> str:
    """Hierarchy labels but order is randomized each call. Vocabulary preserved."""
    labels = record.labels()[:]
    rng.shuffle(labels)
    return f"a photo of {' '.join(labels)}"


PROMPT_FNS: Dict[str, callable] = {
    "C0": prompt_C0_flat,
    "C1": prompt_C1_hierarchical,
    "C2": prompt_C2_random_tokens,
    "C3": prompt_C3_shuffled,
    "C4": prompt_C4_word_bag,
}


def generate_prompts(
    records: List[TaxonomyRecord],
    condition: str,
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """Generate one prompt per record under the given condition.

    Parameters
    ----------
    records : list of TaxonomyRecord
    condition : one of C0, C1, C2, C3, C4
    rng : numpy random generator for stochastic conditions (C3, C4)
    """
    if condition not in PROMPT_FNS:
        raise ValueError(f"Unknown condition {condition}, expected one of {list(PROMPT_FNS)}")

    if rng is None:
        rng = np.random.default_rng(0)

    prompts: List[str] = []
    for r in records:
        if condition == "C0":
            prompts.append(prompt_C0_flat(r))
        elif condition == "C1":
            prompts.append(prompt_C1_hierarchical(r))
        elif condition == "C2":
            prompts.append(prompt_C2_random_tokens(r))
        elif condition == "C3":
            prompts.append(prompt_C3_shuffled(r, records, rng))
        elif condition == "C4":
            prompts.append(prompt_C4_word_bag(r, rng))
    return prompts


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(42)
    demo = [
        TaxonomyRecord("Animalia", "Chordata", "Aves", "Passeriformes",
                       "Passeridae", "Passer", "Passer domesticus"),
        TaxonomyRecord("Animalia", "Chordata", "Aves", "Passeriformes",
                       "Fringillidae", "Carduelis", "Carduelis carduelis"),
        TaxonomyRecord("Plantae", "Tracheophyta", "Magnoliopsida", "Rosales",
                       "Rosaceae", "Rosa", "Rosa rugosa"),
    ]   
    for cond in ["C0", "C1", "C2", "C3", "C4"]:
        ps = generate_prompts(demo, cond, rng)
        print(f"-- {cond} --")
        for p in ps:
            print(" ", p)
