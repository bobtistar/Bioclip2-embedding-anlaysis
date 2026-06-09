"""Data loader for BioCLIP2 hierarchical prompt analysis.

For the code skeleton we provide:
  1) A TOY synthetic dataset (no images required) — used for end-to-end sanity check.
  2) A directory-based real-data loader (PIL images + CSV with 7-rank taxonomy) —
     used when the user has downloaded a TreeOfLife subset or equivalent.

The toy dataset emulates 8 species across 4 taxonomic domains (Aves, Insecta,
Plantae, Fungi); 2 species per domain; 6 samples per species.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import os
import csv
import numpy as np

from prompt_variants import TaxonomyRecord


# ---------------------------------------------------------------------------
# Toy dataset
# ---------------------------------------------------------------------------

TOY_TAXONOMY: List[TaxonomyRecord] = [
    # Aves
    TaxonomyRecord("Animalia", "Chordata", "Aves", "Passeriformes",
                   "Passeridae", "Passer", "Passer domesticus"),
    TaxonomyRecord("Animalia", "Chordata", "Aves", "Passeriformes",
                   "Fringillidae", "Carduelis", "Carduelis carduelis"),
    # Insecta
    TaxonomyRecord("Animalia", "Arthropoda", "Insecta", "Lepidoptera",
                   "Nymphalidae", "Vanessa", "Vanessa cardui"),
    TaxonomyRecord("Animalia", "Arthropoda", "Insecta", "Coleoptera",
                   "Coccinellidae", "Coccinella", "Coccinella septempunctata"),
    # Plantae
    TaxonomyRecord("Plantae", "Tracheophyta", "Magnoliopsida", "Rosales",
                   "Rosaceae", "Rosa", "Rosa rugosa"),
    TaxonomyRecord("Plantae", "Tracheophyta", "Magnoliopsida", "Asterales",
                   "Asteraceae", "Helianthus", "Helianthus annuus"),
    # Fungi
    TaxonomyRecord("Fungi", "Basidiomycota", "Agaricomycetes", "Agaricales",
                   "Amanitaceae", "Amanita", "Amanita muscaria"),
    TaxonomyRecord("Fungi", "Basidiomycota", "Agaricomycetes", "Boletales",
                   "Boletaceae", "Boletus", "Boletus edulis"),
]


def get_toy_dataset(
    samples_per_species: int = 6,
    seed: int = 42,
) -> Tuple[List[TaxonomyRecord], np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """Return (records, labels, sample_records_repeated, tax_table).

    `records`: 8 unique species TaxonomyRecord objects (master list).
    `labels`: (N,) species index per sample.
    `sample_records_repeated`: list of TaxonomyRecord per sample (length N).
    `tax_table`: species_name -> 7-rank label list (for LCA/hierarchy metrics).
    """
    rng = np.random.default_rng(seed)
    records = TOY_TAXONOMY
    n_species = len(records)
    labels = np.repeat(np.arange(n_species), samples_per_species)
    sample_records = [records[i] for i in labels]
    tax_table = {r.species: r.labels() for r in records}
    return records, labels, sample_records, tax_table


def get_toy_images(labels: np.ndarray, image_size: int = 224, seed: int = 42):
    """Generate class-conditioned synthetic 'images' as 3xHxW tensors with normalized statistics.

    These are NOT realistic photos; they exist purely so the pipeline can call model.encode_image
    on a tensor without needing real image files. The class-conditioned channel mean ensures the
    encoder will return per-class structured embeddings (small but non-zero signal).
    """
    import torch
    rng = np.random.default_rng(seed)
    n = len(labels)
    imgs = torch.zeros(n, 3, image_size, image_size)
    # per-class color mean
    class_means = rng.uniform(0.2, 0.8, size=(int(labels.max()) + 1, 3))
    for i, c in enumerate(labels):
        for ch in range(3):
            base = class_means[c, ch]
            imgs[i, ch] = base + 0.05 * torch.randn(image_size, image_size)
    return imgs


# ---------------------------------------------------------------------------
# Real-data loader (CSV-based)
# ---------------------------------------------------------------------------

@dataclass
class RealDatasetSpec:
    image_root: str          # directory containing image files
    metadata_csv: str        # CSV with columns: file, kingdom, phylum, class, order, family, genus, species
    domain_filter: Optional[str] = None  # e.g., "Aves" to restrict to one class
    min_samples_per_species: int = 5


def load_real_dataset(
    spec: RealDatasetSpec,
) -> Tuple[List[TaxonomyRecord], np.ndarray, List[str], Dict[str, List[str]]]:
    """Load a real dataset described by a CSV. Returns:
      records:       unique species TaxonomyRecord list
      labels:        (N,) integer species index
      image_paths:   list of absolute image paths
      tax_table:     species -> 7-rank labels
    """
    if not os.path.isfile(spec.metadata_csv):
        raise FileNotFoundError(spec.metadata_csv)

    rows = []
    with open(spec.metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if spec.domain_filter and r.get("class") != spec.domain_filter:
                continue
            rows.append(r)

    # group by species, filter min samples
    by_species: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_species.setdefault(r["species"], []).append(r)
    by_species = {s: rs for s, rs in by_species.items() if len(rs) >= spec.min_samples_per_species}

    records: List[TaxonomyRecord] = []
    image_paths: List[str] = []
    labels: List[int] = []
    for idx, (sp, rs) in enumerate(sorted(by_species.items())):
        rec = TaxonomyRecord(
            kingdom=rs[0]["kingdom"], phylum=rs[0]["phylum"], cls=rs[0]["class"],
            order=rs[0]["order"], family=rs[0]["family"], genus=rs[0]["genus"],
            species=sp,
        )
        records.append(rec)
        for r in rs:
            image_paths.append(os.path.join(spec.image_root, r["file"]))
            labels.append(idx)

    tax_table = {r.species: r.labels() for r in records}
    return records, np.array(labels), image_paths, tax_table


if __name__ == "__main__":
    recs, labs, samp, tax = get_toy_dataset()
    print(f"toy: {len(recs)} species, {len(labs)} samples")
    print("first record:", recs[0])
    print("tax_table for Rosa rugosa:", tax["Rosa rugosa"])
