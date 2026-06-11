#!/usr/bin/env python3
"""Generate paper figures from results/*/exp2_rank_levels.json.

Headline figure: per-rank delta-silhouette (C1 - C0) for BioCLIP2 vs OpenCLIP,
shown for both datasets (iNat21, Rare Species). Makes the sign-flip at species
and OpenCLIP's early collapse visible at a glance.

Run from repo root:  python3 paper/make_figures.py
Outputs: paper/figures/delta_silhouette_by_rank.{pdf,png}
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO, "paper", "figures")
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

PANELS = [
    ("iNat21", "inat21_bioclip2", "inat21_openclip-vitl14"),
    ("Rare Species", "rare_species_bioclip2", "rare_species_openclip-vitl14"),
]


def load_deltas(run_dir):
    path = os.path.join(REPO, "results", run_dir, "exp2_rank_levels.json")
    with open(path) as f:
        d = json.load(f)
    out = {}
    for r in RANKS:
        x = d.get(r, {})
        out[r] = None if "skipped" in x else x["delta_silhouette_C1_minus_C0"]
    return out


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)

    for ax, (title, bio_dir, oc_dir) in zip(axes, PANELS):
        bio = load_deltas(bio_dir)
        oc = load_deltas(oc_dir)
        x = np.arange(len(RANKS))
        w = 0.38
        bvals = [bio[r] if bio[r] is not None else np.nan for r in RANKS]
        ovals = [oc[r] if oc[r] is not None else np.nan for r in RANKS]
        ax.bar(x - w / 2, bvals, w, label="BioCLIP2", color="#1f77b4")
        ax.bar(x + w / 2, ovals, w, label="OpenCLIP", color="#ff7f0e")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([r[:3].capitalize() for r in RANKS], rotation=0, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        # mark skipped (kingdom in rare species)
        for i, r in enumerate(RANKS):
            if bio[r] is None and oc[r] is None:
                ax.text(i, 0.005, "n/a", ha="center", va="bottom", fontsize=7,
                        color="gray", rotation=90)

    axes[0].set_ylabel(r"$\Delta$silhouette  (C1 $-$ C0)", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle(
        r"Hierarchy effect by taxonomic rank: positive = hierarchy improves organization",
        fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"delta_silhouette_by_rank.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote", out)


if __name__ == "__main__":
    main()
