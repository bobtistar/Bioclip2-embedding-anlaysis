#!/usr/bin/env python
"""Summarize Rare Species JSON outputs for model-to-model comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_MODELS = ["openclip-vitl14", "bioclip2"]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def _row_for_model(results_root: Path, model: str) -> Dict[str, Any]:
    model_dir = results_root / model
    exp1 = _load_json(model_dir / "exp1_geometry.json")
    exp2 = _load_json(model_dir / "exp2_rank_levels.json")
    exp3 = _load_json(model_dir / "exp3_counterfactuals.json")

    success = exp1["success_criteria"]
    latent = exp2["latent_taxonomy_probe"]
    preservation = exp3["preservation_ratio"]
    return {
        "model": model,
        "rq1_pass": success["passes_RQ1_threshold"],
        "intra_drop_pct": success["intra_drop_pct"],
        "inter_margin_ratio": success["inter_margin_ratio"],
        "silhouette_gain": success["silhouette_abs_gain"],
        "latent_real_silhouette": latent["real_silhouette"],
        "latent_z_score": latent["z_score"],
        "c2_sil_preservation": preservation["C2"]["silhouette"]["point"],
        "c3_sil_preservation": preservation["C3"]["silhouette"]["point"],
        "c4_sil_preservation": preservation["C4"]["silhouette"]["point"],
        "semantic_organizer_supported": exp3["semantic_organizer_supported"],
    }


def _print_markdown(rows: List[Dict[str, Any]]) -> None:
    columns = [
        "model",
        "rq1_pass",
        "intra_drop_pct",
        "inter_margin_ratio",
        "silhouette_gain",
        "latent_real_silhouette",
        "latent_z_score",
        "c2_sil_preservation",
        "c3_sil_preservation",
        "c4_sil_preservation",
        "semantic_organizer_supported",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="../results/rare_species")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser()
    rows = [_row_for_model(results_root, model) for model in args.models]
    _print_markdown(rows)


if __name__ == "__main__":
    main()
