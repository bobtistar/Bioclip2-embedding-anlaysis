#!/usr/bin/env python
"""Export imageomics/rare-species into the CSV/image layout used by run_experiment.py.

Output layout:
  <out_dir>/
    images/
      00000.jpg
      00001.jpg
      ...
    taxonomy.csv
    prepare_rare_species.log
    manifest.json

The Rare Species dataset currently exposes the image column as ``file_name`` and
keeps the full scientific name in ``sciName``. The CSV emitted here uses the full
scientific name for the ``species`` field because the prompt generator expects
species-level labels such as "Passer domesticus", not only the epithet.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
CSV_COLUMNS = ["file", *RANKS]
DEFAULT_DATASET = "imageomics/rare-species"


def _setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("prepare_rare_species")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(out_dir / "prepare_rare_species.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _clean_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return " ".join(text.split())


def _full_species_label(row: Mapping[str, Any]) -> str:
    sci_name = _clean_field(row.get("sciName") or row.get("scientific_name"))
    if sci_name:
        return sci_name

    genus = _clean_field(row.get("genus"))
    species = _clean_field(row.get("species"))
    if genus and species and " " not in species and not species.startswith(f"{genus} "):
        return f"{genus} {species}"
    return species


def _taxonomy_from_row(row: Mapping[str, Any]) -> Tuple[Optional[Dict[str, str]], List[str]]:
    tax = {
        "kingdom": _clean_field(row.get("kingdom")),
        "phylum": _clean_field(row.get("phylum")),
        "class": _clean_field(row.get("class")),
        "order": _clean_field(row.get("order")),
        "family": _clean_field(row.get("family")),
        "genus": _clean_field(row.get("genus")),
        "species": _full_species_label(row),
    }
    missing = [rank for rank in RANKS if not tax[rank]]
    if missing:
        return None, missing
    return tax, []


def _find_image_value(row: Mapping[str, Any]) -> Any:
    for key in ("file_name", "image", "jpg", "jpeg"):
        value = row.get(key)
        if value is not None:
            return value
    return None


def _save_image(image_value: Any, out_path: Path) -> None:
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image_value, Image.Image):
        image = image_value
    elif isinstance(image_value, Mapping):
        if image_value.get("bytes") is not None:
            from io import BytesIO

            image = Image.open(BytesIO(image_value["bytes"]))
        elif image_value.get("path"):
            image = Image.open(image_value["path"])
        else:
            raise ValueError("image mapping does not contain bytes or path")
    elif isinstance(image_value, (str, os.PathLike)):
        image = Image.open(image_value)
    else:
        raise TypeError(f"unsupported image value type: {type(image_value).__name__}")

    with image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(out_path, format="JPEG", quality=95)


def _read_existing_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _summarize_rows(rows: Iterable[Mapping[str, str]]) -> Tuple[int, int, Counter]:
    row_list = list(rows)
    species_count = len({r["species"] for r in row_list})
    image_count = len(row_list)
    kingdom_counts = Counter(r["kingdom"] for r in row_list)
    return species_count, image_count, kingdom_counts


def _print_summary(logger: logging.Logger, rows: Iterable[Mapping[str, str]]) -> None:
    species_count, image_count, kingdom_counts = _summarize_rows(rows)
    logger.info("species: %d", species_count)
    logger.info("images: %d", image_count)
    logger.info("kingdom distribution: %s", dict(sorted(kingdom_counts.items())))


def _existing_export_complete(out_dir: Path, logger: logging.Logger) -> bool:
    csv_path = out_dir / "taxonomy.csv"
    image_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.json"
    if not csv_path.is_file() or not image_dir.is_dir():
        return False

    try:
        rows = _read_existing_rows(csv_path)
    except Exception as exc:
        logger.warning("existing taxonomy.csv could not be read (%s); rebuilding", exc)
        return False

    if not rows:
        logger.warning("existing taxonomy.csv is empty; rebuilding")
        return False

    missing_files = [r["file"] for r in rows if not (image_dir / r["file"]).is_file()]
    if missing_files:
        logger.warning(
            "existing export is incomplete: %d referenced image files are missing; rebuilding",
            len(missing_files),
        )
        return False

    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            if int(manifest.get("image_count", -1)) != len(rows):
                logger.warning("manifest image_count does not match taxonomy.csv; rebuilding")
                return False
        except Exception as exc:
            logger.warning("existing manifest could not be read (%s); rebuilding", exc)
            return False

    logger.info("existing Rare Species export found at %s -> skip", out_dir)
    _print_summary(logger, rows)
    return True


def _load_dataset(dataset_name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with: pip install datasets"
        ) from exc

    dataset_obj = load_dataset(dataset_name)
    if hasattr(dataset_obj, "keys"):
        if split not in dataset_obj:
            available = ", ".join(dataset_obj.keys())
            raise KeyError(f"split '{split}' not found in {dataset_name}; available: {available}")
        return dataset_obj[split]
    return dataset_obj


def export_rare_species(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    image_dir = out_dir / "images"
    csv_path = out_dir / "taxonomy.csv"
    manifest_path = out_dir / "manifest.json"
    logger = _setup_logger(out_dir)

    if not args.force and _existing_export_complete(out_dir, logger):
        return

    logger.info("loading dataset: %s (split=%s)", args.dataset, args.split)
    ds = _load_dataset(args.dataset, args.split)
    logger.info("dataset rows: %d", len(ds))

    rows: List[Dict[str, str]] = []
    skipped_missing_rank = 0
    skipped_image_error = 0
    missing_rank_counts: Counter = Counter()

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda x, **_: x  # type: ignore[assignment]

    for source_idx, row in enumerate(tqdm(ds, desc="export rare-species", unit="img")):
        tax, missing = _taxonomy_from_row(row)
        if tax is None:
            skipped_missing_rank += 1
            missing_rank_counts.update(missing)
            if skipped_missing_rank <= args.max_missing_logs:
                logger.warning("skip row %d: missing ranks=%s", source_idx, ",".join(missing))
            continue

        image_value = _find_image_value(row)
        if image_value is None:
            skipped_image_error += 1
            logger.warning("skip row %d: no image column found", source_idx)
            continue

        file_name = f"{len(rows):05d}.jpg"
        try:
            _save_image(image_value, image_dir / file_name)
        except Exception as exc:
            skipped_image_error += 1
            logger.warning("skip row %d: image save failed (%s)", source_idx, exc)
            continue

        rows.append({"file": file_name, **tax})

    if skipped_missing_rank > args.max_missing_logs:
        logger.warning(
            "suppressed %d additional missing-rank row logs",
            skipped_missing_rank - args.max_missing_logs,
        )
    if missing_rank_counts:
        logger.warning("missing-rank counts: %s", dict(sorted(missing_rank_counts.items())))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    species_count, image_count, kingdom_counts = _summarize_rows(rows)
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "species_count": species_count,
        "image_count": image_count,
        "kingdom_distribution": dict(sorted(kingdom_counts.items())),
        "skipped_missing_rank": skipped_missing_rank,
        "skipped_image_error": skipped_image_error,
        "csv": str(csv_path),
        "image_dir": str(image_dir),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("wrote CSV: %s", csv_path)
    logger.info("wrote images: %s", image_dir)
    logger.info("skipped missing-rank rows: %d", skipped_missing_rank)
    logger.info("skipped image-error rows: %d", skipped_image_error)
    _print_summary(logger, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, help="Directory for images/ and taxonomy.csv")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset name")
    parser.add_argument("--split", default="train", help="Dataset split to export")
    parser.add_argument("--force", action="store_true", help="Rebuild even when a complete export exists")
    parser.add_argument(
        "--max_missing_logs",
        type=int,
        default=20,
        help="Maximum number of per-row missing-rank warnings to print",
    )
    return parser.parse_args()


def main() -> None:
    export_rare_species(parse_args())


if __name__ == "__main__":
    main()
