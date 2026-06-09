"""iNat21 (또는 iNat21-mini) val.json → 우리 파이프라인용 CSV 변환기.

목적
----
iNaturalist 2021 챌린지의 val.json은 이미 7-rank Linnaean taxonomy가 들어 있다.
GBIF 같은 외부 API 호출 없이, 이 JSON을 그대로 우리 `data_loader.RealDatasetSpec`
포맷(file, kingdom, phylum, class, order, family, genus, species)으로 평탄화한다.

학부생을 위한 흐름
------------------
  1) val.json 로드 → categories / images / annotations 추출
  2) 각 category_id에 대해 (kingdom..genus + "Genus specific_epithet" species) 매핑
  3) 각 annotation을 (image relative path, 7-rank labels) 로 평탄화
  4) optional: kingdom 필터, 종/이미지 stratified 서브샘플 (메모리·시간 제한)
  5) CSV로 저장 → data_loader.load_real_dataset() 가 그대로 읽음

iNat21 디렉토리 가정
--------------------
    <inat_root>/
      ├─ val.json                       (또는 --json 으로 직접 지정)
      └─ val/<dir_name>/*.jpg          (val.json 의 file_name 이 'val/...' 형태)

iNat21-mini도 동일 — train_mini.json / val.json 만 다르므로 --json 인자로 처리.

서브샘플링 권장
---------------
val.json 전체는 100,000 images / 10,000 species. 본 파이프라인의 cosine distance
matrix는 N²×4B 메모리를 요구하므로 (예: N=100k → 40GB) 반드시 서브샘플 권장.

기본값: 5개 kingdom × 종당 10장, kingdom당 최대 200종 → 약 10,000 이미지.
이는 CUB-200(11,788) 과 비슷한 스케일이며 cosine matrix ~400MB로 RAM 친화적.

실행 예
-------
    # 전체 5 kingdom 균형 stratified (권장 기본):
    python inat21_build_metadata.py \
        --inat_root /workspace/inat21 \
        --json val.json \
        --out_csv /workspace/inat21/inat21_val_meta.csv

    # 단일 kingdom 확인용:
    python inat21_build_metadata.py \
        --inat_root /workspace/inat21 \
        --kingdoms Aves \
        --max_species_per_kingdom 1000 \
        --max_per_species 10 \
        --out_csv /workspace/inat21/inat21_val_aves.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


DEFAULT_KINGDOMS = ["Aves", "Insecta", "Plantae", "Fungi", "Reptilia"]


def _parse_categories(categories: List[dict]) -> Dict[int, Dict[str, str]]:
    """category_id -> {kingdom, phylum, class, order, family, genus, species, image_dir_name}.

    iNat21 categories already carry full taxonomy. We construct the binomial
    species name as "Genus specific_epithet" because that's the form BioCLIP2
    was trained with.
    """
    out: Dict[int, Dict[str, str]] = {}
    for c in categories:
        # required taxonomy fields
        try:
            kingdom = c["kingdom"]
            phylum = c["phylum"]
            cls = c["class"]
            order = c["order"]
            family = c["family"]
            genus = c["genus"]
            species_epi = c.get("specific_epithet") or c.get("species") or ""
        except KeyError as e:
            raise RuntimeError(
                f"category id={c.get('id')} is missing taxonomy field {e}; "
                f"is this really an iNat21 val.json?"
            )
        species = f"{genus} {species_epi}".strip()
        out[int(c["id"])] = {
            "kingdom": kingdom, "phylum": phylum, "class": cls,
            "order": order, "family": family, "genus": genus,
            "species": species,
            "image_dir_name": c.get("image_dir_name", ""),  # iNat21 puts images under this
        }
    return out


def _domain_key(t: Dict[str, str], domain_values: set) -> Optional[str]:
    """Return the matched domain name for record `t`, or None.

    A "domain" in our plan (Aves / Insecta / Plantae / Fungi / Actinopterygii / ...)
    mixes Linnaean ranks: Plantae/Fungi are kingdoms, Aves/Insecta/Reptilia are
    Linnaean *classes* (within Animalia). We accept a match at any rank, in
    decreasing specificity so "Aves" wins over "Animalia".
    """
    for rank in ("class", "phylum", "kingdom"):
        v = t.get(rank, "")
        if v and v in domain_values:
            return v
    return None


def _stratified_sample(
    rows: List[Tuple[str, Dict[str, str]]],
    domains: Optional[List[str]],
    max_species_per_domain: Optional[int],
    max_per_species: Optional[int],
    seed: int,
) -> List[Tuple[str, Dict[str, str]]]:
    """Per-domain stratified sampling of (file, tax) rows.

    Order of cuts:
      1) drop rows whose Linnaean class/phylum/kingdom is not in `domains`
      2) for each domain, randomly keep ≤ max_species_per_domain species
      3) for each surviving species, randomly keep ≤ max_per_species images
    """
    import random
    rng = random.Random(seed)

    if domains:
        allow = set(domains)
        rows = [(f, t) for f, t in rows if _domain_key(t, allow) is not None]

    # group by (matched-domain, species). If `domains` not given, fall back to
    # using class as the grouping key (so stratification is still meaningful).
    by_dom: Dict[str, Dict[str, List[Tuple[str, Dict[str, str]]]]] = defaultdict(lambda: defaultdict(list))
    allow_set = set(domains) if domains else None
    for f, t in rows:
        if allow_set is not None:
            dkey = _domain_key(t, allow_set)
        else:
            dkey = t.get("class") or t.get("phylum") or t.get("kingdom") or "_unknown"
        if dkey is None:
            continue
        by_dom[dkey][t["species"]].append((f, t))

    kept: List[Tuple[str, Dict[str, str]]] = []
    for dom, species_map in by_dom.items():
        species_list = sorted(species_map.keys())
        if max_species_per_domain and len(species_list) > max_species_per_domain:
            rng.shuffle(species_list)
            species_list = sorted(species_list[:max_species_per_domain])
        for sp in species_list:
            samples = species_map[sp]
            if max_per_species and len(samples) > max_per_species:
                rng.shuffle(samples)
                samples = samples[:max_per_species]
            kept.extend(samples)
    return kept


def build_inat21_csv(
    inat_root: str,
    json_name: str,
    out_csv: str,
    domains: Optional[List[str]],
    max_species_per_domain: Optional[int],
    max_per_species: Optional[int],
    min_per_species: int,
    seed: int,
) -> Tuple[int, int]:
    """Returns (num_species, num_images) written to out_csv."""
    json_path = os.path.join(inat_root, json_name) if not os.path.isabs(json_name) else json_name
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"iNat21 annotation JSON not found: {json_path}")

    print(f"[iNat21] loading {json_path} ...", flush=True)
    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    cats = _parse_categories(ann["categories"])
    print(f"[iNat21] {len(cats)} categories", flush=True)

    # image_id -> file_name
    img_id_to_file: Dict[int, str] = {int(im["id"]): im["file_name"] for im in ann["images"]}
    print(f"[iNat21] {len(img_id_to_file)} images, {len(ann['annotations'])} annotations", flush=True)

    # flatten
    rows: List[Tuple[str, Dict[str, str]]] = []
    missing_img = 0
    missing_cat = 0
    for a in ann["annotations"]:
        cid = int(a["category_id"])
        iid = int(a["image_id"])
        if cid not in cats:
            missing_cat += 1
            continue
        if iid not in img_id_to_file:
            missing_img += 1
            continue
        rows.append((img_id_to_file[iid], cats[cid]))
    if missing_img or missing_cat:
        print(f"[iNat21] dropped {missing_cat} bad cat-ids, {missing_img} bad img-ids", flush=True)

    # min-per-species filter BEFORE subsampling (so we don't sample down a class below the floor)
    counts: Dict[str, int] = defaultdict(int)
    for _, t in rows:
        counts[t["species"]] += 1
    rows = [(f, t) for f, t in rows if counts[t["species"]] >= min_per_species]

    # stratified subsample
    if domains or max_species_per_domain or max_per_species:
        rows = _stratified_sample(
            rows, domains=domains,
            max_species_per_domain=max_species_per_domain,
            max_per_species=max_per_species,
            seed=seed,
        )

    # write CSV
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "kingdom", "phylum", "class", "order", "family", "genus", "species"])
        for file_rel, t in rows:
            w.writerow([file_rel, t["kingdom"], t["phylum"], t["class"],
                        t["order"], t["family"], t["genus"], t["species"]])

    species_set = {t["species"] for _, t in rows}
    print(f"[iNat21] wrote {out_csv}: {len(species_set)} species, {len(rows)} images", flush=True)
    # per-domain breakdown (using the matched domain key)
    allow_set = set(domains) if domains else None
    by_dom_count: Dict[str, int] = defaultdict(int)
    by_dom_sp: Dict[str, set] = defaultdict(set)
    for _, t in rows:
        if allow_set is not None:
            dkey = _domain_key(t, allow_set) or "_other"
        else:
            dkey = t.get("class") or t.get("phylum") or t.get("kingdom") or "_unknown"
        by_dom_count[dkey] += 1
        by_dom_sp[dkey].add(t["species"])
    for k in sorted(by_dom_count):
        print(f"    {k:>15s}  {len(by_dom_sp[k]):>5d} species  {by_dom_count[k]:>6d} images", flush=True)
    return len(species_set), len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inat_root", required=True,
                   help="iNat21 root dir (contains val.json and val/<dir>/*.jpg)")
    p.add_argument("--json", default="val.json",
                   help="annotation JSON inside inat_root (val.json / train_mini.json / ...)")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--domains", "--kingdoms", dest="domains",
                   default=",".join(DEFAULT_KINGDOMS),
                   help="comma-separated domain names to keep (matched against class/phylum/"
                        "kingdom in that order), or 'all'. Default: 5 major domains. The legacy "
                        "alias --kingdoms is kept for backwards compatibility.")
    p.add_argument("--max_species_per_domain", "--max_species_per_kingdom",
                   dest="max_species_per_domain", type=int, default=200,
                   help="cap species per domain (default 200 → 5×200=1000 species @ 10/species = 10k)")
    p.add_argument("--max_per_species", type=int, default=10,
                   help="cap images per species (default 10)")
    p.add_argument("--min_per_species", type=int, default=5,
                   help="drop species with fewer than this many images before subsampling")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    domains = None if args.domains.lower() == "all" else [k.strip() for k in args.domains.split(",") if k.strip()]

    build_inat21_csv(
        inat_root=args.inat_root,
        json_name=args.json,
        out_csv=args.out_csv,
        domains=domains,
        max_species_per_domain=args.max_species_per_domain,
        max_per_species=args.max_per_species,
        min_per_species=args.min_per_species,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
