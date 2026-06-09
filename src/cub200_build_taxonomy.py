"""CUB-200-2011 데이터셋용 7-rank Linnaean taxonomy CSV 빌더.

목적
----
CUB-200-2011은 200개 새 종을 영어 일반명(예: "Black_footed_Albatross")으로만 제공한다.
본 스크립트는 GBIF (Global Biodiversity Information Facility) Backbone Taxonomy를
pygbif로 조회하여 각 클래스의 학명과 7-rank 분류(kingdom/phylum/class/order/family/
genus/species)를 자동으로 채워 넣는다.

----------------------
  1) CUB의 classes.txt를 파싱해서 200개 일반명을 얻는다.
  2) 각 일반명을 GBIF의 name_suggest로 검색 → 학명을 얻는다.
  3) 학명을 GBIF의 name_backbone으로 다시 검색 → 상위 6개 rank를 얻는다.
  4) GBIF 응답을 json 파일로 캐싱 → 두 번째 실행부터는 네트워크 호출 없이 즉시 완료.
  5) images.txt + image_class_labels.txt와 join하여 11,788장에 대한 메타 CSV를 작성.

CUB는 모두 새(Aves)이므로 매칭 실패 시 kingdom=Animalia / phylum=Chordata / class=Aves
기본값을 사용하고 species 자리에 일반명을 넣는다.

실행 예
-------
    python cub200_build_taxonomy.py \
        --cub_root D:/project/harness/CUB_200_2011 \
        --out_csv D:/project/harness/CUB_200_2011/cub_taxonomy.csv

옵션
----
    --cache PATH     : GBIF 응답 캐시 파일 경로 (기본: cub_gbif_cache.json)
    --max_classes N  : 처음 N개 클래스만 처리 (디버깅용)
    --sleep SEC      : GBIF API 호출 간 대기 시간 (기본 0.05초, rate-limit 회피)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_classes_txt(path: Path) -> List[Tuple[int, str]]:
    """classes.txt를 (class_id, normalized_common_name) 리스트로 변환.

    원본 한 줄 예: "1 001.Black_footed_Albatross"
    반환 예:       (1, "Black footed Albatross")
    """
    out: List[Tuple[int, str]] = []
    text = Path(path).read_text(encoding="utf-8").strip().splitlines()
    for line in text:
        cid_str, name = line.split(" ", 1)
        # "001.Black_footed_Albatross" → "Black_footed_Albatross"
        name = name.split(".", 1)[1] if "." in name else name
        # 공백 정규화
        name = name.replace("_", " ").strip()
        out.append((int(cid_str), name))
    return out


def query_gbif_lookup(common_name: str, sleep_sec: float) -> Optional[dict]:
    """GBIF name_lookup으로 일반명(또는 학명)을 검색하여 SPECIES 항목 1개를 반환.

    name_lookup은 학명과 vernacular name(일반명) 둘 다 검색한다. 반환된 record에
    이미 kingdom~genus 풀 taxonomy가 포함되어 있어 추가 backbone 호출이 불필요.

    선택 우선순위:
      1) class='Aves' + taxonomicStatus='ACCEPTED' + rank='SPECIES'
      2) class='Aves' + rank='SPECIES'
      3) taxonomicStatus='ACCEPTED' + rank='SPECIES' (Aves 무관)
      4) 그냥 첫 SPECIES

    Aves 필터로 동명이의(곤충/식물에 같은 일반명이 있는 경우)를 막는다.
    """
    from pygbif import species as gbif_species

    try:
        resp = gbif_species.name_lookup(q=common_name, rank="SPECIES", limit=10)
    except Exception as exc:
        print(f"  [GBIF name_lookup 실패] '{common_name}': {exc}")
        return None
    time.sleep(sleep_sec)

    results = (resp or {}).get("results", [])
    if not results:
        return None

    def _pick(predicate):
        for r in results:
            if predicate(r):
                return r
        return None

    return (
        _pick(lambda r: r.get("rank") == "SPECIES"
                       and r.get("taxonomicStatus") == "ACCEPTED"
                       and r.get("class") == "Aves")
        or _pick(lambda r: r.get("rank") == "SPECIES" and r.get("class") == "Aves")
        or _pick(lambda r: r.get("rank") == "SPECIES"
                          and r.get("taxonomicStatus") == "ACCEPTED")
        or _pick(lambda r: r.get("rank") == "SPECIES")
    )


def _name_variants(common_name: str) -> List[str]:
    """CUB의 underscore 정규화로 잃어버린 아포스트로피 등을 복원한 검색 후보 생성.

    예시:
      "Brewer Blackbird"     → ["Brewer Blackbird", "Brewer's Blackbird"]
      "Chuck will Widow"     → ["Chuck will Widow", "Chuck-will's-Widow",
                                "Chuck will's Widow"]

    GBIF가 가장 관대하게 받아주는 변형부터 차례로 시도한다.
    """
    out = [common_name]
    parts = common_name.split()
    if len(parts) >= 2:
        # "Brewer Blackbird" → "Brewer's Blackbird"
        possessive = parts[0] + "'s " + " ".join(parts[1:])
        out.append(possessive)
        # "Chuck will Widow" → "Chuck-will's-Widow" (CUB 특수 케이스)
        if len(parts) >= 3:
            hyphenated = parts[0] + "-" + parts[1] + "'s-" + "-".join(parts[2:])
            out.append(hyphenated)
    # 중복 제거하되 순서 유지
    seen = set()
    return [n for n in out if not (n in seen or seen.add(n))]


def query_gbif_family(family: str, sleep_sec: float) -> Optional[str]:
    """family 이름 → order 보강 (Aves 한정).

    species record에 order가 비어있을 때 family로 별도 lookup하여 보강한다.
    """
    from pygbif import species as gbif_species
    try:
        resp = gbif_species.name_lookup(q=family, rank="FAMILY", limit=5)
    except Exception:
        return None
    time.sleep(sleep_sec)
    for r in (resp or {}).get("results", []):
        if (r.get("rank") == "FAMILY"
                and r.get("class") == "Aves"
                and r.get("order")):
            return r["order"]
    return None


def normalize_kingdom(k: str) -> str:
    """GBIF가 일부 record에서 'Metazoa'를 쓰는 것을 'Animalia'로 통일."""
    if k and k.lower() == "metazoa":
        return "Animalia"
    return k or "Animalia"


def resolve_taxonomy(
    common_name: str,
    cache: Dict[str, dict],
    sleep_sec: float,
) -> Optional[Dict[str, str]]:
    """일반명 → 7-rank taxonomy dict 변환 (캐시 사용).

    여러 일반명 변형을 순서대로 시도하여 첫 매칭을 사용한다.
    매칭 실패 시 None 반환 (호출자가 fallback_taxonomy로 대체).
    """
    if common_name in cache:
        cached = cache[common_name]
        return cached if cached else None

    rec = None
    for variant in _name_variants(common_name):
        rec = query_gbif_lookup(variant, sleep_sec)
        if rec:
            break

    if not rec:
        cache[common_name] = None
        return None

    sci_name = rec.get("canonicalName") or rec.get("scientificName")
    result = {
        "kingdom":         normalize_kingdom(rec.get("kingdom", "Animalia")),
        "phylum":          rec.get("phylum", "Chordata"),
        "class":           rec.get("class", "Aves"),
        "order":           rec.get("order") or "unknown",
        "family":          rec.get("family") or "unknown",
        "genus":           rec.get("genus") or "unknown",
        "species":         rec.get("species") or sci_name or common_name,
        "scientific_name": sci_name or common_name,
    }
    cache[common_name] = result
    return result


def fill_missing_orders(
    class_to_taxonomy: Dict[int, Dict[str, str]],
    family_order_cache: Dict[str, str],
    sleep_sec: float,
) -> int:
    """order='unknown'인 클래스를 family로부터 보강. 보강된 개수를 반환."""
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it

    targets = [
        (cid, tax) for cid, tax in class_to_taxonomy.items()
        if tax.get("order") == "unknown" and tax.get("family") not in (None, "unknown")
    ]
    if not targets:
        return 0
    fixed = 0
    for cid, tax in tqdm(targets, desc="Fill missing orders", unit="cls"):
        fam = tax["family"]
        if fam in family_order_cache:
            order = family_order_cache[fam]
        else:
            order = query_gbif_family(fam, sleep_sec) or ""
            family_order_cache[fam] = order
        if order:
            tax["order"] = order
            fixed += 1
    return fixed


def fallback_taxonomy(common_name: str) -> Dict[str, str]:
    """매칭 실패한 클래스의 안전한 기본값. CUB 200종은 모두 새이므로 Aves를 기본으로."""
    # 일반명을 학명 자리에 그대로 (sentinel로 동작)
    placeholder = f"Aves_{re.sub(r'[^A-Za-z]', '_', common_name)}"
    return {
        "kingdom":         "Animalia",
        "phylum":          "Chordata",
        "class":           "Aves",
        "order":           "unknown",
        "family":          "unknown",
        "genus":           "unknown",
        "species":         placeholder,
        "scientific_name": placeholder,
    }


def load_cache(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"  [경고] 캐시 파일 {path}가 손상되어 무시합니다.")
        return {}


def save_cache(cache: Dict[str, dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False, default=str)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cub_root", required=True, type=Path,
                        help="CUB_200_2011 디렉토리 경로")
    parser.add_argument("--out_csv", required=True, type=Path,
                        help="출력 CSV 경로")
    parser.add_argument("--cache", default=None, type=Path,
                        help="GBIF 응답 캐시 (기본: out_csv와 동일 폴더의 cub_gbif_cache.json)")
    parser.add_argument("--max_classes", type=int, default=None,
                        help="처음 N개 클래스만 처리 (디버깅용)")
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="GBIF 호출 간 대기 (초). rate-limit이 의심되면 0.2로.")
    args = parser.parse_args()

    cub_root: Path = args.cub_root
    classes_txt = cub_root / "classes.txt"
    images_txt = cub_root / "images.txt"
    labels_txt = cub_root / "image_class_labels.txt"
    for p in [classes_txt, images_txt, labels_txt]:
        if not p.exists():
            print(f"[에러] 필수 파일이 없습니다: {p}")
            return 1

    # tqdm은 선택적 의존성 — 없으면 평문 출력
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it

    # 1) 클래스 파싱
    classes = parse_classes_txt(classes_txt)
    if args.max_classes:
        classes = classes[: args.max_classes]
    print(f"[1/5] Loaded {len(classes)} classes from classes.txt")

    # 2) GBIF lookup (캐시 사용)
    cache_path: Path = args.cache or args.out_csv.with_name("cub_gbif_cache.json")
    cache = load_cache(cache_path)
    cached_keys_before = len(cache)
    print(f"[2/5] Cache: {cached_keys_before} entries (path: {cache_path})")

    class_to_taxonomy: Dict[int, Dict[str, str]] = {}
    failed: List[str] = []
    for cid, common in tqdm(classes, desc="GBIF lookup", unit="cls"):
        tax = resolve_taxonomy(common, cache, args.sleep)
        if tax is None:
            failed.append(common)
            class_to_taxonomy[cid] = fallback_taxonomy(common)
        else:
            class_to_taxonomy[cid] = tax

    # 3) order='unknown' 보강 (family로부터 추가 조회)
    family_order_cache: Dict[str, str] = cache.setdefault("__family_orders__", {})
    if not isinstance(family_order_cache, dict):
        family_order_cache = {}
        cache["__family_orders__"] = family_order_cache
    n_fixed = fill_missing_orders(class_to_taxonomy, family_order_cache, args.sleep)
    if n_fixed:
        print(f"  → Filled {n_fixed} missing 'order' fields from family lookup.")

    # 캐시에 적용된 후처리 결과 반영 (failed → fallback도 같이 normalize)
    for cid in class_to_taxonomy:
        tax = class_to_taxonomy[cid]
        tax["kingdom"] = normalize_kingdom(tax.get("kingdom", "Animalia"))

    # 4) 캐시 저장
    save_cache(cache, cache_path)
    new_entries = len(cache) - cached_keys_before - (1 if "__family_orders__" in cache else 0)
    print(f"[3/5] Cache saved: {len(cache) - 1} species + family-order map "
          f"(new species: {new_entries})")

    # 4) 이미지 ↔ 클래스 매핑 로드
    img_id_to_path: Dict[int, str] = {}
    for line in images_txt.read_text(encoding="utf-8").strip().splitlines():
        iid, path = line.split(" ", 1)
        img_id_to_path[int(iid)] = path

    img_id_to_class: Dict[int, int] = {}
    for line in labels_txt.read_text(encoding="utf-8").strip().splitlines():
        iid, cid = line.split(" ", 1)
        img_id_to_class[int(iid)] = int(cid)

    # 5) CSV 작성
    out_csv: Path = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "kingdom", "phylum", "class", "order",
                        "family", "genus", "species", "scientific_name", "class_id"],
        )
        writer.writeheader()
        for iid, path in img_id_to_path.items():
            cid = img_id_to_class[iid]
            if cid not in class_to_taxonomy:
                continue  # max_classes 제한 적용 시
            tax = class_to_taxonomy[cid]
            writer.writerow({"file": path, **tax, "class_id": cid})
            n_written += 1

    # 6) 요약
    print(f"[4/5] CSV written: {out_csv}")
    print(f"[5/5] Summary:")
    print(f"  Total image rows:     {n_written}")
    print(f"  Mapped classes:       {len(classes) - len(failed)}/{len(classes)}")
    if failed:
        sample = failed[:10]
        more = f" ... (total {len(failed)})" if len(failed) > 10 else ""
        print(f"  Failed (Aves fallback): {sample}{more}")
    print()
    print("Next step:")
    print(f"  python run_experiment.py --csv {out_csv} \\")
    print(f"      --image_root {cub_root / 'images'} \\")
    print(f"      --model openclip-vitb32 --out ../results/cub200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
