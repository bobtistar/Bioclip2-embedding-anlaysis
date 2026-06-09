"""Main experiment pipeline for BioCLIP2 hierarchical prompt analysis.

Runs Exp1 (RQ1), Exp2 (RQ2 single-model), Exp3 (RQ3 counterfactuals), and prints a
summary JSON. Real BioCLIP2/BioCLIP weights are loaded if the `--model` flag points
to them and the network/HF cache is available; otherwise the pipeline falls back to
either OpenCLIP (real but smaller) or synthetic mock embeddings.

Usage
-----
    python run_experiment.py --model openclip-vitb32 --toy --seed 42 --out ../results
    python run_experiment.py --model bioclip2 --csv data/treeoflife_eval.csv --out ../results
    python run_experiment.py --mock                          # fully offline mock run

Outputs
-------
  <out>/exp1_geometry.json
  <out>/exp2_rank_levels.json
  <out>/exp3_counterfactuals.json
  <out>/run_log.txt
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Ensure local imports work when run from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_variants import generate_prompts, TaxonomyRecord, RANKS
from metrics import (
    intra_class_variance, inter_class_margin, silhouette_cosine,
    cosine_distance_matrix,
    rankme, uniformity, knn_purity_at_k,
    paired_permutation_test, bootstrap_ci, cohens_d_paired,
    plot_embeddings, plot_comparison_grid,
)
from data_loader import get_toy_dataset, get_toy_images, load_real_dataset, RealDatasetSpec
from extract_embeddings import (
    load_model, encode_images, encode_texts, mock_image_embeddings, mock_text_embeddings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def fuse_image_text(img_emb: np.ndarray, txt_emb: np.ndarray, mode: str = "concat_norm") -> np.ndarray:
    """Combine image and text embeddings into a single per-sample vector.

    mode='concat_norm' : per-modality L2 normalize then concat (preserves both modalities)
    mode='image_only'  : ignore text (used for sanity check)
    mode='text_anchor' : project image onto text anchor (subtract image-text mean alignment)
    """
    if mode == "image_only":
        return img_emb
    if mode == "concat_norm":
        i = img_emb / (np.linalg.norm(img_emb, axis=1, keepdims=True) + 1e-12)
        if txt_emb.shape[1] == 1 and (txt_emb == 0).all():
            return i
        t = txt_emb / (np.linalg.norm(txt_emb, axis=1, keepdims=True) + 1e-12)
        return np.concatenate([i, t], axis=1)
    raise ValueError(mode)


def geometric_metrics(Z: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict[str, float]:
    return {
        "intra_var": intra_class_variance(Z, y),
        "inter_margin": inter_class_margin(Z, y),
        "silhouette": silhouette_cosine(Z, y),
        "rankme": rankme(Z),
        "uniformity": uniformity(Z),
        "knn_purity@10": knn_purity_at_k(Z, y, k=min(10, max(1, len(Z) - 1))),
    }


def zero_shot_accuracy(records, img_emb, labels, encoder_state, cond, rng) -> Optional[float]:
    """Top-1 zero-shot accuracy for one prompt condition.

    Builds ONE text prototype per class (species) and assigns each image to the
    nearest prototype by cosine similarity. Only meaningful with a real shared-space
    text encoder, so returns None when no encoder is available, when the condition
    has no text (C5), or when text/image dims do not match (e.g., mock embeddings).
    """
    if encoder_state is None:
        return None
    prompts = generate_prompts(records, cond, rng)  # one prompt per class
    if any(p is None for p in prompts):
        return None
    txt_emb = encode_texts(encoder_state["model"], encoder_state["tokenizer"],
                           prompts, device=encoder_state["device"])
    if txt_emb.shape[1] != img_emb.shape[1]:
        return None
    I = img_emb / (np.linalg.norm(img_emb, axis=1, keepdims=True) + 1e-12)
    T = txt_emb / (np.linalg.norm(txt_emb, axis=1, keepdims=True) + 1e-12)
    pred = (I @ T.T).argmax(axis=1)
    return float((pred == labels).mean())


def _pick_color_rank(sample_records):
    """Choose a taxonomic rank with a legible number of classes (2..20) for coloring."""
    for ri in range(len(RANKS)):  # kingdom -> species
        vals = {r.labels()[ri] for r in sample_records}
        if 2 <= len(vals) <= 20:
            return ri, RANKS[ri]
    return len(RANKS) - 2, RANKS[-2]  # fallback: genus


def save_tsne(img_emb, sample_records, out_path, seed=42, max_points=2000) -> Optional[str]:
    """Save a 2-D t-SNE scatter of image embeddings, colored by a coarse rank.

    Optional/best-effort: silently returns None if sklearn/matplotlib are missing.
    Subsamples to `max_points` for speed.
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    n = len(img_emb)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(max_points, n), replace=False)
    Z = img_emb[idx]
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    sub_records = [sample_records[i] for i in idx]
    ri, rname = _pick_color_rank(sub_records)
    color_labels = np.array([r.labels()[ri] for r in sub_records])
    classes, inv = np.unique(color_labels, return_inverse=True)
    emb2d = TSNE(n_components=2, init="pca", metric="cosine", random_state=seed,
                 perplexity=min(30, max(5, len(Z) - 1))).fit_transform(Z)
    plt.figure(figsize=(7, 6))
    for ci, c in enumerate(classes):
        m = inv == ci
        plt.scatter(emb2d[m, 0], emb2d[m, 1], s=6, alpha=0.6, label=str(c))
    plt.legend(title=rname, fontsize=6, markerscale=2, loc="best", ncol=2)
    plt.title(f"t-SNE of image embeddings (colored by {rname})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Per-condition pipeline
# ---------------------------------------------------------------------------

def condition_embeddings(
    cond: str,
    records: List[TaxonomyRecord],
    sample_records: List[TaxonomyRecord],
    img_emb: np.ndarray,
    encoder_state: Optional[dict],
    rng: np.random.Generator,
    mock: bool,
    seed: int = 42,
) -> np.ndarray:
    prompts = generate_prompts(sample_records, cond, rng)
    if mock or encoder_state is None:
        # Use a hashed mock so distinct prompt strings produce distinct vectors;
        # this preserves the *information* dimension of the experiment but the
        # absolute numbers are NOT meaningful.
        txt_emb = mock_text_embeddings(prompts, dim=128, seed=seed)
    else:
        model = encoder_state["model"]
        tokenizer = encoder_state["tokenizer"]
        txt_emb = encode_texts(model, tokenizer, prompts, device=encoder_state["device"])

    return fuse_image_text(img_emb, txt_emb, mode="concat_norm")


def run_condition(
    cond: str,
    records: List[TaxonomyRecord],
    sample_records: List[TaxonomyRecord],
    labels: np.ndarray,
    img_emb: np.ndarray,
    encoder_state: Optional[dict],
    rng: np.random.Generator,
    mock: bool,
    seed: int = 42,
) -> Dict[str, float]:
    """Run one prompt condition and return geometric metrics."""
    Z = condition_embeddings(cond, records, sample_records, img_emb, encoder_state, rng, mock, seed=seed)
    return geometric_metrics(Z, labels, seed=seed)


# ---------------------------------------------------------------------------
# Experiment 1 (RQ1): flat vs hierarchical, with paired permutation test
# ---------------------------------------------------------------------------

def experiment_1(records, sample_records, labels, img_emb, encoder_state, mock, n_seeds=5, seed: int = 42):
    results: Dict[str, dict] = {"C0": {}, "C1": {}, "stats": {}}
    metric_keys = ["intra_var", "inter_margin", "silhouette", "rankme", "uniformity", "knn_purity@10"]
    per_seed = {"C0": {k: [] for k in metric_keys}, "C1": {k: [] for k in metric_keys}}
    vis_embeddings: Dict[str, np.ndarray] = {}

    for s_idx in range(n_seeds):
        rng = np.random.default_rng(42 + s_idx)
        # add tiny noise to image embeddings to simulate stochastic sampling
        noise = 1e-3 * np.random.default_rng(seed + s_idx).normal(0, 1, size=img_emb.shape)
        z = img_emb + noise.astype(img_emb.dtype)
        for cond in ["C0", "C1"]:
            Z_cond = condition_embeddings(cond, records, sample_records, z, encoder_state, rng, mock, seed=seed)
            m = geometric_metrics(Z_cond, labels, seed=seed)
            if s_idx == n_seeds - 1:
                vis_embeddings[cond] = Z_cond
            for k in metric_keys:
                per_seed[cond][k].append(m[k])

    for cond in ["C0", "C1"]:
        results[cond] = {k: {"mean": float(np.mean(per_seed[cond][k])),
                              "std": float(np.std(per_seed[cond][k]))}
                          for k in metric_keys}

    # paired permutation test on each metric
    rng_perm = np.random.default_rng(7)
    rng_boot = np.random.default_rng(13)
    for k in metric_keys:
        a = np.array(per_seed["C1"][k])
        b = np.array(per_seed["C0"][k])
        stat = paired_permutation_test(a, b, n_perm=1000, rng=rng_perm)
        _, ci_lo, ci_hi = bootstrap_ci(a - b, n_boot=1000, rng=rng_boot)
        stat["ci_lo"] = ci_lo
        stat["ci_hi"] = ci_hi
        d = cohens_d_paired(a, b)
        results["stats"][k] = {**stat, "cohens_d": d}

    # success criteria check
    intra_drop_pct = (results["C0"]["intra_var"]["mean"] - results["C1"]["intra_var"]["mean"]) / max(
        results["C0"]["intra_var"]["mean"], 1e-8) * 100
    margin_ratio = results["C1"]["inter_margin"]["mean"] / max(results["C0"]["inter_margin"]["mean"], 1e-8)
    sil_diff = results["C1"]["silhouette"]["mean"] - results["C0"]["silhouette"]["mean"]
    success = (intra_drop_pct >= 10.0) and (margin_ratio >= 1.2) and (sil_diff >= 0.05)
    results["success_criteria"] = {
        "intra_drop_pct": float(intra_drop_pct),
        "inter_margin_ratio": float(margin_ratio),
        "silhouette_abs_gain": float(sil_diff),
        "passes_RQ1_threshold": bool(success),
    }
    results["_vis_embeddings"] = vis_embeddings
    return results


# ---------------------------------------------------------------------------
# Experiment 2 (RQ2): rank-level effect, single model variant
# ---------------------------------------------------------------------------

def experiment_2(records, sample_records, labels, img_emb, tax_table, encoder_state, mock, seed: int = 42):
    """For each rank in {species..kingdom}, recompute labels at that rank and measure
    silhouette/MI/kNN purity. Compares C0 vs C1 prompt at each rank.
    """
    results: Dict[str, dict] = {}
    vis_embeddings: Dict[str, dict] = {}

    rng = np.random.default_rng(42)
    for rank_idx, rank_name in enumerate(RANKS):
        # build rank label per sample
        sample_rank = np.array([tax_table[sample_records[i].species][rank_idx]
                                 for i in range(len(sample_records))])
        _, rank_int = np.unique(sample_rank, return_inverse=True)
        # require >= 2 classes with >= 2 samples
        cls, cnt = np.unique(rank_int, return_counts=True)
        if len(cls) < 2 or cnt.min() < 2:
            results[rank_name] = {"skipped": "degenerate at this rank"}
            continue

        results[rank_name] = {}
        rank_vis: Dict[str, np.ndarray] = {}
        for cond in ["C0", "C1"]:
            Z_cond = condition_embeddings(cond, records, sample_records, img_emb,
                                          encoder_state, rng, mock, seed=seed)
            m = geometric_metrics(Z_cond, rank_int, seed=seed)
            results[rank_name][cond] = m
            rank_vis[cond] = Z_cond
        results[rank_name]["delta_silhouette_C1_minus_C0"] = (
            results[rank_name]["C1"]["silhouette"] - results[rank_name]["C0"]["silhouette"]
        )
        vis_embeddings[rank_name] = {"embeddings": rank_vis, "y": rank_int}

    # latent taxonomy probing: random-taxonomy permutation control at species rank.
    # Precompute the cosine-distance matrix once and reuse for 50 permutations + the
    # real-label silhouette — sklearn otherwise rebuilds the N×N distance on every call.
    rng2 = np.random.default_rng(7)
    D_cos = cosine_distance_matrix(img_emb)
    perm_sil = []
    for _ in range(50):
        perm = rng2.permutation(labels)
        perm_sil.append(silhouette_cosine(img_emb, perm, precomputed_distance=D_cos))
    real_sil = silhouette_cosine(img_emb, labels, precomputed_distance=D_cos)
    del D_cos  # ~556MB at N=12k; free before next experiment
    results["latent_taxonomy_probe"] = {
        "real_silhouette": float(real_sil),
        "random_taxonomy_mean": float(np.mean(perm_sil)),
        "random_taxonomy_std": float(np.std(perm_sil)),
        "z_score": float((real_sil - np.mean(perm_sil)) / (np.std(perm_sil) + 1e-8)),
    }
    results["_vis_embeddings"] = vis_embeddings
    return results


# ---------------------------------------------------------------------------
# Experiment 3 (RQ3): 5-condition counterfactual ablation
# ---------------------------------------------------------------------------

def experiment_3(records, sample_records, labels, img_emb, encoder_state, mock, n_seeds=5, seed: int = 42):
    metric_keys = ["intra_var", "inter_margin", "silhouette"]
    conditions = ["C0", "C1", "C2", "C3", "C4"]
    per_seed = {c: {k: [] for k in metric_keys} for c in conditions}
    vis_embeddings: Dict[str, np.ndarray] = {}
    for s_idx in range(n_seeds):
        rng = np.random.default_rng(100 + s_idx)
        noise = 1e-3 * np.random.default_rng(seed + s_idx).normal(0, 1, size=img_emb.shape)
        z = img_emb + noise.astype(img_emb.dtype)
        for cond in conditions:
            Z_cond = condition_embeddings(cond, records, sample_records, z, encoder_state, rng, mock, seed=seed)
            m = geometric_metrics(Z_cond, labels, seed=seed)
            if s_idx == n_seeds - 1:
                vis_embeddings[cond] = Z_cond
            for k in metric_keys:
                per_seed[cond][k].append(m[k])

    means = {c: {k: float(np.mean(per_seed[c][k])) for k in metric_keys} for c in conditions}
    stds = {c: {k: float(np.std(per_seed[c][k])) for k in metric_keys} for c in conditions}

    # Effect preservation ratios for C2, C3, C4 relative to (C1 - C0)
    rng_boot = np.random.default_rng(7)
    preservation: Dict[str, Dict[str, dict]] = {}
    for cond in ["C2", "C3", "C4"]:
        preservation[cond] = {}
        for k in metric_keys:
            # bootstrap over seeds (small n, but illustrative)
            arr = np.array(per_seed[cond][k]) - np.array(per_seed["C0"][k])
            denom = np.array(per_seed["C1"][k]) - np.array(per_seed["C0"][k])
            # paired ratio per seed
            ratios = []
            for r1, r2 in zip(arr, denom):
                if abs(r2) > 1e-8:
                    ratios.append(r1 / r2)
            if ratios:
                pt, lo, hi = bootstrap_ci(np.array(ratios), n_boot=1000, rng=rng_boot)
            else:
                pt = lo = hi = float("nan")
            preservation[cond][k] = {"point": pt, "ci_lo": lo, "ci_hi": hi}

    # success criteria: silhouette as the headline metric
    c2_sil = preservation["C2"]["silhouette"]
    c3_sil = preservation["C3"]["silhouette"]
    semantic_organizer_support = (
        (c2_sil["point"] >= 0.5 and c2_sil["ci_lo"] >= 0.3)
        and (c3_sil["point"] <= 0.2 and c3_sil["ci_hi"] <= 0.4)
    )
    return {
        "means_by_condition": means,
        "stds_by_condition": stds,
        "preservation_ratio": preservation,
        "semantic_organizer_supported": bool(semantic_organizer_support),
        "_vis_embeddings": vis_embeddings,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _autodetect_device(prefer: Optional[str]) -> str:
    """device 인자가 None이면 CUDA -> MPS -> CPU 순서로 자동 결정."""
    if prefer:
        return prefer
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if (hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()):
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def _iter_parent_paths(start: Path):
    yield start
    yield from start.parents


def _load_env_file(path: Optional[str], disabled: bool = False) -> Optional[str]:
    """Load simple KEY=VALUE entries from .env without overriding existing env vars."""
    if disabled:
        return None

    candidates: List[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    else:
        starts = [Path.cwd(), Path(__file__).resolve().parent]
        seen = set()
        for start in starts:
            for parent in _iter_parent_paths(start):
                env_path = parent / ".env"
                if env_path not in seen:
                    candidates.append(env_path)
                    seen.add(env_path)

    selected = next((p for p in candidates if p.is_file()), None)
    if selected is None:
        if path:
            raise FileNotFoundError(f".env file not found: {path}")
        return None

    with selected.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    return str(selected)


def _configure_hf_token(args) -> Optional[str]:
    """Load an HF token from CLI/file/prompt/env and expose it to HuggingFace Hub."""
    source = None
    token = args.hf_token
    if token:
        source = "--hf_token"
    elif args.hf_token_file:
        token_path = os.path.expanduser(args.hf_token_file)
        with open(token_path, "r", encoding="utf-8") as f:
            token = f.read().strip()
        source = "--hf_token_file"
    elif args.hf_token_prompt:
        token = getpass.getpass("Hugging Face token: ").strip()
        source = "--hf_token_prompt"
    else:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            source = "environment"

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return source


def _redacted_args(args) -> Dict[str, object]:
    """Return argparse values with secrets hidden before logging."""
    safe = vars(args).copy()
    if safe.get("hf_token"):
        safe["hf_token"] = "<redacted>"
    return safe


def _filter_unreadable_images(labels, image_paths, sample_records, log=print):
    """PIL이 열 수 없는 이미지(손상/truncated 다운로드)를 per-sample 리스트에서 제거.

    geometry metric은 유효한 임베딩을 전제로 하므로, 깨진 샘플을 검은 placeholder로
    대체하지 않고 아예 빼낸다. labels / image_paths / sample_records 세 리스트의
    정렬(1:1 대응)을 유지하기 위해 동일한 keep 인덱스로 함께 필터링한다.

    header만 확인(verify)하므로 빠르다. body가 truncated 된 파일은 통과할 수 있으나,
    extract_embeddings._PathImageDataset 가 ImageFile.LOAD_TRUNCATED_IMAGES=True 로
    디코딩하므로 인코딩 단계에서 worker가 죽지 않는다.
    """
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # truncated JPEG도 디코딩 허용
    keep = []
    bad_examples = []
    for i, p in enumerate(image_paths):
        try:
            with Image.open(p) as im:
                im.verify()
            keep.append(i)
        except Exception as e:
            if len(bad_examples) < 5:
                bad_examples.append(f"{p} ({type(e).__name__})")
    n_bad = len(image_paths) - len(keep)
    if n_bad == 0:
        return labels, image_paths, sample_records, 0
    for ex in bad_examples:
        log(f"    unreadable: {ex}")
    new_labels = np.array([labels[i] for i in keep])
    new_paths = [image_paths[i] for i in keep]
    new_records = [sample_records[i] for i in keep]
    return new_labels, new_paths, new_records, n_bad


def _subsample_per_class(labels, image_paths, sample_records, max_per_class, seed):
    """클래스당 최대 N장만 남기는 서브샘플링 (학부생 빠른 테스트용)."""
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = {}
    for i, c in enumerate(labels):
        by_class.setdefault(int(c), []).append(i)
    keep_idx = []
    for c, idx_list in by_class.items():
        if len(idx_list) > max_per_class:
            chosen = rng.choice(idx_list, size=max_per_class, replace=False)
            keep_idx.extend(sorted(chosen.tolist()))
        else:
            keep_idx.extend(idx_list)
    keep_idx = sorted(keep_idx)
    new_labels = np.array([labels[i] for i in keep_idx])
    new_paths = [image_paths[i] for i in keep_idx] if image_paths else None
    new_records = [sample_records[i] for i in keep_idx]
    return new_labels, new_paths, new_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openclip-vitb32",
                        choices=["openclip-vitb32", "openclip-vitl14", "bioclip", "bioclip2"])
    parser.add_argument("--toy", action="store_true",
                        help="Use toy synthetic dataset (8 species).")
    parser.add_argument("--mock", action="store_true",
                        help="Skip model loading entirely; use deterministic hashed mock embeddings.")
    parser.add_argument("--allow_mock_fallback", action="store_true",
                        help="Allow fallback to mock embeddings if real model loading fails.")
    parser.add_argument("--csv", default=None,
                        help="Path to metadata CSV for real-data run.")
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--visualize", action="store_true",
                        help="UMAP/t-SNE 시각화를 PNG로 저장.")
    parser.add_argument("--vis_method", default="umap", choices=["umap", "tsne"])
    parser.add_argument("--device", default=None,
                        help="'cuda' / 'mps' / 'cpu'. 미지정 시 CUDA -> MPS -> CPU 순서로 자동 감지.")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    parser.add_argument("--batch_size", type=int, default=64,
                        help="이미지 인코딩 배치 크기. GPU 메모리에 맞춰 조정.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker 수. Windows는 0 권장.")
    parser.add_argument("--amp", action="store_true",
                        help="CUDA에서 float16 mixed precision으로 인코딩 (보통 1.5~2배 빠름).")
    parser.add_argument("--cache_emb", default=None,
                        help="이미지 임베딩 .npz 캐시 경로. 존재하면 로드, 없으면 인코딩 후 저장.")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="클래스당 이미지 수 상한 (빠른 테스트용). 예: 10 → 200 클래스 × 10 = 2000장.")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token. 쉘 히스토리에 남을 수 있으므로 --hf_token_file 또는 --hf_token_prompt 권장.")
    parser.add_argument("--hf_token_file", default=None,
                        help="HuggingFace token을 담은 파일 경로. 예: ~/.cache/huggingface/token")
    parser.add_argument("--hf_token_prompt", action="store_true",
                        help="실행 시 HuggingFace token을 숨김 입력으로 받음.")
    parser.add_argument("--env_file", default=None,
                        help="로드할 .env 파일 경로. 미지정 시 현재 디렉토리/스크립트 디렉토리의 상위에서 .env 자동 탐색.")
    parser.add_argument("--no_env_file", action="store_true",
                        help=".env 자동 로드를 비활성화.")
    args = parser.parse_args()
    env_file_loaded = _load_env_file(args.env_file, disabled=args.no_env_file)
    hf_token_source = _configure_hf_token(args)

    set_global_seeds(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "run_log.txt")
    log_lines: List[str] = []

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_lines.append(line)

    args.device = _autodetect_device(args.device)
    log(f"args: {_redacted_args(args)}")
    result_files = ["exp1_geometry.json", "exp2_rank_levels.json", "exp3_counterfactuals.json"]
    if any(os.path.exists(os.path.join(args.out, name)) for name in result_files):
        log("WARN: output directory already contains results — files will be overwritten.")
    if env_file_loaded:
        log(f".env loaded from {env_file_loaded}")
    if hf_token_source:
        log(f"HF token configured from {hf_token_source} (value hidden).")

    # ---- 1) data ----
    if args.csv:
        spec = RealDatasetSpec(image_root=args.image_root or ".", metadata_csv=args.csv)
        records, labels, image_paths, tax_table = load_real_dataset(spec)
        sample_records = [records[i] for i in labels]
        log(f"Real dataset: {len(records)} species, {len(image_paths)} images")
        if args.max_per_class:
            labels, image_paths, sample_records = _subsample_per_class(
                labels, image_paths, sample_records, args.max_per_class, args.seed
            )
            log(f"  → subsample({args.max_per_class}/class): {len(labels)} images remain")
        if not args.mock:
            labels, image_paths, sample_records, n_bad = _filter_unreadable_images(
                labels, image_paths, sample_records, log=log
            )
            if n_bad:
                log(f"  → dropped {n_bad} unreadable images; {len(image_paths)} remain")
    else:
        records, labels, sample_records, tax_table = get_toy_dataset(samples_per_species=6,
                                                                     seed=args.seed)
        image_paths = None
        log(f"Toy dataset: {len(records)} species, {len(labels)} samples")

    # ---- 2) load model (or mock) + 이미지 인코딩 (캐시 지원) ----
    encoder_state = None
    img_emb: np.ndarray

    cache_hit = False
    if args.cache_emb and os.path.isfile(args.cache_emb):
        try:
            blob = np.load(args.cache_emb, allow_pickle=False)
            img_emb_cached = blob["img_emb"]
            labels_cached = blob["labels"]
            if img_emb_cached.shape[0] == len(labels) and np.array_equal(labels_cached, labels):
                img_emb = img_emb_cached
                cache_hit = True
                log(f"임베딩 캐시 적중: {args.cache_emb}  shape={img_emb.shape}")
            else:
                log(f"  캐시 크기 불일치 (got {img_emb_cached.shape[0]} vs {len(labels)}) → 재인코딩")
        except Exception as e:
            log(f"  캐시 로드 실패 ({e}); 재인코딩")

    if args.mock:
        log("MOCK mode: skipping model load, using synthetic image embeddings.")
        img_emb = mock_image_embeddings(labels, n_class=len(records), dim=128,
                                        intra_noise=0.45, seed=args.seed)
    elif not cache_hit:
        try:
            log(f"Loading model: {args.model} (device={args.device})")
            model, preprocess, tokenizer, info = load_model(args.model, hf_token=args.hf_token)
            encoder_state = {"model": model, "preprocess": preprocess, "tokenizer": tokenizer,
                              "info": info, "device": args.device}
            log(f"Model loaded: {info}")
            if image_paths is None:
                import torch
                imgs = get_toy_images(labels, image_size=info.image_size, seed=args.seed)
                img_emb = encode_images(model, preprocess, imgs, device=args.device,
                                         batch_size=args.batch_size, use_amp=args.amp)
                log(f"Encoded {len(imgs)} toy images -> embeddings shape {img_emb.shape}")
            else:
                img_emb = encode_images(model, preprocess, image_paths,
                                         device=args.device, batch_size=args.batch_size,
                                         num_workers=args.num_workers, use_amp=args.amp)
                log(f"Encoded {len(image_paths)} real images -> {img_emb.shape}")
            if args.cache_emb:
                np.savez_compressed(args.cache_emb, img_emb=img_emb, labels=labels)
                log(f"  임베딩 캐시 저장: {args.cache_emb}")
        except Exception as e:
            warning = f"WARN: real model load failed ({type(e).__name__}: {e})"
            print(f"\033[1m{warning}\033[0m", file=sys.stderr)
            if not args.allow_mock_fallback:
                sys.exit(1)
            log(f"{warning}; falling back to MOCK.")
            args.mock = True
            img_emb = mock_image_embeddings(labels, n_class=len(records), dim=128,
                                            intra_noise=0.45, seed=args.seed)
    else:
        # 캐시 적중 → 텍스트 인코딩을 위해 모델만 로드
        try:
            log(f"캐시 적중 후 텍스트 인코더 로드: {args.model}")
            model, preprocess, tokenizer, info = load_model(args.model, hf_token=args.hf_token)
            encoder_state = {"model": model, "preprocess": preprocess, "tokenizer": tokenizer,
                              "info": info, "device": args.device}
        except Exception as e:
            log(f"  텍스트 인코더 로드 실패 ({e}) → mock 텍스트로 진행")

    # ---- 3) Experiment 1: RQ1 ----
    log("== Experiment 1 (RQ1): hierarchical vs flat ==")
    exp1 = experiment_1(records, sample_records, labels, img_emb, encoder_state,
                         mock=args.mock, n_seeds=args.n_seeds, seed=args.seed)
    exp1_vis_emb = exp1.pop("_vis_embeddings", {})
    with open(os.path.join(args.out, "exp1_geometry.json"), "w") as f:
        json.dump(exp1, f, indent=2)
    log(f"exp1 success: {exp1['success_criteria']}")
    if args.visualize:
        for cond, Z_vis in exp1_vis_emb.items():
            plot_embeddings(Z_vis, labels, title=f"Exp1_{cond}", method=args.vis_method,
                            save_path=os.path.join(args.out, f"vis_exp1_{cond}.png"), seed=args.seed)
        if exp1_vis_emb:
            plot_comparison_grid(
                exp1_vis_emb, labels,
                title=f"Exp1 C0 vs C1 ({args.model})",
                method=args.vis_method,
                save_path=os.path.join(args.out, "vis_exp1_compare.png"),
                seed=args.seed,
            )

    # ---- 4) Experiment 2: RQ2 ----
    log("== Experiment 2 (RQ2): per-rank silhouette + latent taxonomy probe ==")
    exp2 = experiment_2(records, sample_records, labels, img_emb, tax_table,
                         encoder_state, mock=args.mock, seed=args.seed)
    exp2_vis_emb = exp2.pop("_vis_embeddings", {})
    with open(os.path.join(args.out, "exp2_rank_levels.json"), "w") as f:
        json.dump(exp2, f, indent=2)
    log(f"exp2 latent probe: {exp2.get('latent_taxonomy_probe')}")
    if args.visualize:
        for rank_name, payload in exp2_vis_emb.items():
            plot_comparison_grid(
                payload["embeddings"], payload["y"],
                title=f"Exp2 {rank_name} ({args.model})",
                method=args.vis_method,
                save_path=os.path.join(args.out, f"vis_exp2_{rank_name}.png"),
                seed=args.seed,
            )

    # ---- 5) Experiment 3: RQ3 counterfactual ablation ----
    log("== Experiment 3 (RQ3): counterfactual ablation C0..C4 ==")
    exp3 = experiment_3(records, sample_records, labels, img_emb, encoder_state,
                         mock=args.mock, n_seeds=args.n_seeds, seed=args.seed)
    exp3_vis_emb = exp3.pop("_vis_embeddings", {})
    with open(os.path.join(args.out, "exp3_counterfactuals.json"), "w") as f:
        json.dump(exp3, f, indent=2)
    log(f"exp3 semantic_organizer_supported: {exp3['semantic_organizer_supported']}")
    if args.visualize:
        for cond, Z_vis in exp3_vis_emb.items():
            plot_embeddings(Z_vis, labels, title=f"Exp3_{cond}", method=args.vis_method,
                            save_path=os.path.join(args.out, f"vis_exp3_{cond}.png"), seed=args.seed)
        if exp3_vis_emb:
            plot_comparison_grid(
                exp3_vis_emb, labels,
                title=f"Exp3 Ablation ({args.model})",
                method=args.vis_method,
                save_path=os.path.join(args.out, "vis_exp3_compare.png"),
                seed=args.seed,
            )

    # ---- 6) Zero-shot top-1 accuracy per condition (downstream recognition) ----
    log("== Zero-shot top-1 accuracy (per-class text prototypes) ==")
    zs_rng = np.random.default_rng(args.seed)
    zeroshot = {}
    for cond in ["C0", "C1", "C2", "C3", "C4"]:  # C5 has no text prompts
        acc = zero_shot_accuracy(records, img_emb, labels, encoder_state, cond, zs_rng)
        zeroshot[cond] = acc
    with open(os.path.join(args.out, "zeroshot_accuracy.json"), "w") as f:
        json.dump(zeroshot, f, indent=2)
    log(f"zero-shot top-1: {zeroshot}")

    # ---- 7) t-SNE visualization of image embeddings ----
    tsne_path = save_tsne(img_emb, sample_records,
                          os.path.join(args.out, "tsne_image_embeddings.png"),
                          seed=args.seed)
    log(f"t-SNE figure: {tsne_path if tsne_path else 'skipped (matplotlib/sklearn unavailable)'}")

    # ---- save log ----
    import subprocess
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = "unknown"
    log(f"git_hash: {git_hash}")

    import importlib
    for pkg in ["numpy", "torch", "open_clip", "sklearn"]:
        try:
            ver = importlib.import_module(pkg).__version__
        except Exception:
            ver = "n/a"
        log(f"  {pkg}: {ver}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"Done. Outputs in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
