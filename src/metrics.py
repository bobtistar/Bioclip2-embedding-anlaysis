"""Embedding geometry & taxonomic metrics for BioCLIP2 hierarchical prompt analysis.

All functions assume L2-normalized embeddings unless otherwise stated.

Metrics
-------
- intra_class_variance     : mean within-class pairwise cosine distance (lower = better)
- inter_class_margin       : nearest-other-centroid distance / within-class std (higher = better)
- silhouette_cosine        : sklearn silhouette with cosine distance
- uniformity               : Wang & Isola 2020
- rankme                   : Garrido et al. 2023 (effective rank)
- knn_purity_at_k          : taxonomic kNN purity

Statistical helpers
-------------------
- paired_permutation_test
- bootstrap_ci
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

# Soft import sklearn so the module can be imported even if sklearn missing.
try:
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    _SKLEARN_OK = True
except Exception as e:  # pragma: no cover
    _SKLEARN_OK = False
    _SKLEARN_ERR = e


def l2_normalize(Z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / (n + eps)


def intra_class_variance(Z: np.ndarray, y: np.ndarray) -> float:
    """Mean over classes of mean within-class pairwise cosine distance.

    Z: (N, D) L2-normalized embeddings
    y: (N,)   integer class labels
    """
    Z = l2_normalize(Z)
    classes = np.unique(y)
    vals: List[float] = []
    for c in classes:
        Zc = Z[y == c]
        if len(Zc) < 2:
            continue
        sim = Zc @ Zc.T  # cosine since Z normalized
        # take upper triangle (i<j)
        iu = np.triu_indices(len(Zc), k=1)
        dist = 1.0 - sim[iu]
        vals.append(float(dist.mean()))
    return float(np.mean(vals)) if vals else float("nan")


def inter_class_margin(Z: np.ndarray, y: np.ndarray) -> float:
    """(nearest-other-centroid cosine distance) / (within-class std), mean over classes."""
    Z = l2_normalize(Z)
    classes = np.unique(y)
    centroids = np.stack([Z[y == c].mean(axis=0) for c in classes])
    centroids = l2_normalize(centroids)
    cent_sim = centroids @ centroids.T
    cent_dist = 1.0 - cent_sim
    np.fill_diagonal(cent_dist, np.inf)

    ratios: List[float] = []
    for ci, c in enumerate(classes):
        Zc = Z[y == c]
        if len(Zc) < 2:
            continue
        # std of within-class cosine distance to centroid
        sim_to_mean = Zc @ centroids[ci]
        dist_to_mean = 1.0 - sim_to_mean
        sigma = float(dist_to_mean.std())
        if sigma < 1e-8:
            sigma = 1e-8
        nearest = float(cent_dist[ci].min())
        ratios.append(nearest / sigma)
    return float(np.mean(ratios)) if ratios else float("nan")


def silhouette_cosine(
    Z: np.ndarray,
    y: np.ndarray,
    precomputed_distance: Optional[np.ndarray] = None,
) -> float:
    """Cosine silhouette. When the caller computes silhouette many times with
    the *same* Z but different labels (e.g. random-taxonomy permutation probe),
    pass a precomputed (N, N) cosine-distance matrix to avoid re-doing the
    pairwise distance each call.

    `precomputed_distance` should be `1 - Z_norm @ Z_norm.T` with diagonal=0.
    """
    if not _SKLEARN_OK:
        raise RuntimeError(f"sklearn not available: {_SKLEARN_ERR}")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return float("nan")
    if precomputed_distance is not None:
        return float(silhouette_score(precomputed_distance, y, metric="precomputed"))
    Z = l2_normalize(Z)
    return float(silhouette_score(Z, y, metric="cosine"))


def cosine_distance_matrix(Z: np.ndarray) -> np.ndarray:
    """Pairwise cosine *distance* matrix for L2-normalizable inputs.

    Returns float32 (N, N). Memory: ~4·N²  bytes. For N≈12k this is ≈560MB —
    callers should clean up promptly. Diagonal is forced to 0 and the matrix
    is clamped to ≥ 0 to avoid sklearn precomputed-metric complaints from
    floating-point noise.
    """
    Zn = l2_normalize(Z).astype(np.float32, copy=False)
    sim = Zn @ Zn.T
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    np.maximum(D, 0.0, out=D)
    return D


def alignment(Z_pos_a: np.ndarray, Z_pos_b: np.ndarray, alpha: float = 2.0) -> float:
    """Wang & Isola alignment: E_{positives} || f(x) - f(x+) ||^alpha."""
    Z_pos_a = l2_normalize(Z_pos_a)
    Z_pos_b = l2_normalize(Z_pos_b)
    d = np.linalg.norm(Z_pos_a - Z_pos_b, axis=1)
    return float((d ** alpha).mean())


def uniformity(Z: np.ndarray, t: float = 2.0) -> float:
    """Wang & Isola uniformity: log E_{x,y} exp(-t || f(x) - f(y) ||^2)."""
    Z = l2_normalize(Z)
    n = len(Z)
    if n < 2:
        return float("nan")
    # pairwise squared distance via expansion
    # |x-y|^2 = 2 - 2 x.y  (for normalized)
    sim = Z @ Z.T
    d2 = 2.0 - 2.0 * sim
    iu = np.triu_indices(n, k=1)
    return float(np.log(np.exp(-t * d2[iu]).mean()))


def rankme(Z: np.ndarray, eps: float = 1e-7) -> float:
    """RankMe effective rank (Garrido et al. 2023)."""
    if Z.shape[0] < 2:
        return float("nan")
    # Use np.linalg.svd; for tall matrices use full_matrices=False
    try:
        s = np.linalg.svd(Z, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    s_sum = s.sum() + eps
    p = s / s_sum + eps
    # exp of Shannon entropy of normalized singular values
    return float(np.exp(-(p * np.log(p)).sum()))


def knn_purity_at_k(Z: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    if not _SKLEARN_OK:
        raise RuntimeError(f"sklearn not available: {_SKLEARN_ERR}")
    Z = l2_normalize(Z)
    n = len(Z)
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine").fit(Z)
    _, idx = nn.kneighbors(Z)
    # exclude self (column 0)
    nbr = idx[:, 1:]
    purity = (y[nbr] == y[:, None]).mean(axis=1)
    return float(purity.mean())


def plot_embeddings(
    Z: np.ndarray,
    y: np.ndarray,
    title: str = "",
    method: str = "umap",
    save_path: Optional[str] = None,
    seed: int = 42,
    label_names: Optional[Dict[int, str]] = None,
) -> None:
    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise RuntimeError("umap-learn not installed") from e
        reducer = umap.UMAP(n_components=2, random_state=seed)
        coords = reducer.fit_transform(Z)
    elif method == "tsne":
        if not _SKLEARN_OK:
            raise RuntimeError(f"sklearn not available: {_SKLEARN_ERR}")
        from sklearn.manifold import TSNE
        perplexity = min(30, len(Z) // 4)
        if perplexity < 1:
            perplexity = 1
        coords = TSNE(n_components=2, random_state=seed, perplexity=perplexity).fit_transform(Z)
    else:
        raise ValueError(method)

    import matplotlib.pyplot as plt

    classes = np.unique(y)
    n_cls = len(classes)
    w = 7 + max(0, (n_cls - 10) * 0.15)
    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(w, 5))
    for i, c in enumerate(classes):
        mask = y == c
        label = label_names.get(int(c), str(c)) if label_names else str(c)
        plt.scatter(coords[mask, 0], coords[mask, 1], s=18, color=cmap(i % 20), label=label)
    plt.title(title)
    if n_cls > 20:
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    else:
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_comparison_grid(
    embeddings_by_cond: Dict[str, np.ndarray],
    y: np.ndarray,
    title: str = "",
    method: str = "umap",
    save_path: Optional[str] = None,
    seed: int = 42,
    label_names: Optional[Dict[int, str]] = None,
) -> None:
    """여러 조건의 임베딩을 같은 그림에 서브플롯으로 비교."""
    import matplotlib.pyplot as plt

    conditions = list(embeddings_by_cond.keys())
    n = len(conditions)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.asarray(axes).reshape(-1)
    classes = np.unique(y)
    cmap = plt.get_cmap("tab20")

    for ax_idx, cond in enumerate(conditions):
        Z = embeddings_by_cond[cond]
        if method == "umap":
            try:
                import umap
            except ImportError as e:
                raise RuntimeError("umap-learn not installed") from e
            coords = umap.UMAP(n_components=2, random_state=seed).fit_transform(Z)
        elif method == "tsne":
            if not _SKLEARN_OK:
                raise RuntimeError(f"sklearn not available: {_SKLEARN_ERR}")
            from sklearn.manifold import TSNE
            perplexity = min(30, len(Z) // 4)
            if perplexity < 1:
                perplexity = 1
            coords = TSNE(n_components=2, random_state=seed, perplexity=perplexity).fit_transform(Z)
        else:
            raise ValueError(method)

        ax = axes[ax_idx]
        for i, c in enumerate(classes):
            mask = y == c
            label = label_names.get(int(c), str(c)) if label_names else str(c)
            ax.scatter(coords[mask, 0], coords[mask, 1], s=14, color=cmap(i % 20), label=label)
        ax.set_title(cond)
        if ax_idx == 0 and len(classes) <= 15:
            ax.legend()

    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def paired_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Two-sided paired permutation test on mean difference (a - b).

    Returns dict: {mean_diff, p_value, ci_lo, ci_hi}. Vectorized — draws all
    permutations in one shot. Note: because the RNG is consumed in one large
    batch instead of n_perm small batches, numeric p-values will differ from
    the previous implementation at the 1/n_perm precision level, but the
    underlying distribution and statistical conclusion are unchanged.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    obs = diff.mean()
    n = len(diff)

    # permutation: random sign-flip of paired differences
    signs = rng.choice([-1, 1], size=(n_perm, n))
    sim_means = (signs * diff).mean(axis=1)
    count = int((np.abs(sim_means) >= abs(obs)).sum())
    p = (count + 1) / (n_perm + 1)

    # bootstrap CI on the mean difference
    idx = rng.integers(0, n, size=(n_perm, n))
    boots = diff[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": float(obs), "p_value": float(p),
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}


def bootstrap_ci(
    values: np.ndarray,
    statistic: callable = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Generic bootstrap CI. Returns (point_estimate, ci_lo, ci_hi).

    For axis-aware statistics (np.mean, np.median, np.std, ...) the bootstrap
    samples are drawn and reduced in a single vectorized call. Other callables
    fall back to a per-resample Python loop.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    point = float(statistic(values))
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    resamples = values[idx]
    try:
        boots = np.asarray(statistic(resamples, axis=1), dtype=float)
        if boots.shape != (n_boot,):
            raise TypeError
    except TypeError:
        boots = np.fromiter(
            (statistic(resamples[i]) for i in range(n_boot)),
            dtype=float, count=n_boot,
        )
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a) - np.asarray(b)
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    return float(diff.mean() / sd) if sd > 1e-8 else 0.0


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # toy: 4 classes, 8 samples each, 16-dim
    Z = []
    y = []
    for c in range(4):
        mu = rng.normal(0, 1, size=16)
        for _ in range(8):
            Z.append(mu + 0.3 * rng.normal(0, 1, size=16))
            y.append(c)
    Z = np.array(Z); y = np.array(y)
    print("intra_var :", intra_class_variance(Z, y))
    print("inter_marg:", inter_class_margin(Z, y))
    print("silhouette:", silhouette_cosine(Z, y))
    print("rankme    :", rankme(Z))
    print("uniformity:", uniformity(Z))
    print("knn_pur@3 :", knn_purity_at_k(Z, y, k=3))
