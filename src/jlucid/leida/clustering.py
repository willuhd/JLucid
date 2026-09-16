"""Pooled k-means and k-selection for LEiDA V1 time series (Paper 7, D6-D9)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ..config import N_INIT, SEED


def pool_v1(subject_v1: dict[str, np.ndarray],
            keep_ids: list[str] | None = None) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Concatenate per-subject V1 arrays into (M, N) plus subject row ranges."""
    ids = [s for s in subject_v1 if (keep_ids is None or s in keep_ids)]
    X = np.vstack([subject_v1[s] for s in ids])
    index = {}
    start = 0
    for s in ids:
        n = len(subject_v1[s])
        index[s] = (start, start + n)
        start += n
    return X, index


@dataclass
class ClusterResult:
    k: int
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    silhouette: float | None
    dunn: float | None
    stability: float | None
    subsample_idx: np.ndarray | None = field(default=None, repr=False)


def dunn_index_centroid(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Centroid-based Dunn index (valid for large M).

    inter = min over cluster pairs of the cosine distance between centroids;
    intra = max over clusters of the mean cosine distance of members to their
    centroid.  Dunn = inter / intra, higher is better (Paper 7 uses Dunn for
    k selection; this centroid formulation is the tractable variant for
    ~150k x 90 pooled data).
    """
    k = centroids.shape[0]
    if k < 2:
        return float("nan")
    inter = float(np.min(squareform(pdist(centroids, metric="cosine")) + np.eye(k) * 2))
    intra = 0.0
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            continue
        d = np.mean(
            1.0 - np.abs(centroids[c] @ X[mask].T) / (
                np.linalg.norm(centroids[c]) * np.linalg.norm(X[mask], axis=1)
            )
        )
        intra = max(intra, float(d))
    if intra == 0:
        return float("nan")
    return inter / intra


def _subsample(X: np.ndarray, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= size:
        return X, np.arange(len(X))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=size, replace=False)
    return X[idx], idx


def cluster_k(X: np.ndarray, k: int, n_init: int = N_INIT, seed: int = SEED,
              eval_subsample: int = 10_000) -> ClusterResult:
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    km.fit(X)
    labels = km.labels_
    centroids = km.cluster_centers_
    # normalize centroids to unit norm (V1 rows are unit-norm by construction)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-12)

    sil: float | None = None
    dunn: float | None = None
    stab: float | None = None
    sub_idx = None
    if eval_subsample and len(X) > 0:
        Xs, sub_idx = _subsample(X, min(eval_subsample, len(X)), seed)
        lab_s = labels[sub_idx]
        if len(np.unique(lab_s)) > 1:
            sil = float(silhouette_score(Xs, lab_s, metric="cosine"))
        dunn = dunn_index_centroid(Xs, lab_s, centroids)
    return ClusterResult(k=k, labels=labels, centroids=centroids,
                         inertia=float(km.inertia_), silhouette=sil,
                         dunn=dunn, stability=stab, subsample_idx=sub_idx)


def cluster_stability(X: np.ndarray, k: int, n_runs: int, seed: int,
                      subsample: int = 10_000) -> float:
    """Mean adjusted Rand index across re-clusterings on a fixed subsample."""
    Xs, _ = _subsample(X, min(subsample, len(X)), seed)
    ari = []
    for r in range(n_runs):
        km = KMeans(n_clusters=k, n_init=5, random_state=seed + r + 1)
        km.fit(Xs)
        ari.append(km.labels_)
    mean_ari = np.mean(
        [adjusted_rand_score(a, b) for i, a in enumerate(ari) for b in ari[i + 1 :]]
    )
    return float(mean_ari)


def kmeans_grid(X: np.ndarray, k_range=range(3, 21), n_init: int = N_INIT,
                seed: int = SEED, stability_runs: int = 10,
                eval_subsample: int = 10_000) -> dict[int, ClusterResult]:
    results = {}
    for k in k_range:
        res = cluster_k(X, k, n_init=n_init, seed=seed, eval_subsample=eval_subsample)
        res.stability = cluster_stability(X, k, n_runs=stability_runs, seed=seed,
                                          subsample=eval_subsample)
        results[k] = res
    return results


def select_k(results: dict[int, ClusterResult], anchor: int = 5) -> tuple[int, pd.DataFrame]:
    """Choose k from the grid.

    Primary criterion: maximum Dunn index (Paper 7).  Ties (within 2%) are
    broken by silhouette, then stability.  The decision table is returned so
    the report can document the choice; `anchor` is used only as a fallback
    when the grid is degenerate.
    """
    rows = []
    for k, res in sorted(results.items()):
        rows.append(
            {
                "k": k,
                "inertia": res.inertia,
                "silhouette": res.silhouette,
                "dunn": res.dunn,
                "stability_ari": res.stability,
            }
        )
    table = pd.DataFrame(rows)
    valid = table.dropna(subset=["dunn"])
    if valid.empty:
        return anchor, table
    best_dunn = valid.dunn.max()
    candidates = valid[valid.dunn >= best_dunn * 0.98]
    candidates = candidates.sort_values(
        ["silhouette", "stability_ari"], ascending=False, na_position="last"
    )
    return int(candidates.iloc[0]["k"]), table


def assign_states(V1: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each V1 row to the nearest centroid by cosine distance."""
    from sklearn.metrics import pairwise_distances_argmin_min

    labels, _ = pairwise_distances_argmin_min(V1, centroids, metric="cosine")
    return labels.astype(int)
