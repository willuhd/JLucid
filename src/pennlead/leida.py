"""LEiDA — Leading Eigenvector Dynamics Analysis, isolated for src-pennlead.
Hilbert -> theta -> dPC = cos(theta_n - theta_p) -> V1(t) -> k-means.
Same math as src-leida but handles PennLEAD N=514, T=156, and ESWAN regression.
No import from src-leida at import time (parallel-safe).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import hilbert
from sklearn.cluster import KMeans

def compute_leida_V1(timeseries_list: list[np.ndarray]):
    """Compute V1(t) for each subject.
    Args:
        timeseries_list: list of (T,N) arrays
    Returns:
        subj_V1_list: list of (T,N) V1 per subject
        all_V1: (n_subjects*T, N) concatenated
    """
    subj_V1_list = []
    for ts in timeseries_list:
        T, N = ts.shape
        analytic = hilbert(ts, axis=0)  # T x N complex
        phase = np.angle(analytic)  # T x N
        V1s = []
        for t in range(T):
            theta = phase[t]  # N
            d = np.cos(theta[:, None] - theta[None, :])  # N x N
            vals, vecs = np.linalg.eigh(d)
            v1 = vecs[:, -1]  # leading
            if np.sum(v1 > 0) > 0.5 * N:
                v1 = -v1
            V1s.append(v1)
        V1s = np.array(V1s)  # T x N
        subj_V1_list.append(V1s)
    all_V1 = np.concatenate(subj_V1_list, axis=0) if subj_V1_list else np.empty((0,0))
    return subj_V1_list, all_V1

def kmeans_leida(all_V1: np.ndarray, k: int = 5, n_init: int = 20, seed: int = 42):
    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = kmeans.fit_predict(all_V1)
    centroids = kmeans.cluster_centers_  # k x N
    return labels, centroids, kmeans

def leida_metrics(subj_V1_list, cluster_labels_all, centroids, k: int = 5):
    """Occurrence, lifetime, TPM per subject."""
    n_subj = len(subj_V1_list)
    Ts = [v.shape[0] for v in subj_V1_list]
    # split labels back
    subj_clusters = []
    idx = 0
    for T in Ts:
        subj_clusters.append(cluster_labels_all[idx:idx+T])
        idx += T
    occ = np.zeros((n_subj, k))
    lifetimes = np.zeros((n_subj, k))
    tpms = np.zeros((n_subj, k, k))
    for s, cl in enumerate(subj_clusters):
        T = len(cl)
        for state in range(k):
            occ[s, state] = np.mean(cl == state)
            # runs
            runs = []
            cur = 0
            for t in range(T):
                if cl[t] == state:
                    cur += 1
                else:
                    if cur > 0:
                        runs.append(cur)
                        cur = 0
            if cur > 0:
                runs.append(cur)
            lifetimes[s, state] = np.mean(runs) if runs else 0
        for t in range(T-1):
            tpms[s, cl[t], cl[t+1]] += 1
        for i in range(k):
            rs = tpms[s, i].sum()
            if rs > 0:
                tpms[s, i] /= rs
    return occ, lifetimes, tpms, subj_clusters
