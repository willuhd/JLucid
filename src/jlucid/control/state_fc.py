"""Per-subject, per-state functional connectivity (step-2 Phase B, D2-4).

FC_s is Pearson correlation computed only on frames assigned to LEiDA state
s, with a minimum frame count; global FC uses all frames (Paper-6 static
baseline).  Ledoit-Wolf shrinkage is provided as a sensitivity estimator.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from ..config import MIN_STATE_FRAMES


def pearson_fc(frames: np.ndarray) -> np.ndarray:
    """Pearson correlation across ROIs for a (T, N) time-course matrix."""
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 2:
        raise ValueError(f"expected (T, N), got {frames.shape}")
    t = frames.shape[0]
    if t < 2:
        raise ValueError("need >= 2 frames to estimate FC")
    x = frames - frames.mean(axis=0)
    sd = x.std(axis=0)
    if np.any(sd == 0):
        raise ValueError("constant ROI time series (zero variance)")
    x = x / sd
    return x.T @ x / (t - 1)


def ledoit_fc(frames: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance (sensitivity estimator, D2-4)."""
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 2 or frames.shape[0] < 2:
        raise ValueError("need (T, N) with >= 2 frames")
    return LedoitWolf().fit(frames).covariance_


def compute_subject_state_fc(
    tc: np.ndarray,
    labels: np.ndarray,
    n_states: int,
    mask: np.ndarray | None = None,
    min_frames: int = MIN_STATE_FRAMES,
) -> tuple[dict[int, tuple[np.ndarray, int]], tuple[np.ndarray, int]]:
    """Per-state FC for one subject.

    Returns (state_fc, global_fc) where each is (fc_matrix, n_frames) with
    fc_matrix None when n_frames < min_frames (missing per D2-12).
    Censored frames (mask False) never enter FC computation (D2-5).
    """
    tc = np.asarray(tc, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if mask is None:
        mask = np.ones(len(tc), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(tc),):
        raise ValueError("mask length must match tc length")
    keep = mask & (labels >= 0)
    state_fc: dict[int, tuple[np.ndarray, int]] = {}
    for s in range(n_states):
        idx = np.flatnonzero(keep & (labels == s))
        n = len(idx)
        if n < max(min_frames, 2):
            state_fc[s] = (None, n)
            continue
        state_fc[s] = (pearson_fc(tc[idx]), n)
    g_idx = np.flatnonzero(mask)
    g_n = len(g_idx)
    global_fc = (pearson_fc(tc[g_idx]), g_n) if g_n >= 2 else (None, g_n)
    return state_fc, global_fc
