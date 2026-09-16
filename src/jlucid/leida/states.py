"""Per-subject state dynamics: occurrence, lifetime, TPM, order parameter."""

from __future__ import annotations

import numpy as np


def occurrence_probability(labels: np.ndarray, k: int) -> np.ndarray:
    """T_i / T for each state (censored frames excluded from T)."""
    valid = labels >= 0
    T = valid.sum()
    if T == 0:
        return np.zeros(k)
    out = np.zeros(k)
    for s in range(k):
        out[s] = np.sum(labels[valid] == s) / T
    return out


def lifetime(labels: np.ndarray, k: int) -> np.ndarray:
    """Mean run length per state; -1 (censored) breaks runs."""
    out = np.full(k, np.nan)
    for s in range(k):
        runs = []
        current = 0
        for lab in labels:
            if lab == s:
                current += 1
            elif lab != s:
                if current > 0:
                    runs.append(current)
                current = 0
        if current > 0:
            runs.append(current)
        if runs:
            out[s] = np.mean(runs)
    return out


def tpm_counts(labels: np.ndarray, k: int) -> np.ndarray:
    """Raw transition counts over consecutive non-censored frame pairs."""
    counts = np.zeros((k, k), dtype=float)
    prev = None
    for lab in labels:
        if lab < 0:
            prev = None
            continue
        if prev is not None:
            counts[prev, int(lab)] += 1
        prev = int(lab)
    return counts


def tpm(labels: np.ndarray, k: int) -> np.ndarray:
    """Transition probability matrix P(S_j(t+1) | S_i(t)) over observed
    consecutive non-censored frame pairs.

    Rows with no outgoing pair are 0 (legacy). Prefer :func:`tpm_export`
    for group statistics (0 vs NaN).
    """
    counts = tpm_counts(labels, k)
    row_sum = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts, np.maximum(row_sum, 1.0),
        out=np.zeros_like(counts), where=row_sum > 0,
    )


def tpm_export(labels: np.ndarray, k: int) -> np.ndarray:
    """Full k x k TPM for export / group tests.

    ``P(j | i)`` is 0 when state i has at least one outgoing pair but never
    went to j. It is NaN when i has no outgoing pair (never visited, or
    only isolated / last-frame visits after scrubbing).
    """
    counts = tpm_counts(labels, k)
    row_sum = counts.sum(axis=1)
    out = np.full((k, k), np.nan)
    for i in range(k):
        if row_sum[i] > 0:
            out[i] = counts[i] / row_sum[i]
    return out


def state_metrics_subject(labels: np.ndarray, phase: np.ndarray, k: int) -> dict:
    """Full per-subject metric bundle (occurrence, lifetime, TPM, order param).

    The order parameter is averaged over valid (labels >= 0) frames only, so
    censored frames do not bias it.
    """
    occ = occurrence_probability(labels, k)
    life = lifetime(labels, k)
    tp = tpm(labels, k)
    op = None
    if phase is not None:
        valid = labels >= 0
        if valid.sum() > 0:
            op = np.abs(np.mean(np.exp(1j * phase[valid]), axis=1))
    return {
        "occurrence": occ,
        "lifetime": life,
        "tpm": tp,
        "order_param_mean": float(op.mean()) if op is not None else np.nan,
        "n_frames": int(np.sum(labels >= 0)),
        "n_states_visited": int(np.sum(occ > 0)),
    }
