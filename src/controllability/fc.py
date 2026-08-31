"""Functional connectivity utilities — Pearson FC + normalizations.
Strictly paper 6 §2.4: Pearson correlation between regional time series.
Advisor question: normalization / z-score handling is explored via variants.
"""
from __future__ import annotations
import numpy as np

def compute_fc(timeseries: np.ndarray, zscore_timeseries: bool = False) -> np.ndarray:
    """Compute Pearson FC from (T x N) time series.
    Args:
        timeseries: array shape (T, N) — T time points, N ROIs
        zscore_timeseries: if True, z-score each ROI time series before correlation (time-axis).
    Returns:
        FC symmetric N x N, diagonal 1, values in [-1,1]
    """
    ts = np.asarray(timeseries, dtype=np.float64)
    if zscore_timeseries:
        ts = (ts - ts.mean(axis=0, keepdims=True)) / (ts.std(axis=0, keepdims=True) + 1e-12)
    # Pearson correlation — column-wise
    # Use np.corrcoef on transposed (N x T) -> (N x N)
    # Handle T < N (paper 6: 142 vols, 100 ROIs) robustly.
    fc = np.corrcoef(ts.T)
    # corrcoef returns (N,N); clip numerical drift
    fc = np.clip(fc, -1.0, 1.0)
    # Ensure symmetry exactly
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 1.0)
    # Replace NaNs (constant time series) with 0 off-diagonal
    fc = np.nan_to_num(fc, nan=0.0)
    return fc

def normalize_fc(fc: np.ndarray, method: str = "none") -> np.ndarray:
    """Advisor: '有没有做归一化，或者z-score' — explore FC-level norms.
    - none: return as-is
    - fisher: atanh clip to avoid inf (Fisher r-to-z)
    - global_zscore: (FC - mean(offdiag))/std(offdiag)
    - row_zscore: z-score each row (symmetrize after)
    """
    fc = np.asarray(fc, dtype=np.float64)
    if method == "none":
        return fc
    if method == "fisher":
        clipped = np.clip(fc, -0.999, 0.999)
        z = np.arctanh(clipped)
        # keep diagonal interpretable
        np.fill_diagonal(z, 0.0)
        return z
    if method == "global_zscore":
        # off-diagonal only
        mask = ~np.eye(fc.shape[0], dtype=bool)
        m = fc[mask].mean()
        s = fc[mask].std() + 1e-12
        out = (fc - m) / s
        np.fill_diagonal(out, 0.0)
        # resymmetrize
        out = (out + out.T)/2
        return out
    if method == "row_zscore":
        m = fc.mean(axis=1, keepdims=True)
        s = fc.std(axis=1, keepdims=True) + 1e-12
        out = (fc - m) / s
        out = (out + out.T)/2
        np.fill_diagonal(out, 0.0)
        return out
    raise ValueError(f"Unknown fc_norm method: {method}")

def ensure_symmetric(fc: np.ndarray) -> np.ndarray:
    return (fc + fc.T) / 2.0
