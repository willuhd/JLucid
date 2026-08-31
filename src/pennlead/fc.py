"""Functional connectivity — Pearson FC + normalizations.
Isolated copy for src-pennlead (identical logic to src-controllability/controllability/fc.py)
so parallel ADHD200 grind does not share file handles.
Paper 6 §2.4: Pearson correlation between regional time series.
"""
from __future__ import annotations
import numpy as np

def compute_fc(timeseries: np.ndarray, zscore_timeseries: bool = False) -> np.ndarray:
    """Pearson FC from (T x N) time series."""
    ts = np.asarray(timeseries, dtype=np.float64)
    if zscore_timeseries:
        ts = (ts - ts.mean(axis=0, keepdims=True)) / (ts.std(axis=0, keepdims=True) + 1e-12)
    fc = np.corrcoef(ts.T)
    fc = np.clip(fc, -1.0, 1.0)
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 1.0)
    fc = np.nan_to_num(fc, nan=0.0)
    return fc

def normalize_fc(fc: np.ndarray, method: str = "none") -> np.ndarray:
    fc = np.asarray(fc, dtype=np.float64)
    if method == "none":
        return fc
    if method == "fisher":
        clipped = np.clip(fc, -0.999, 0.999)
        z = np.arctanh(clipped)
        np.fill_diagonal(z, 0.0)
        return z
    if method == "global_zscore":
        mask = ~np.eye(fc.shape[0], dtype=bool)
        m = fc[mask].mean()
        s = fc[mask].std() + 1e-12
        out = (fc - m) / s
        np.fill_diagonal(out, 0.0)
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
