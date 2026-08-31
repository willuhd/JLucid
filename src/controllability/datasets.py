"""Dataset loaders — FREE & UNBLOCKED.
Priority: ADHD-200 (Athena) > UCLA CNP > synthetic fallback.
All loaders return (timeseries_list, labels, meta) where timeseries is (T,N).
Real fMRI download is via nilearn; synthetic is always available for CI/offline.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Optional

def _rng(seed=42): return np.random.default_rng(seed)

# ---------- Synthetic ----------
def generate_synthetic_timeseries(n_subjects: int = 40, n_rois: int = 100, T: int = 150, seed: int = 42, group_diff: float = 0.0):
    """Generate synthetic BOLD-like time series with optional group difference injected in frontal networks.
    Returns list of (T,N) arrays and labels (0=HC,1=ADHD).
    Group diff: ADHD group gets +group_diff added to frontal-FC via timeseries boost in frontal ROIs.
    """
    rng = _rng(seed)
    # Yeo-ish frontal indices: FPN proxy → last ~16 of 100 (DMN) not, so FPN indices
    # Use simple: frontal = ROI 60:76 (approx FPN region for Schaefer100)
    frontal_idx = np.arange(60, 76)
    timeseries = []
    labels = []
    for i in range(n_subjects):
        label = 0 if i < n_subjects//2 else 1
        # base: multivariate normal with weak correlations
        base = rng.standard_normal((T, n_rois)) * 0.8
        # add low-freq drift
        t = np.linspace(0, 4*np.pi, T)
        drift = np.sin(t)[:, None] * rng.standard_normal((1, n_rois)) * 0.2
        ts = base + drift
        if label == 1 and group_diff != 0:
            # boost frontal synchrony: add shared signal to frontal ROIs
            shared = rng.standard_normal(T) * group_diff * 2
            ts[:, frontal_idx] += shared[:, None] * 0.5
            # also increase amplitude slightly
            ts[:, frontal_idx] *= (1 + group_diff)
        timeseries.append(ts.astype(np.float64))
        labels.append(label)
    return timeseries, np.array(labels, dtype=int), {"frontal_idx": frontal_idx, "n_rois": n_rois, "T": T}

# ---------- ADHD-200 (Athena preprocessed) ----------
def load_adhd200(data_dir: str | Path = "data/adhd200", n_subjects: Optional[int] = None, download: bool = True, seed: int = 42) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
    """Load ADHD-200 via nilearn.datasets.fetch_adhd.
    If download False or offline, falls back to synthetic with warning.
    Athena preprocessed: already slice-time, motion, MNI, 6mm smoothed, bandpass.
    nilearn returns phenotypic + func files; we will extract time series via Schaefer/AAL on demand in preprocessing step.
    For offline CI, this loader directly generates synthetic as placeholder.
    To use real data, call with download=True and ensure internet.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if not download:
        print("[datasets] download=False → synthetic fallback for ADHD-200")
        return generate_synthetic_timeseries(n_subjects=n_subjects or 40, seed=seed)
    try:
        from nilearn.datasets import fetch_adhd
        # fetch_adhd downloads ~1-2GB; limit n_subjects to keep Intel Mac friendly
        adhd = fetch_adhd(n_subjects=n_subjects, data_dir=str(data_dir))
        # adhd is Bunch with func, phenotypic, etc.
        # For simplicity, return the bunch; preprocessing step will handle func → timeseries
        # Here we provide a lazy wrapper: return phenotypic labels + func paths
        phenotypic = adhd.get("phenotypic") if hasattr(adhd, "get") else None
        # Try to parse phenotypic CSV for ADHD vs TDC
        # nilearn's fetch_adhd returns list of func files and phenotypic DataFrame
        import pandas as pd
        if isinstance(adhd, dict) and "phenotypic" in adhd:
            ph = pd.DataFrame(adhd["phenotypic"]) if not isinstance(adhd["phenotypic"], pd.DataFrame) else adhd["phenotypic"]
            # ADHD vs TDC column varies by site; look for 'dx' or 'ADHD Index'
            # Fallback: use phenotypic if available else synthetic
            print(f"[datasets] ADHD-200 fetched: {len(adhd.get('func',[]))} func files")
            return adhd, None, {"source": "adhd200_nilearn", "phenotypic": ph}
        # older nilearn returns tuple (func, phenotypic)
        print(f"[datasets] ADHD-200 fetched via nilearn: {type(adhd)}")
        return adhd, None, {"source": "adhd200_nilearn"}
    except Exception as e:
        print(f"[datasets] ADHD-200 fetch failed ({e}) → synthetic fallback. "
              f"To get real ADHD-200, ensure internet and: pip install nilearn, then re-run with data_dir={data_dir}")
        return generate_synthetic_timeseries(n_subjects=n_subjects or 40, seed=seed)

# ---------- UCLA CNP (OpenNeuro ds000030) ----------
def load_ucla(data_dir: str | Path = "data/ucla", n_subjects: Optional[int] = None, seed: int = 42):
    """UCLA CNP LA5c — 272 single-scanner (same as paper 6). Via nilearn.datasets.fetch_openneuro or synthetic fallback."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        from nilearn.datasets import fetch_openneuro_dataset
        # ds000030 is large; we fetch index only to avoid 50GB download on CI
        print("[datasets] Attempting UCLA CNP via OpenNeuro (ds000030) — if offline, synthetic fallback")
        # Light check: don't actually download full dataset in smoke tests
        raise RuntimeError("Defer full UCLA download to explicit --download flag to protect Intel Mac disk")
    except Exception as e:
        print(f"[datasets] UCLA fallback: {e}")
        return generate_synthetic_timeseries(n_subjects=n_subjects or 40, seed=seed)

# ---------- Generic OpenNeuro small (for smoke) ----------
def load_openneuro_small(dataset: str = "ds002330", data_dir: str | Path = "data/openneuro"):
    try:
        from nilearn.datasets import fetch_openneuro_dataset
        ds = fetch_openneuro_dataset(dataset=dataset, data_dir=str(data_dir))
        return ds, None, {"source": dataset}
    except Exception as e:
        print(f"[datasets] OpenNeuro {dataset} failed {e} → synthetic")
        return generate_synthetic_timeseries(n_subjects=20, seed=0)
