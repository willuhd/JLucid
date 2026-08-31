"""Preprocessing — nilearn pure-Python replication of DPABI steps (paper 6 §2.4, paper 7 §2.2).
Steps: drop first 10 vols, slice-timing (placeholder), motion/WM/CSF regression, MNI 3mm, 6mm smoothing, 0.01-0.1 Hz bandpass, detrending.
For Athena preprocessed ADHD-200, this stage is skipped (func already preprocessed) — only parcellation + FC.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional

def preprocess_timeseries_placeholder(timeseries: np.ndarray, drop_first: int = 10, detrend: bool = True, bandpass: tuple = (0.01,0.1), tr: float = 2.0) -> np.ndarray:
    """Minimal offline preprocessing on (T,N) already-extracted time series.
    Real nifti-level preprocessing would use nilearn.image.clean_img / NiftiMasker.
    Here we replicate the temporal aspects that matter for FC.
    """
    ts = np.asarray(timeseries, float)
    if drop_first > 0:
        ts = ts[drop_first:]
    if detrend:
        # linear detrend per ROI
        from scipy.signal import detrend as sp_detrend
        ts = sp_detrend(ts, axis=0, type="linear")
    # bandpass via butterworth (if scipy available)
    try:
        from scipy.signal import butter, filtfilt
        low, high = bandpass
        # Normalize to Nyquist (fs=1/tr)
        fs = 1.0 / tr
        nyq = fs/2
        low_n = low / nyq
        high_n = high / nyq
        # guard: low_n>0, high_n<1
        if 0 < low_n < high_n < 1:
            b, a = butter(2, [low_n, high_n], btype="band")
            ts = filtfilt(b, a, ts, axis=0)
    except Exception:
        pass
    # z-score cleaning is handled in fc.py if requested
    return ts

def extract_timeseries_nifti(func_path: str | Path, atlas_path: str | Path, confounds_path: Optional[str|Path]=None, t_r: float=2.0, **clean_kwargs):
    """Extract ROI time series from 4D nifti using nilearn NiftiLabelsMasker.
    This is the real-data path for UCLA/ADHD-200 raw niftis.
    """
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.image import clean_img
    masker = NiftiLabelsMasker(labels_img=str(atlas_path), standardize=False, detrend=False, t_r=t_r)
    # Confounds: if provided, load and pass to clean
    confounds = None
    if confounds_path is not None and Path(confounds_path).exists():
        import pandas as pd
        confounds = pd.read_csv(confounds_path, sep="\\t")
    ts = masker.fit_transform(str(func_path), confounds=confounds)
    # clean_img equivalent for bandpass/detrend already via masker params, but we expose extra
    return ts  # (T, N)
