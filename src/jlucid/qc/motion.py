"""Motion QC.

Stage 1 uses the per-scan summary CSVs (max displacement) as the subject-level
motion gate (plan section 14 fallback).  The Power-2012 framewise-displacement
implementation below is tested and ready for when the per-frame ``rp_*.1D``
files are added (deferred; they live in the ~95 GB original per-site preproc
tarballs; the ~42 GB filtfix repacks contain no rp_*.1D - verified on KKI
2026-08-11).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FD_THRESHOLD_MM, SCRUB_AFTER, SCRUB_BEFORE


def compute_fd(realign: np.ndarray) -> tuple[np.ndarray, float]:
    """Power et al. 2012 framewise displacement from 6 rigid-body params.

    realign: (T, 6) columns [roll, pitch, yaw, dS, dL, dP] as in the Athena
    ``rp_*.1D`` files.  AFNI ``-1Dfile`` (which produced the Athena rp files)
    writes rotations in **degrees** and translations in mm; rotations are
    converted to radians, then scaled by 50 mm to approximate cortical radius:
    FD_t = sum |d_trans_t| + 50 * sum |d_rot_rad_t|.
    """
    realign = np.asarray(realign, dtype=float)
    if realign.ndim != 2 or realign.shape[1] != 6:
        raise ValueError("expected (T, 6) realignment parameters")
    diff = np.abs(np.diff(realign, axis=0))
    # columns 0-2 are rotations (roll/pitch/yaw, degrees), 3-5 translations (mm)
    rot_rad = np.deg2rad(diff[:, :3])
    fd = diff[:, 3:].sum(axis=1) + 50.0 * rot_rad.sum(axis=1)
    fd = np.concatenate([[0.0], fd])
    return fd, float(fd.mean())


def scrub_mask(fd: np.ndarray, threshold: float = FD_THRESHOLD_MM,
               n_before: int = SCRUB_BEFORE, n_after: int = SCRUB_AFTER) -> np.ndarray:
    """Boolean keep-mask: True = keep.  Frames with FD > threshold are
    censored together with n_before preceding and n_after following frames."""
    fd = np.asarray(fd, dtype=float)
    bad = fd > threshold
    mask = np.ones(len(fd), dtype=bool)
    idx = np.flatnonzero(bad)
    for i in idx:
        lo = max(0, i - n_before)
        hi = min(len(fd), i + n_after + 1)
        mask[lo:hi] = False
    return mask


def motion_gate(motion: pd.DataFrame, max_mm: float) -> pd.Series:
    """Subject-level keep flag from the per-scan summary CSV."""
    return motion["max_motion_mm"] < max_mm
