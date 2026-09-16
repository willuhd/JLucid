"""Step 3 data preparation (plan Task A / T1).

Builds the training bundle consumed by every step-3 task:

- cohort: the 307 motion-controlled subjects that have V1 series
  (keep_motion_final minus subject 2427408, which has no V1);
- per-subject aligned arrays: V1(t), k=3 LEiDA state labels (-1 on
  motion-scrubbed frames), scrubbing mask, and soft state probabilities;
- subject-level 5-fold split stratified by (dx_group, site);
- normalization stats (v1_mean / v1_std) computed over usable frames.

Full-length arrays are kept aligned (len(v1) == len(labels) == len(mask))
so the Neural ODE integrates across real temporal gaps; censored frames are
excluded from soft-probability rows, encoder windows, and every loss.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from jlucid.config import (
    KEEP_COL,
    QC_SUMMARY_MOTION,
    SEED,
    STEP3_DATA_H5,
    STEP3_DATASET,
    STEP3_DIR,
    STEP3_K_COND,
    STATE_LABELS_PARQUET,
    TC_FILE,
    WINDOW_TR,
)

log = logging.getLogger("step3_data")


def load_qc_cohort() -> pd.DataFrame:
    """Motion-controlled cohort intersected with subjects that have V1."""
    qc = pd.read_parquet(QC_SUMMARY_MOTION)
    qc = qc[qc[KEEP_COL]].copy()
    qc["raw_id"] = qc["raw_id"].astype(str)
    qc["scan_dir_id"] = qc["scan_dir_id"].astype(str)
    with h5py.File(TC_FILE, "r") as h5:
        v1_keys = set(h5["v1"].keys())
    qc = qc[qc["raw_id"].isin(v1_keys)]
    qc = qc.sort_values("raw_id").reset_index(drop=True)
    return qc


def load_subject(h5, labels: pd.DataFrame, raw_id: str, scan_dir_id: str) -> dict:
    """Return aligned v1/labels/mask for one subject (asserted equal length)."""
    v1 = h5[f"v1/{raw_id}"][:].astype(np.float32)
    mask = h5[f"mask/{raw_id}"][:].astype(bool)
    sub = labels[labels["scan_dir_id"].astype(str) == scan_dir_id]
    if sub.empty:
        raise KeyError(f"no state labels for scan_dir_id {scan_dir_id}")
    lab = sub.sort_values("frame")["state_label"].to_numpy(dtype=np.int8)
    if not (len(v1) == len(lab) == len(mask)):
        raise ValueError(
            f"length mismatch {raw_id}: v1={len(v1)} labels={len(lab)} mask={len(mask)}"
        )
    if not np.array_equal(mask, lab >= 0):
        raise ValueError(
            f"mask/label mismatch {raw_id}: mask {mask.sum()} usable vs labels "
            f"{(lab >= 0).sum()} valid"
        )
    return {"v1": v1, "labels": lab, "mask": mask}


def soft_state_prob(labels: np.ndarray, k: int = STEP3_K_COND,
                    ema_window: int = 0) -> np.ndarray:
    """Soft state-probability vector p(t).

    One-hot over the k=3 labels; censored frames (-1) get an all-zero row.
    ema_window optionally smooths with a centered moving average over
    valid frames only (sensitivity; 0 = plain one-hot).
    """
    t = len(labels)
    p = np.zeros((t, k), dtype=np.float32)
    valid = labels >= 0
    rows = np.where(valid)[0]
    p[rows, labels[valid]] = 1.0
    if ema_window > 1:
        out = np.zeros_like(p)
        half = ema_window // 2
        for i in rows:
            lo, hi = max(0, i - half), min(t, i + half + 1)
            idx = np.arange(lo, hi)
            idx = idx[(idx >= 0) & (idx < t)]
            idx = idx[labels[idx] >= 0]
            if len(idx):
                out[i] = p[idx].mean(axis=0)
        return out
    return p


def valid_windows(mask: np.ndarray, w: int = WINDOW_TR) -> np.ndarray:
    """Boolean (T,) marking frames whose full w-window (ending at t) is usable."""
    t = len(mask)
    cum = np.concatenate([[0], np.cumsum(mask)])
    out = np.zeros(t, dtype=bool)
    for end in range(w - 1, t):
        out[end] = (cum[end + 1] - cum[end + 1 - w]) == w
    return out


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = SEED) -> np.ndarray:
    """Subject-level folds stratified by (dx_group, site)."""
    strat = (df["dx_group"].astype(str) + "_" + df["site"].astype(str)).to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for f, (_, test_idx) in enumerate(skf.split(df, strat)):
        fold[test_idx] = f
    if (fold < 0).any():
        raise RuntimeError("StratifiedKFold failed to assign some subjects")
    return fold


def build_step3_data(out_dir: Path = STEP3_DIR,
                     data_h5: Path = STEP3_DATA_H5,
                     meta_csv: Path = STEP3_DATASET,
                     window: int = WINDOW_TR,
                     ema_window: int = 0) -> pd.DataFrame:
    """Write the step-3 training bundle; returns the metadata frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    qc = load_qc_cohort()
    labels = pd.read_parquet(STATE_LABELS_PARQUET)
    labels["scan_dir_id"] = labels["scan_dir_id"].astype(str)

    rows = []
    all_v1 = []
    with h5py.File(TC_FILE, "r") as h5, h5py.File(data_h5, "w") as out:
        for _, subj in qc.iterrows():
            sid = subj["raw_id"]
            rec = load_subject(h5, labels, sid, subj["scan_dir_id"])
            v1, lab, mask = rec["v1"], rec["labels"], rec["mask"]
            p = soft_state_prob(lab, ema_window=ema_window)
            wmask = valid_windows(mask, window)
            out.create_dataset(f"v1/{sid}", data=v1, compression="gzip")
            out.create_dataset(f"labels/{sid}", data=lab, compression="gzip")
            out.create_dataset(f"mask/{sid}", data=mask, compression="gzip")
            out.create_dataset(f"p/{sid}", data=p, compression="gzip")
            out.create_dataset(f"wmask/{sid}", data=wmask, compression="gzip")
            all_v1.append(v1[mask])
            rows.append({
                "raw_id": sid,
                "scan_dir_id": subj["scan_dir_id"],
                "dx_group": subj["dx_group"],
                "site": subj["site"],
                "age": subj.get("age", np.nan),
                "gender": subj.get("gender", np.nan),
                "mean_fd": subj.get("mean_fd", np.nan),
                "n_raw": int(len(v1)),
                "n_usable": int(mask.sum()),
                "n_windows": int(wmask.sum()),
            })
        pooled = np.concatenate(all_v1)
        out.attrs["v1_mean"] = pooled.mean(axis=0).astype(np.float32)
        out.attrs["v1_std"] = pooled.std(axis=0).astype(np.float32)
        out.attrs["n_subjects"] = len(rows)
        out.attrs["raw_frames"] = int(sum(r["n_raw"] for r in rows))
        out.attrs["usable_frames"] = int(sum(r["n_usable"] for r in rows))
        out.attrs["k_cond"] = STEP3_K_COND

    df = pd.DataFrame(rows)
    df["fold"] = make_folds(df)
    df.to_parquet(meta_csv, index=False)

    with h5py.File(data_h5, "r") as h5:
        raw_frames = int(h5.attrs["raw_frames"])
        usable_frames = int(h5.attrs["usable_frames"])

    log.info(
        "step3 data: %d subjects, raw %d / usable %d frames, k=%d, window=%d",
        len(df), raw_frames, usable_frames, STEP3_K_COND, window,
    )
    fold_tab = pd.crosstab(df["fold"], df["dx_group"]).to_string()
    log.info("fold balance: " + fold_tab.replace(chr(10), " | "))
    site_tab = pd.crosstab(df["site"], df["dx_group"]).to_string()
    log.info("site x dx: " + site_tab.replace(chr(10), " | "))
    log.info("excluded 2427408 (no V1); v1_mean/std written")
    return df


def load_step3_dataset() -> tuple[pd.DataFrame, dict]:
    """Read dataset.parquet plus normalization stats from train_data.h5."""
    df = pd.read_parquet(STEP3_DATASET)
    with h5py.File(STEP3_DATA_H5, "r") as h5:
        stats = {
            "v1_mean": h5.attrs["v1_mean"][:],
            "v1_std": h5.attrs["v1_std"][:],
            "raw_frames": int(h5.attrs["raw_frames"]),
            "usable_frames": int(h5.attrs["usable_frames"]),
            "k_cond": int(h5.attrs["k_cond"]),
        }
    return df, stats

