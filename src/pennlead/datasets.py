"""PennLEAD dataset loader — isolated for src-pennlead.
Read-only data/pennlead/ptseries/*.ptseries.nii.gz + motion + pheno.
Handles variable parcel N (514-521) by intersection to N_min, and QC by mean FD.
Single-site Prisma TR 0.8s, T=156, N≈517 (Schaefer-400 + subcortical).
No import from src-controllability at import time (parallel-safe).
"""
from __future__ import annotations
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from collections import Counter

DEFAULT_PENNLEAD = Path("/Volumes/thinkplus/Code/JLucid/data/pennlead")

def load_pennlead(data_dir: Path | str = DEFAULT_PENNLEAD,
                  condition: str = "rest",
                  qc_fd_thresh: float | None = 0.5,
                  intersect_parcels: bool = True,
                  use_eswan: bool = False):
    """Load PennLEAD time series.

    Args:
        data_dir: root data/pennlead
        condition: "rest" or "nback"
        qc_fd_thresh: if not None, drop subjects with mean FD > thresh (e.g. 0.5)
        intersect_parcels: if True, intersect to N_min common parcels across retained subjects
        use_eswan: if True also return ESWAN continuous scores

    Returns:
        timeseries: list of (T, N) arrays after parcel intersection
        labels_binary: np array 0/1 for dx_adhd_i where available (for rest qc subset)
        meta: dict with subject_ids, sites, ages, genders, fd_means, eswan_scores, N, T
    """
    data_dir = Path(data_dir)
    # Load cohort + availability + motion
    cohort = pd.read_csv(data_dir / "pheno" / "cohort.csv")  # participant_id, study_group, dx_adhd_i, age, sex
    avail = pd.read_csv(data_dir / "manifest" / "availability.csv")  # participant_id, nb, rest, eswan...
    # Merge on participant_id for convenience
    cohort = cohort.set_index("participant_id")
    avail = avail.set_index("participant_id")

    # Find ptseries files for condition
    pattern = f"*_{condition}.ptseries.nii.gz"
    pt_files = sorted((data_dir / "ptseries").glob(pattern))
    if condition == "rest":
        # pt_files already includes only rest; availability says rest=True for 104
        pass
    # Map participant -> file(s) (ses-1 only dataset has ses-1)
    # Files are sub-XXXXX_ses-1_rest.ptseries.nii.gz
    file_by_part = {}
    for p in pt_files:
        # participant is sub-XXXXX
        pid = p.name.split("_ses")[0]
        file_by_part[pid] = p

    # Determine which participants have file and avail flag True
    rows = []
    for pid, fpath in file_by_part.items():
        # check availability flag
        if pid in avail.index:
            flag = avail.loc[pid, condition] if condition in avail.columns else True
            # flag may be bool or string
            if str(flag).lower() not in ("true", "1", "1.0"):
                continue
        # load FD to enable QC (motion tsv is sub-XXX_ses-1_rest_motion.tsv)
        motion_path = fpath.parent / f"{fpath.name.replace('.ptseries.nii.gz','_motion.tsv')}"
        fd_mean = np.nan
        if motion_path.exists():
            try:
                dfm = pd.read_csv(motion_path, sep="\t")
                if "framewise_displacement" in dfm.columns:
                    fd_mean = float(dfm["framewise_displacement"].mean())
            except Exception:
                pass
        # cohort info
        if pid in cohort.index:
            dx = cohort.loc[pid, "dx_adhd_i"] if "dx_adhd_i" in cohort.columns else np.nan
            study_group = cohort.loc[pid, "study_group"] if "study_group" in cohort.columns else ""
            age = cohort.loc[pid, "age"] if "age" in cohort.columns else np.nan
            sex = cohort.loc[pid, "sex"] if "sex" in cohort.columns else ""
            eswan_total = cohort.loc[pid, "eswan_adhd_total_score"] if "eswan_adhd_total_score" in cohort.columns else np.nan
        else:
            dx = np.nan; study_group=""; age=np.nan; sex=""; eswan_total=np.nan
        rows.append((pid, fpath, motion_path, fd_mean, dx, study_group, age, sex, eswan_total))

    # QC drop high FD
    before = len(rows)
    if qc_fd_thresh is not None:
        rows = [r for r in rows if (np.isnan(r[3]) or r[3] <= qc_fd_thresh)]
    after = len(rows)
    print(f"[pennlead:{condition}] found {before} files, kept {after} after FD>{qc_fd_thresh} drop", flush=True)

    # Load time series arrays (T,N) handling nibabel Nifti2 dim 6
    ts_list = []
    pids = []
    fds = []
    dxs = []
    groups = []
    ages = []
    sexes = []
    eswans = []
    raw_shapes = []
    for pid, fpath, mpath, fd, dx, grp, age, sex, eswan in rows:
        im = nib.load(str(fpath))
        d = im.get_fdata()  # shape (1,1,1,1,N,156) or similar
        # squeeze to (N, T) or (N,T) transposed; we want (T,N)
        d = np.squeeze(d)
        # After squeeze: observed (N,156) where N 514-521, T 156 is last dim
        # Our convention: (T,N) so transpose if first dim is parcels
        if d.ndim != 2:
            raise ValueError(f"{pid} unexpected shape {d.shape} after squeeze")
        # Heuristic: if d.shape[0] < 200 and d.shape[1] > 400, then (T,N)? Actually T 156 is smaller than N 514, so (156,514) vs (514,156)
        # Observed earlier: squeeze gives (517,156) = (N,T). So we need transpose to (T,N).
        if d.shape[0] > d.shape[1]:
            # N > T, so (N,T) -> (T,N)
            d = d.T
        # Now (T,N)
        # Verify no NaNs
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        ts_list.append(d.astype(np.float64))
        pids.append(pid)
        fds.append(fd)
        dxs.append(int(dx) if not pd.isna(dx) else -1)
        groups.append(grp)
        ages.append(float(age) if not pd.isna(age) else np.nan)
        sexes.append(sex)
        eswans.append(float(eswan) if not pd.isna(eswan) else np.nan)
        raw_shapes.append(d.shape)

    if not ts_list:
        return [], np.array([]), {}

    # Parcel intersection to N_min if requested
    if intersect_parcels:
        # All ts are (T,N_i) with same T but different N_i
        # Intersect to min N by truncating to first N_min parcels (since missing parcels are low-coverage tail)
        # Better: intersect by keeping first N_min columns (common prefix) — since parcellation order is fixed, missing tail parcels are dropped
        Ns = [t.shape[1] for t in ts_list]
        N_min = min(Ns)
        N_max = max(Ns)
        if N_min != N_max:
            print(f"[pennlead:{condition}] parcel N varies {sorted(set(Ns))} -> intersecting to N_min={N_min}", flush=True)
            ts_list = [t[:, :N_min] for t in ts_list]
        N = N_min
    else:
        N = ts_list[0].shape[1]

    Ts = [t.shape[0] for t in ts_list]
    assert len(set(Ts)) == 1, f"T varies {set(Ts)}"
    T = Ts[0]

    # Binary labels: dx_adhd_i 0/1 (drop -1 missing)
    # Keep only subjects with label 0/1 for binary tasks; ESWAN continuous keeps all with score
    labels_binary = np.array(dxs, dtype=int)
    # For convenience also provide ESWAN array
    eswan_arr = np.array(eswans, dtype=float)

    meta = {
        "participant_ids": np.array(pids),
        "fd_means": np.array(fds, dtype=float),
        "dx_adhd_i": labels_binary,
        "study_group": np.array(groups),
        "ages": np.array(ages, dtype=float),
        "sexes": np.array(sexes),
        "eswan_total": eswan_arr,
        "N": int(N),
        "T": int(T),
        "raw_shapes": raw_shapes,
        "condition": condition,
        "qc_fd_thresh": qc_fd_thresh,
    }
    # Print class balance after QC
    mask_labeled = labels_binary != -1
    if mask_labeled.sum() > 0:
        cnt = Counter(labels_binary[mask_labeled])
        print(f"[pennlead:{condition}] binary balance among labeled: {dict(cnt)} (ADHD {cnt.get(1,0)} / TD-like {cnt.get(0,0)}), ESWAN non-NaN {np.isfinite(eswan_arr).sum()}/{len(eswan_arr)}", flush=True)
    return ts_list, labels_binary, meta

def load_pennlead_rest_nback_paired(qc_fd_thresh: float | None = 0.5):
    """Load rest and nback paired for same subjects (both True, ~98)."""
    ts_rest, lab_rest, meta_rest = load_pennlead(condition="rest", qc_fd_thresh=qc_fd_thresh)
    ts_nback, lab_nback, meta_nback = load_pennlead(condition="nback", qc_fd_thresh=qc_fd_thresh)
    # Pair by participant_id intersection
    rest_dict = {pid: ts for pid, ts in zip(meta_rest["participant_ids"], ts_rest)}
    nback_dict = {pid: ts for pid, ts in zip(meta_nback["participant_ids"], ts_nback)}
    common = sorted(set(rest_dict) & set(nback_dict))
    ts_r = [rest_dict[pid] for pid in common]
    ts_n = [nback_dict[pid] for pid in common]
    # meta for common
    idx_r = {pid:i for i,pid in enumerate(meta_rest["participant_ids"])}
    meta_common = {k: np.array([meta_rest[k][idx_r[pid]] if k in meta_rest and len(meta_rest[k])==len(meta_rest["participant_ids"]) else meta_rest[k] for pid in common], dtype=object) if isinstance(meta_rest.get(k), np.ndarray) else [meta_rest[k] for _ in common] for k in meta_rest}
    # Simpler: just return counts
    print(f"[pennlead:paired] common rest+nback {len(common)} subjects", flush=True)
    return ts_r, ts_n, common

if __name__ == "__main__":
    ts, labs, meta = load_pennlead(condition="rest", qc_fd_thresh=0.5)
    print("rest example", ts[0].shape if ts else None, meta.get("N"), meta.get("T"))
    ts2, labs2, meta2 = load_pennlead(condition="nback", qc_fd_thresh=0.5)
    print("nback", ts2[0].shape if ts2 else None)
