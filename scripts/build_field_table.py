#!/usr/bin/env python3
"""Build flat field_table.csv (+ field_runs.csv) from retained inputs.

Ported from .agents/scripts/t3_07_partitionfree.py (field-metric math,
subject aggregation, pheno/motion join) and the subject-table join semantics
of .agents/scripts/t3_04_amplitude.py, adapted to the lean tree:
  - lean src/leida_lib instead of archived t3_00_lib
  - repo-relative data/ instead of legacy datasets/
  - flat ignored outputs/ instead of outputs/turn3/

Writes (gitignored via outputs/):
  outputs/field_runs.csv   per-run R1/R2/gap/disp/rho1/dphi + amp/dvars/T
  outputs/field_table.csv  subject-level metrics + ADHD/Site/Age/sex/maxmotion_mm
                           + logdvars[_z], maxmotion_mm_z, logT, amp_rank
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import leida_lib as L

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "outputs")
EST = ("R1", "R2", "gap", "disp", "rho1", "dphi")


def field_metrics(ph):
    """Partition-free per-frame field quantities from phases (T, N)."""
    T, N = ph.shape
    E1 = np.exp(1j * ph)
    E2 = np.exp(2j * ph)
    R1 = np.abs(E1.mean(1))
    R2 = np.abs(E2.mean(1))
    phi = 0.5 * np.angle(E2.sum(1))                    # cut direction
    d = np.angle(np.exp(1j * (ph - phi[:, None])))      # wrapped deviation
    disp = np.abs(d).mean(1)
    dphi = np.diff(np.unwrap(phi))                      # psi = 2*phi avoids pi jumps
    out = dict(R1=float(R1.mean()), R2=float(R2.mean()),
               gap=float(((1 - R2) / (1 + R2)).mean()),
               disp=float(disp.mean()),
               dphi=float(np.mean(np.abs(dphi)))) if len(dphi) else dict(
                   R1=float(R1.mean()), R2=float(R2.mean()),
                   gap=float(((1 - R2) / (1 + R2)).mean()),
                   disp=float(disp.mean()), dphi=np.nan)
    if len(dphi) > 5 and dphi.std() > 0:
        z = np.exp(1j * dphi)
        out["rho1"] = float(np.abs(z.mean()))
    else:
        out["rho1"] = np.nan
    return out


def load_motion():
    """Per-subject max motion from retained per-site tables (first match kept)."""
    mot = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "adhd200", "*", "*_motion.csv"))):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "File" not in d.columns or "Max Motion (mm)" not in d.columns:
            continue
        sn = d["File"].str.extract(r"rp_(\d+)_")[0]
        mot.append(pd.DataFrame({
            "sid": sn.str.zfill(7),
            "maxmotion_mm": pd.to_numeric(d["Max Motion (mm)"], errors="coerce"),
        }))
    if not mot:
        return pd.DataFrame(columns=["sid", "maxmotion_mm"])
    return pd.concat(mot).drop_duplicates("sid", keep="first")


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = L.cc200_paths("sfnwmrda")
    rows, subs, amps, dvars, Ts = [], [], [], [], []
    for i, f in enumerate(fs):
        x = L.load_cc200(f)
        if x is None:
            continue
        _, ph = L.leida_v1(x)
        r = field_metrics(ph)
        r["sub"] = L.subject_of(f)
        r["run"] = os.path.basename(f)
        rows.append(r)
        subs.append(r["sub"])
        Ts.append(len(x))
        amps.append(float(np.median(x.std(0))))
        dv = np.sqrt(((np.diff(x, axis=0)) ** 2).mean(1))
        dvars.append(float(np.median(dv)))
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(fs)}", flush=True)
    R = pd.DataFrame(rows)
    print(f"\nruns {len(R)} subjects {R['sub'].nunique()}", flush=True)

    S = R.groupby("sub")[list(EST)].mean().reset_index().rename(columns={"sub": "sid"})
    M = pd.DataFrame({"sid": subs, "amp": amps, "dvars": dvars, "T": Ts})
    M = M.groupby("sid").mean(numeric_only=True).reset_index()
    S = S.merge(M, on="sid")
    S["logdvars"] = np.log(S.dvars.clip(lower=1e-9))

    ph = pd.read_csv(os.path.join(ROOT, "data", "adhd200",
                                  "adhd200_preprocessed_phenotypics.tsv"), sep="\t")
    ph["sid"] = ph["ScanDir ID"].astype(str).str.zfill(7)
    ph["ADHD"] = np.where(ph["DX"].astype(str).str.lower() == "pending", np.nan,
                          ph["DX"].astype(str).isin(["1", "2", "3"]).astype(float))
    ph["Site"] = pd.to_numeric(ph["Site"], errors="coerce")
    ph["Age"] = pd.to_numeric(ph["Age"], errors="coerce")
    ph["sex"] = pd.to_numeric(ph["Gender"], errors="coerce")
    S = S.merge(ph[["sid", "ADHD", "Site", "Age", "sex"]], on="sid", how="left")
    S = S.merge(load_motion(), on="sid", how="left")
    S = S.dropna(subset=["Site", "Age", "sex"]).copy()
    S["Site"] = S["Site"].astype(int)
    for c in ("logdvars", "maxmotion_mm"):
        S[c + "_z"] = S.groupby("Site")[c].transform(lambda s: (s - s.mean()) / s.std())
    S["logT"] = np.log(S["T"].clip(lower=10))
    S["amp_rank"] = S.groupby("Site")["amp"].rank(pct=True)

    R.to_csv(os.path.join(OUT, "field_runs.csv"), index=False)
    S.to_csv(os.path.join(OUT, "field_table.csv"), index=False)
    print(f"analytic N={len(S)} ADHD={int(S.ADHD.sum())}")
    print(f"wrote {OUT}/field_runs.csv and {OUT}/field_table.csv")


if __name__ == "__main__":
    main()
