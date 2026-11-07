#!/usr/bin/env python3
"""Canonical true-per-run motion check for the k=5 occupancy effect.
Writes outputs/turn7/essay_motion_stats.json; used by the manuscript.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, "outputs", "turn7", "essay_motion_stats.json")


def main():
    fs = sorted([f for f in glob.glob(os.path.join(ROOT, "data", "adhd200", "*", "*",
                                                   "*cc200_TCs.1D"))
                 if "sfnwmrda" in os.path.basename(f)])
    runs = []
    for f in fs:
        nt = sum(1 for _ in open(f)) - 1
        if nt < 120:
            continue
        b = os.path.basename(f)
        m = re.match(r"sfnwmrda(\d+)_(session_\d+_rest_\d+)_cc200_TCs.1D", b)
        if not m:
            continue
        runs.append((m.group(1).zfill(7), "rp_" + m.group(1) + "_" + m.group(2) + ".1D"))
    mot = {}
    for f in glob.glob(os.path.join(ROOT, "data", "adhd200", "*", "*_motion.csv")):
        try:
            mm = pd.read_csv(f)
        except Exception:
            continue
        if "File" not in mm.columns or "Max Motion (mm)" not in mm.columns:
            continue
        for _, row in mm.iterrows():
            fn = str(row["File"])
            v = pd.to_numeric(row["Max Motion (mm)"], errors="coerce")
            if fn.startswith("rp_") and np.isfinite(v):
                mot[fn] = float(v)
    M = pd.DataFrame(runs, columns=["sid", "run"])
    M["mm"] = M.run.map(mot)
    agg = M.groupby("sid").mm.mean().reset_index()

    O = pd.read_csv(os.path.join(ROOT, "outputs", "turn6", "o2", "o4_subject.csv"),
                    dtype={"sid": str})
    O["sid"] = O.sid.str.zfill(7)
    A = O[O.Site.isin([1, 3, 4, 5]) & O.ADHD.notna()].copy()
    A = A.merge(agg, on="sid", how="left")

    pairs = []
    for site, g in A.dropna(subset=["mm"]).groupby("Site"):
        a = g[g.ADHD == 1]
        b = g[g.ADHD == 0].copy()
        used = set()
        for i, row in a.iterrows():
            cand = b[~b.index.isin(used)].copy()
            if len(cand) == 0:
                break
            j = (np.log(cand.mm) - np.log(row.mm)).abs().idxmin()
            used.add(j)
            pairs.append((row.o4, cand.loc[j, "o4"]))
    P = pd.DataFrame(pairs, columns=["a", "c"])
    diff = P.a.values - P.c.values
    dz = float(diff.mean() / diff.std(ddof=1))
    tp = stats.ttest_rel(P.a, P.c).pvalue
    sp = float(np.sqrt((P.a.var(ddof=1) + P.c.var(ddof=1)) / 2))
    dbt = float((P.a.mean() - P.c.mean()) / sp)
    rng = np.random.default_rng(20261106)
    bs = []
    for _ in range(5000):
        ii = rng.integers(0, len(P), len(P))
        aa = P.a.values[ii]
        cc = P.c.values[ii]
        spp = np.sqrt((aa.var(ddof=1) + cc.var(ddof=1)) / 2)
        bs.append((aa.mean() - cc.mean()) / spp)
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    # Welch test on matched subjects
    pbt = stats.ttest_ind(P.a, P.c, equal_var=False).pvalue

    B = A.dropna(subset=["mm"]).copy()
    B["logmm"] = np.log(B.mm)
    m1 = smf.ols("o4 ~ ADHD + C(Site) + logmm", data=B).fit(cov_type="HC3")
    res = {
        "n_pairs": int(len(P)),
        "paired_dz": dz,
        "paired_p": float(tp),
        "between_group_d": dbt,
        "between_group_d_ci": ci,
        "between_group_p_welch": float(pbt),
        "ols_site_motion_b": float(m1.params["ADHD"]),
        "ols_site_motion_p": float(m1.pvalues["ADHD"]),
        "n_ols": int(m1.nobs),
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
