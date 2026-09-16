#!/usr/bin/env python3
"""C1-C4 specificity verification from batch15c artifact (no rebuild needed).

The fork session names but does not quote the C1-C4 heredoc (TD-only restriction).
This script re-derives the four contrasts directly from the VERIFIED batch15c
artifact + cohort.csv, single-site pooled perms (exchangeable; matches fork's
conventions: seeds explicit, (k+1)/(n+1)). Closes hole #6 by recomputation.
Prespecified contrasts (sieve-table/dispersion.md values in comments).
"""
import numpy as np, pandas as pd
BASE = "/Volumes/thinkplus/Code/JLucid"
z = np.load(f"{BASE}/results/batch15c_penn_dispz.npz", allow_pickle=True)
Dn = z["comb_nback"]; dxn = z["dx_nback"]
pids = [str(x) for x in z["pids_nback"]]
cohort = pd.read_csv(f"{BASE}/data/pennlead/pheno/cohort.csv", dtype=str)
cm = cohort.set_index("participant_id")
grp = np.array([str(cm.loc[p, "study_group"]) if p in cm.index else "?" for p in pids])
is_td = grp == "TD/NC"
is_adhd_dx = dxn == 1
is_primary = is_adhd_dx & (grp == "ADHD")
is_comorb = is_adhd_dx & (grp == "PRO/CHR")
is_pronoadhd = (grp == "PRO/CHR") & (dxn == 0)
rng = np.random.RandomState(33)
def perm2(a, b, n=5000):
    a = np.array(a); b = np.array(b)
    o = np.nanmean(a) - np.nanmean(b)
    pool = np.concatenate([a, b]); nls = []
    for _ in range(n):
        rp = rng.permutation(pool)
        nls.append(rp[:len(a)].mean() - rp[len(a):].mean())
    nls = np.array(nls)
    return o, (np.sum(np.abs(nls) >= abs(o))+1)/(n+1), (np.sum(nls >= o)+1)/(n+1)
print("C1 ADHD(dx=1) vs TD/NC-only [certified +0.318, p~0.038]:")
for name, mA in [("C1 all-ADHD", is_adhd_dx), ("C2 primary-ADHD", is_primary),
                 ("C3 comorbid PRO+ADHD", is_comorb), ("C4 PRO-noADHD", is_pronoadhd)]:
    D = Dn[np.isfinite(Dn)]
    a = Dn[mA & np.isfinite(Dn)]; b = Dn[is_td & np.isfinite(Dn)]
    o, p2, p1 = perm2(a, b)
    print(f"  {name:22s} obs={o:+.3f} two-sided p={p2:.4f} one-sided p={p1:.4f} (n={len(a)}/{len(b)})")
