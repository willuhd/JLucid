#!/usr/bin/env python3
"""Item-3 RESOLUTION (R35): the auditor's +0.319/p=0.0007/VR 2.94 REPRODUCED.

Recipe: site-median low-motion mask (per site, both groups pooled: mo<=site median)
applied to certified non-coded combined index, within-site perms (5000, seed 4100).
Result here: diff=+0.319 VR=2.96 p=0.0004 (n=87/219) — matches the auditor's first
report to 3 decimals on the diff, VR within .02, p within perm noise.
Pooled-median variant: +0.230 VR=2.65 p=0.031 (n=96/202) — matches the auditor's
revised pooled-median direction (+0.23/p=0.027 class).
Conclusion: the number is REAL under the site-median rule; fork denial ("not in
session") holds only because the rule postdates the fork. Split-rule dependence is
genuine (0.0004 vs 0.03) but BOTH variants support anti-motion (low-motion
stronger-or-equal vs full +0.251). Motion robustness: INTACT (with honest range).
Caveat: frozen-D null (certified construction); refit would shift up; VR nearly
untouched (mean-shift bias, not variance bias).
"""
import numpy as np
z = np.load("/Volumes/thinkplus/Code/JLucid/results/batch15_adhd200_comb.npz")
comb, do, s, mo, medk = z["comb"], z["do"], z["s"], z["mo"], z["medk"].astype(str)
non = (do == 0) | ((do == 1) & np.isin(medk, ["2", "-999", "nan"]))
rng = np.random.RandomState(4100)
def test(mask, name):
    m = mask & non & np.isfinite(comb)
    D = comb[m]; L = do[m]; S = s[m]
    o = np.nanmean(D[L == 1]) - np.nanmean(D[L == 0])
    vr = np.var(D[L == 1], ddof=1) / np.var(D[L == 0], ddof=1)
    nls = []
    for _ in range(5000):
        lp = L.copy()
        for st in np.unique(S):
            k = (S == st); lp[k] = rng.permutation(L[k])
        nls.append(np.nanmean(D[lp == 1]) - np.nanmean(D[lp == 0]))
    nls = np.array(nls)
    print(f"{name}: diff={o:+.3f} VR={vr:.2f} p={(np.sum(np.abs(nls) >= abs(o))+1)/5001:.4f} n={int((L==1).sum())}/{int((L==0).sum())}")
lm = np.zeros(len(do), bool)
for st in np.unique(s):
    k = (s == st); lm[k] = (mo[k] <= np.nanmedian(mo[k]))
test(lm, "site-median low")
test(mo <= np.nanmedian(mo[non]), "pooled-median low")
