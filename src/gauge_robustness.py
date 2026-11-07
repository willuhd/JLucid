#!/usr/bin/env python3
"""Gauge robustness of the o4 occupancy effect (ADHD-200, k=5 lens).

Ported from .agents/Gauge_test_and_verification_2026-11-07.md sections 5.6-5.9
(frame extraction, gauge refits, max-statistic FWER, motion ladder on the
projective survivor), adapted to the lean tree: lean src/leida_lib instead of
inline helpers, repo-relative data/, flat ignored outputs/.

Design: refit the k=5 partition under three sign-gauge choices -- argmax (the
paper's gauge), majneg (Paper 7's gauge), and a gauge-free projective
partition (chordal distance on RP^(N-1)) -- and re-test the occupancy effect
in each under the same within-site permutation null with max-statistic FWER
over the five states (2,000 draws). A gauge artifact would appear under one
gauge and vanish under the others; survival under all three, strongest
gauge-free, rules that out.

Writes (gitignored via outputs/):
  outputs/gauge_robustness.json   per-partition effects, FWER, motion ladder
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import leida_lib as L

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "outputs")
NPERM = 2000
RES = {}


def g_argmax(X):
    return L.apply_gauge(np.asarray(X, float), "argmax")


def g_majneg(X):
    return L.apply_gauge(np.asarray(X, float), "majneg")


def build_frames():
    """Per-frame leading eigenvectors for all retained sfn runs (no detrend)."""
    Vraw, sids = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "adhd200", "*", "*",
                                           "*cc200_TCs.1D"))):
        if "sfnwmrda" not in os.path.basename(f):
            continue
        x = L.load_cc200(f)
        if x is None:
            continue
        X, _ = L.leida_v1(x)
        Vraw.append(X.astype(np.float32))
        sids += [os.path.basename(os.path.dirname(f))] * len(X)
    Vall = np.concatenate(Vraw)
    print(f"frames {Vall.shape}, runs {len(Vraw)}", flush=True)
    return Vall, np.array(sids)


def build_bal():
    """Subject table: phenotypics + per-site max motion (first match kept)."""
    ph = pd.read_csv(os.path.join(ROOT, "data", "adhd200",
                                  "adhd200_preprocessed_phenotypics.tsv"), sep="\t")
    ph["sid"] = ph["ScanDir ID"].astype(str).str.zfill(7)
    ph["ADHD"] = np.where(ph["DX"].astype(str).str.lower() == "pending", np.nan,
                          ph["DX"].astype(str).isin(["1", "2", "3"]).astype(float))
    ph["Site"] = pd.to_numeric(ph["Site"], errors="coerce").astype(int)
    ph["Age"] = pd.to_numeric(ph["Age"], errors="coerce")
    ph["sex"] = pd.to_numeric(ph["Gender"], errors="coerce")
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
    M = pd.concat(mot).drop_duplicates("sid", keep="first")
    bal = ph[["sid", "ADHD", "Site", "Age", "sex"]].merge(M, on="sid", how="left")
    return bal[bal.Site.isin([1, 3, 4, 5])].copy()


def occ_from_labels(lab, sids):
    out = {}
    for s in np.unique(sids):
        l = lab[sids == s]
        out[s] = np.bincount(l, minlength=5)[:5] / len(l)
    return out


def fe_meta(occ, state, bal):
    ds, vs = [], []
    for s in [1, 3, 4, 5]:
        sub = bal[bal.Site == s]
        a = np.array([occ[x][state] for x in sub[sub.ADHD == 1].sid])
        b = np.array([occ[x][state] for x in sub[sub.ADHD == 0].sid])
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 10 or len(b) < 10:
            continue
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = (a.mean() - b.mean()) / sp
        se = np.sqrt((len(a) + len(b)) / (len(a) * len(b)) + d ** 2 / (2 * (len(a) + len(b))))
        ds.append(float(d))
        vs.append(float(se ** 2))
    w = 1 / np.array(vs)
    return float((np.array(ds) * w).sum() / w.sum()), [round(float(v), 3) for v in ds]


def report(label, occ, cents, bal, Cn):
    print("\n--- %s ---" % label)
    res = {"states": {}}
    best = None
    for st in range(5):
        d, ds = fe_meta(occ, st, bal)
        res["states"][st] = {"d": d, "sites": ds}
        print("   state %d : FE d=%+.3f  sites=%s" % (st, d, ds))
        if best is None or abs(d) > abs(best[1]):
            best = (st, d)
    res["max_state"] = best[0]
    res["max_d"] = best[1]
    print("   MAX |d| : state %d d=%+.3f" % (best[0], best[1]))
    Cm = cents @ Cn.T
    r, c = linear_sum_assignment(-Cm)
    res["match"] = []
    for a, b in zip(r, c):
        d, ds = fe_meta(occ, int(a), bal)
        res["match"].append({"ref": int(b), "mine": int(a), "d": d,
                             "cos": float(Cm[a, b])})
        print("   ref%d <-> mine%d : d=%+.3f (|cos|=%.2f)" % (int(b), int(a), d, Cm[a, b]))
    return res


def projective_fit(Xsub, k=5, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xsub), k, replace=False)
    M = Xsub[idx] / np.linalg.norm(Xsub[idx], axis=1, keepdims=True)
    for _ in range(iters):
        S = Xsub @ M.T
        lab = np.argmax(np.abs(S), axis=1)
        sgn = np.sign(np.take_along_axis(S, lab[:, None], 1).ravel())
        newM = M.copy()
        for j in range(k):
            sel = lab == j
            if sel.sum() == 0:
                continue
            W = Xsub[sel] * sgn[sel, None]
            m = W.mean(0)
            n = np.linalg.norm(m)
            if n > 0:
                newM[j] = m / n
        if np.allclose(newM, M, atol=1e-6):
            M = newM
            break
        M = newM
    return M


def fe_meta_p(occ, state, pa, bal):
    ds, vs = [], []
    for s in [1, 3, 4, 5]:
        m = bal.Site.values == s
        a = np.array([occ[x][state] for x in bal.sid[m & (pa == 1)]])
        b = np.array([occ[x][state] for x in bal.sid[m & (pa == 0)]])
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 10 or len(b) < 10:
            continue
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = (a.mean() - b.mean()) / sp
        se = np.sqrt((len(a) + len(b)) / (len(a) * len(b)) + d ** 2 / (2 * (len(a) + len(b))))
        ds.append(float(d))
        vs.append(float(se ** 2))
    w = 1 / np.array(vs)
    return float((np.array(ds) * w).sum() / w.sum())


def fwer(occ, bal, rng):
    pa0 = bal.ADHD.values
    obs = np.array([fe_meta_p(occ, st, pa0, bal) for st in range(5)])
    mx = []
    for _ in range(NPERM):
        pa = pa0.copy()
        for s in [1, 3, 4, 5]:
            m = bal.Site.values == s
            pa[m] = rng.permutation(pa[m])
        per = np.array([fe_meta_p(occ, st, pa, bal) for st in range(5)])
        mx.append(np.abs(per).max())
    f = (1 + (np.array(mx) >= np.abs(obs).max()).sum()) / (NPERM + 1)
    return obs, float(f)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2)
    Vall, sids = build_frames()
    bal = build_bal()
    bal = bal[bal.sid.isin(set(sids))].copy()  # labels exist only for kept runs
    CEN = np.load(os.path.join(ROOT, "data", "e123_adhd200_denovo_C.npy"))
    Cn = CEN / np.linalg.norm(CEN, axis=1, keepdims=True)

    parts = {}
    sub = Vall[::5]
    out = {"partitions": {}}
    for mode, gf in [("argmax", g_argmax), ("majneg", g_majneg)]:
        Xg = gf(Vall.copy())
        km = KMeans(n_clusters=5, n_init=5, random_state=0).fit(gf(sub.copy()))
        occ = occ_from_labels(km.predict(Xg), sids)
        parts[mode] = (km.predict(Xg), km.cluster_centers_)
        out["partitions"][mode] = report(mode, occ, km.cluster_centers_, bal, Cn)
        _, fw = fwer(occ, bal, rng)
        out["partitions"][mode]["fwer"] = fw
        print("   FWER(max|d|) = %.4f" % fw)
    Mp = projective_fit(sub.copy())
    plab = np.argmax(np.abs(Vall @ Mp.T), axis=1)
    parts["projective"] = (plab, Mp)
    out["partitions"]["projective"] = report("projective (gauge-free)",
                                             occ_from_labels(plab, sids), Mp, bal, Cn)
    _, fw = fwer(occ_from_labels(plab, sids), bal, rng)
    out["partitions"]["projective"]["fwer"] = fw
    print("   FWER(max|d|) = %.4f" % fw)

    # motion ladder on the projective survivor (state with max |d|)
    occ = occ_from_labels(plab, sids)
    B = bal.copy()
    B["o4p"] = [occ[x][out["partitions"]["projective"]["max_state"]] for x in B.sid]
    B2 = B.dropna(subset=["Age", "sex", "maxmotion_mm"]).copy()
    B2["logmot"] = np.log(B2.maxmotion_mm.astype(float).clip(lower=1e-3))
    import statsmodels.formula.api as smf
    lad = {}
    for name, fml in [("site only", "o4p ~ ADHD + C(Site)"),
                      ("+motion", "o4p ~ ADHD + C(Site) + logmot"),
                      ("+age+sex", "o4p ~ ADHD + C(Site) + logmot + Age + sex")]:
        r = smf.ols(fml, data=B2).fit(cov_type="HC1")
        lad[name] = {"b": float(r.params["ADHD"]), "p": float(r.pvalues["ADHD"]),
                     "n": int(r.nobs)}
        print("%-12s b=%+.4f p=%.4f N=%d" % (name, lad[name]["b"], lad[name]["p"],
                                             lad[name]["n"]))
    out["projective_motion_ladder"] = lad

    with open(os.path.join(OUT, "gauge_robustness.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote %s/gauge_robustness.json" % OUT)


if __name__ == "__main__":
    main()
