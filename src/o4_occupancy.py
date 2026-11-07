"""turn6 / o2 occupancy — instrumented re-test of the suppressed group claim.

Claim tested: ADHD-200 children spend less time in a specific de-novo LEiDA state
(the state whose pattern is denovo-index 4 / recon-index 0 / seed1-index 4 after denovo-order permutation);
the literal "o2/o4 index" claim is tested separately for identifiability.

Instrumentation (pre-stated before tests):
  MDE_d = 0.154 (simple) / 0.226 (null-SD). Falsifiers:
    KILL  if matched perm p > 0.10 or sign flip
    BOUND if |d| < 0.10
    REAL  if perm p < 0.01, FWER <= 0.05, SB >= 0.5, full covariates p < 0.05
  Exact null: outcome (DX) permuted within site; motion = per-run max motion from
  the per-site <Site>_motion.csv files (NOT the defective adhd200_maxmotion.tsv);
  cluster identifiability tested by 40 independent k-means refits matched by |cos|.

Outputs (this script writes flat files under outputs/).
Provenance: executed read-only by freq-audit during the revival run; numbers matched
a separately reported run (d=-0.302, perm <= .001, FWER .0010, SB .586).
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

rng = np.random.default_rng(0)
t0 = time.time()
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
ft = pd.read_csv("outputs/field_table.csv")
ft["sid"] = ft.sid.astype(str).str.zfill(7)
ENV = ["o0", "o1", "o2", "o3", "o4"]


def occ_from_labels(lab, k=5):
    agg = {}
    for key, a in lab.items():
        s = key.split("||")[0]
        if s not in agg:
            agg[s] = [np.zeros(k), 0]
        agg[s][0] += np.bincount(a, minlength=k)[:k]
        agg[s][1] += len(a)
    cols = [f"o{i}" for i in range(k)]
    O = pd.DataFrame([[s] + list(c / n) + [n] for s, (c, n) in agg.items()], columns=["sid"] + cols + ["nfr"])
    O["sid"] = O.sid.astype(str).str.zfill(7)
    return O, cols


def meta(y, df):
    ds, vs = [], []
    for s in [1, 3, 4, 5]:
        x = df[df.Site == s]
        a = x[x.ADHD == 1][y].dropna()
        b = x[x.ADHD == 0][y].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        dd = (a.mean() - b.mean()) / sp
        se = np.sqrt((len(a) + len(b)) / (len(a) * len(b)) + dd ** 2 / (2 * (len(a) + len(b))))
        ds.append(float(dd))
        vs.append(float(se ** 2))
    ds = np.array(ds)
    vs = np.array(vs)
    w = 1 / vs
    fe = float((ds * w).sum() / w.sum())
    Q = float((w * (ds - fe) ** 2).sum())
    dfree = len(ds) - 1
    tau2 = max(0.0, (Q - dfree) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (vs + tau2)
    re = float((ds * wr).sum() / wr.sum())
    re_p = float(2 * (1 - stats.norm.cdf(abs(re / np.sqrt(1 / wr.sum())))))
    return dict(d=fe, p=float(2 * (1 - stats.norm.cdf(abs(fe / np.sqrt(1 / w.sum()))))),
                ds=[round(float(v), 3) for v in ds], Q=Q,
                qp=float(stats.chi2.sf(Q, dfree)), i2=max(0.0, (Q - dfree) / Q) * 100,
                re=re, re_p=re_p, n=len(df))


def effect(lab, state, k=5):
    O, _ = occ_from_labels(lab, k)
    M = O.merge(ft, on="sid")
    A = M[M.ADHD.notna() & M.Site.isin([1, 3, 4, 5])]
    return meta(f"o{state}", A)


def perm(y, df, n=2000):
    obs = meta(y, df)["d"]
    null = np.empty(n)
    for i in range(n):
        d = df.copy()
        d["ADHD"] = d.groupby("Site").ADHD.transform(lambda s: rng.permutation(s.values))
        null[i] = meta(y, d)["d"]
    return float((1 + (np.abs(null) >= abs(obs)).sum()) / (n + 1))


# ---- features: e123 producer pipeline (z, FFT-Hilbert, 2x2 eigvec, argmax sign) ----
fs = sorted([f for f in glob.glob("data/adhd200/*/*/*cc200_TCs.1D") if "sfnwmrda" in os.path.basename(f)])
keys, X = [], []
for i, f in enumerate(fs):
    if sum(1 for _ in open(f)) - 1 < 120:
        continue
    ts = np.loadtxt(f, skiprows=1, usecols=range(2, 192))
    z = (ts - ts.mean(0)) / np.where(ts.std(0) == 0, 1e-12, ts.std(0))
    T = z.shape[0]
    F = np.fft.fft(z, axis=0)
    h = np.zeros(T)
    h[0] = 1
    h[T // 2] = 1
    h[1:T // 2] = 2
    ph = np.angle(np.fft.ifft(F * h[:, None], axis=0))
    out = np.empty((T, 190))
    for t in range(T):
        c = np.cos(ph[t])
        s = np.sin(ph[t])
        G = np.array([[c @ c, c @ s], [c @ s, s @ s]])
        _, V = np.linalg.eigh(G)
        v = V[:, -1]
        x = v[0] * c + v[1] * s
        n = np.linalg.norm(x)
        x = x / n if n > 0 else x
        k = int(np.argmax(np.abs(x)))
        x = -x if x[k] < 0 else x
        out[t] = x
    keys.append(f"{os.path.basename(os.path.dirname(f))}||{i}")
    X.append(out.astype(np.float32))
print("EXTRACT runs", len(keys), "frames", sum(len(x) for x in X), "sec", round(time.time() - t0, 1))

CEN = np.load("data/e123_adhd200_denovo_C.npy")  # never name this C (patsy C() shadowing)
L = {}
for name, p in [("denovo", "data/e123_adhd200_denovo_labels.npz"),
                ("recon", "data/e123_adhd200_filelens_recon.npz"),
                ("seed1", "data/e123_adhd200_seed1_labels.npz")]:
    d = np.load(p, allow_pickle=True)
    L[name] = {k: d[k] for k in d.files}
ag = [((X[i] @ CEN.T - 0.5 * (CEN * CEN).sum(1)).argmax(1) == L["denovo"][keys[i]]).mean()
      for i in range(0, len(keys), 10)]
print("VALIDATION agreement", round(float(np.mean(ag)), 4), "n", len(ag))

A0 = occ_from_labels(L["denovo"])[0].merge(ft, on="sid")
A = A0[A0.ADHD.notna() & A0.Site.isin([1, 3, 4, 5])]
print("GATES subs", len(occ_from_labels(L["denovo"])[0]), "balanced", len(A), "ADHD", int(A.ADHD.sum()))

for st in range(5):
    e = effect(L["denovo"], st)
    print(f"STATE o{st} d={e['d']:+.3f} p={e['p']:.4f} sites={e['ds']} Qp={e['qp']:.3f} I2={e['i2']:.0f}")
pp = perm("o4", A)
print("PERM_o4", pp)
mx = []
for i in range(1000):
    d = A.copy()
    d["ADHD"] = d.groupby("Site").ADHD.transform(lambda s: rng.permutation(s.values))
    mx.append(max(abs(meta(f"o{s}", d)["d"]) for s in range(5)))
fwer = float((1 + (np.array(mx) >= abs(effect(L["denovo"], 4)["d"])).sum()) / 1001)
print("FWER", fwer)
print("CROSS recon_o0", {k: effect(L["recon"], 0)[k] for k in ("d", "p", "n")},
      "seed1_o4", {k: effect(L["seed1"], 4)[k] for k in ("d", "p", "n")})

A2 = A.dropna(subset=["Age", "sex", "maxmotion_mm", "amp", "T"]).copy()
A2["logmot"] = np.log(A2.maxmotion_mm.astype(float).clip(lower=1e-3))
A2["logT"] = np.log(A2["T"].astype(float))
for name, fml in [("motion", "o4 ~ ADHD + C(Site) + logmot"),
                  ("motion+age+sex", "o4 ~ ADHD + C(Site) + logmot + Age + sex"),
                  ("+T", "o4 ~ ADHD + C(Site) + logmot + Age + sex + logT"),
                  ("+amp", "o4 ~ ADHD + C(Site) + logmot + Age + sex + logT + amp")]:
    r = smf.ols(fml, data=A2).fit(cov_type="HC1")
    print(f"LADDER {name} b={r.params['ADHD']:+.4f} p={r.pvalues['ADHD']:.4f} N={int(r.nobs)}")

pairs = []
for s in [1, 3, 4, 5]:
    x = A2[A2.Site == s]
    a = x[x.ADHD == 1].sort_values("logmot")
    b = x[x.ADHD == 0].sort_values("logmot").copy()
    used = set()
    for _, ra in a.iterrows():
        cand = b[~b.index.isin(used)]
        if len(cand) == 0:
            break
        j = (cand.logmot - ra.logmot).abs().idxmin()
        used.add(j)
        pairs.append((ra.o4, cand.loc[j, "o4"]))
P = pd.DataFrame(pairs, columns=["a", "c"])
tt = stats.ttest_rel(P.a, P.c)
dm = (P.a - P.c).mean() / (P.a - P.c).std(ddof=1)
print(f"MATCHED n={len(P)} diff={(P.a-P.c).mean():+.4f} t={tt.statistic:.2f} p={tt.pvalue:.4f} d={dm:+.3f}")
for drop in [1, 3, 4, 5]:
    e = meta("o4", A[A.Site != drop])
    print(f"LOSO drop{drop} d={e['d']:+.3f} p={e['p']:.4f} N={e['n']}")

sub = {}
for k in keys:
    sub.setdefault(k.split("||")[0], []).append(k)
rows = []
for s, ks in sub.items():
    if len(ks) < 2:
        continue
    ks = sorted(ks)
    h1 = np.zeros(5)
    h2 = np.zeros(5)
    n1 = n2 = 0
    for j, k in enumerate(ks):
        c = np.bincount(L["denovo"][k], minlength=5)
        if j % 2 == 0:
            h1 += c
            n1 += len(L["denovo"][k])
        else:
            h2 += c
            n2 += len(L["denovo"][k])
    rows.append((h1[4] / n1, h2[4] / n2))
R = pd.DataFrame(rows, columns=["a", "b"])
rr = float(np.corrcoef(R.a, R.b)[0, 1])
sb = 2 * rr / (1 + rr)
bs = []
for i in range(2000):
    s = R.sample(len(R), replace=True, random_state=i)
    v = float(np.corrcoef(s.a, s.b)[0, 1])
    bs.append(2 * v / (1 + v))
print(f"SB n={len(R)} r={rr:.3f} SB={sb:.3f} CI=[{np.nanpercentile(bs,2.5):.3f},{np.nanpercentile(bs,97.5):.3f}]")

for nm, fn in [("max", lambda v: v.max()), ("min", lambda v: v.min()), ("ent", lambda v: 1 - (v * v).sum())]:
    O, _ = occ_from_labels(L["denovo"])
    O["x"] = O[ENV].apply(fn, axis=1)
    M = O.merge(ft, on="sid")
    e = meta("x", M[M.ADHD.notna() & M.Site.isin([1, 3, 4, 5])])
    print(f"INVARIANT {nm} d={e['d']:+.3f} p={e['p']:.4f}")

lab_gf = {keys[i]: (np.abs(X[i] @ CEN.T)).argmax(1) for i in range(len(keys))}
e = effect(lab_gf, 4)
print(f"GAUGEFREE d={e['d']:+.3f} p={e['p']:.4f}")


def majneg(x):
    s = np.sign(x).sum(1)
    y = x.copy()
    y[s > 0] *= -1
    return y


Xm = np.concatenate([majneg(x) for x in X])
Cm = KMeans(n_clusters=5, n_init=5, random_state=0).fit(Xm[rng.choice(len(Xm), size=len(Xm) // 3, replace=False)]).cluster_centers_
Mc = np.abs(Cm @ CEN.T / np.linalg.norm(Cm, axis=1, keepdims=True) / np.linalg.norm(CEN, axis=1)[None, :])
ri, ci = linear_sum_assignment(-Mc)
mp = {int(ci[j]): int(ri[j]) for j in range(5)}
lab_m = {keys[i]: (np.abs(X[i]) @ Cm.T).argmax(1) for i in range(len(keys))}
e = effect(lab_m, mp[4])
print(f"MAJNEG idx={mp[4]} cos={Mc[mp[4],4]:.3f} d={e['d']:+.3f} p={e['p']:.4f}")

res = []
for b in range(40):
    r2 = np.random.default_rng(1000 + b)
    pick = r2.choice(len(X), size=max(1, len(X) // 3), replace=False)
    Cr = KMeans(n_clusters=5, n_init=3, random_state=b).fit(np.concatenate([X[i] for i in pick])).cluster_centers_
    Mc = np.abs(Cr @ CEN.T / np.linalg.norm(Cr, axis=1, keepdims=True) / np.linalg.norm(CEN, axis=1)[None, :])
    ri, ci = linear_sum_assignment(-Mc)
    mp = {int(ci[j]): int(ri[j]) for j in range(5)}
    labs = {keys[i]: (X[i] @ Cr.T - 0.5 * (Cr * Cr).sum(1)).argmax(1) for i in range(len(keys))}
    res.append((effect(labs, mp[4])["d"], effect(labs, 4)["d"], mp[4], float(Mc[mp[4], 4])))
Rr = pd.DataFrame(res, columns=["d_matched", "d_literal4", "idx", "matchcos"])
print("REFIT mean", round(Rr.d_matched.mean(), 3), "sd", round(Rr.d_matched.std(), 3),
      "CI", np.round(Rr.d_matched.quantile([.025, .975]).values, 3),
      "neg", float((Rr.d_matched < 0).mean()), "cos", round(Rr.matchcos.mean(), 3))
print("REFIT literal4 mean", round(Rr.d_literal4.mean(), 3), "sd", round(Rr.d_literal4.std(), 3),
      "fracneg", float((Rr.d_literal4 < 0).mean()), "idx4frac", float((Rr.idx == 4).mean()),
      Rr.idx.value_counts().to_dict())

for k in [3, 4, 5, 6]:
    r2 = np.random.default_rng(7)
    Ck = KMeans(n_clusters=k, n_init=3, random_state=7).fit(
        np.concatenate([X[i] for i in r2.choice(len(X), size=len(X) // 3, replace=False)])).cluster_centers_
    Mc = np.abs(Ck @ CEN.T / np.linalg.norm(Ck, axis=1, keepdims=True) / np.linalg.norm(CEN, axis=1)[None, :])
    best = int(np.argmax(Mc[:, 4]))
    labs = {keys[i]: (X[i] @ Ck.T - 0.5 * (Ck * Ck).sum(1)).argmax(1) for i in range(len(keys))}
    O, cols = occ_from_labels(labs, k)
    M = O.merge(ft, on="sid")
    Mx = M[M.ADHD.notna() & M.Site.isin([1, 3, 4, 5])]
    e = meta(cols[best], Mx)
    print(f"KRANGE k={k} idx={best} cos={Mc[best,4]:.3f} d={e['d']:+.3f} p={e['p']:.4f}")

print("DONE secs", round(time.time() - t0, 1))

# persist
A0[["sid", "o4"]].merge(ft[["sid", "ADHD", "Site"]], on="sid").to_csv(f"{OUT}/o4_subject.csv", index=False)
Rr.to_csv(f"{OUT}/o4_refits.csv", index=False)
json.dump({"primary": {k: (v if isinstance(v, (int, float)) else list(v)) for k, v in effect(L["denovo"], 4).items()},
           "perm_o4": pp, "fwer": fwer, "sb": sb,
           "sb_ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
           "refit_mean": float(Rr.d_matched.mean()),
           "refit_ci": [float(Rr.d_matched.quantile(.025)), float(Rr.d_matched.quantile(.975))],
           "idx4_frac": float((Rr.idx == 4).mean()),
           "literal4_mean": float(Rr.d_literal4.mean())},
          open(f"{OUT}/o4_summary.json", "w"), indent=1)
