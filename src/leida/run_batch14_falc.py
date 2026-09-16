#!/usr/bin/env python3
"""exp/37 BATCH 14 — fALFF anchors + FC decomposition (preregistered)."""
import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass
BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"

sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200

def morlet(TR, f, cycles, cap):
    sigma = cycles / (2 * np.pi * f)
    L = int(6 * sigma / TR); L3 = int(3.0 / f / TR); L = max(L, L3)
    if L % 2 == 0: L += 1
    Lcap = int(cap / TR); Lcap = Lcap if Lcap % 2 == 1 else Lcap - 1
    if L > Lcap: L = Lcap
    t = (np.arange(L) - L // 2) * TR
    w = np.pi ** -0.25 * np.exp(1j * 2 * np.pi * f * t) * np.exp(-t ** 2 / (2 * sigma ** 2))
    return w / np.sqrt(np.sum(np.abs(w) ** 2) + 1e-12), L
def cwt(ts, w, L):
    T, N = ts.shape
    n_fft = 1
    while n_fft < T + L: n_fft <<= 1
    return np.fft.ifft(np.fft.fft(ts, n=n_fft, axis=0) * np.fft.fft(w, n=n_fft)[:, None], axis=0)[L // 2:L // 2 + T]

t0 = time.time()
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites = np.array(meta_all["sites"])
ages = np.array(meta_all["ages"], dtype=float)
dx = np.array(labels_all).astype(int)
n = len(ts_all)
yeo = json.load(open(f"{BASE}/src/leida/atlas/yeo_true_idx.json"))
NETS = ['VIS','SOM','DAN','SAL','LIM','FPN','DMN']
net_idx = [np.array(yeo[nn]) for nn in NETS]
UP = [(a, b) for a in range(7) for b in range(a, 7)]
# meta-analytic region sets from cc200 ROI names? Use network proxies:
# frontal/OFC ≈ FPN+LIM, precuneus ≈ DMN subset, occipital ≈ VIS (closest available).
# fALFF-style: mean |W| in band / mean |W| broadband. Broadband = std of full ts per ROI.
TR = 2.0

def worker(args):
    ts, = args
    out = {}
    for bname, (f, cyc, cap) in {"s5": (0.02, 5, 60.0), "s4": (0.05, 5, 60.0)}.items():
        w, L = morlet(TR, f, cyc, cap)
        if ts is None or ts.shape[0] < L + 20:
            out[bname] = None
            continue
        Wk = cwt(ts, w, L)
        A = np.abs(Wk) * 2
        band_mean_roi = A.mean(axis=0)
        broad = ts.std(axis=0)
        frac = band_mean_roi / (broad + 1e-12)
        # region features: network means of frac
        out[bname] = np.array([frac[ix].mean() for ix in net_idx])
        # FC decomposition (slow-4 only): PL factor and amp factor per network pair
        if bname == "s4":
            ph = np.angle(Wk)
            pl_f = np.zeros(28); am_f = np.zeros(28)
            for kk, (a_, b_) in enumerate(UP):
                ia, ib = net_idx[a_], net_idx[b_]
                dtheta = ph[:, ia][:, :, None] - ph[:, ib][:, None, :]
                plmat = np.cos(dtheta)
                Aa = A[:, ia]; Ab = A[:, ib]
                # amp co-fluctuation: mean(Aa_i * Ab_j) normalized by outer sigma
                az = (Aa - Aa.mean(0)) / (Aa.std(0) + 1e-12)
                bz = (Ab - Ab.mean(0)) / (Ab.std(0) + 1e-12)
                ammat = (az[:, :, None] * bz[:, None, :]).mean(axis=0)
                if a_ == b_:
                    ii, jj = np.triu_indices(len(ia), k=1)
                    pl_f[kk] = plmat[:, ii, jj].mean()
                    am_f[kk] = ammat[ii, jj].mean()
                else:
                    pl_f[kk] = plmat.mean()
                    am_f[kk] = ammat.mean()
            out["pl_f28"] = pl_f
            out["am_f28"] = am_f
    return out

with mp.Pool(4) as pool:
    outs = pool.map(worker, [(ts_all[i],) for i in range(n)], chunksize=16)
print(f"features done ({time.time()-t0:.0f}s)")

FEAT = {
    "s5_frac7": np.array([o["s5"] if o and o.get("s5") is not None else np.full(7, np.nan) for o in outs]),
    "s4_frac7": np.array([o["s4"] if o and o.get("s4") is not None else np.full(7, np.nan) for o in outs]),
    "s4_pl_f28": np.array([o.get("pl_f28", np.full(28, np.nan)) if o else np.full(28, np.nan) for o in outs]),
    "s4_am_f28": np.array([o.get("am_f28", np.full(28, np.nan)) if o else np.full(28, np.nan) for o in outs]),
}
# narrowband FC itself: pl_f * am_f product ≈ FC proxy
FEAT["s4_fc28"] = FEAT["s4_pl_f28"] * FEAT["s4_am_f28"]
np.savez_compressed(f"{OUT}/batch14_features.npz", sites=sites, **FEAT)

ph = pd.read_csv(f"{BASE}/data/adhd200/adhd200_preprocessed_phenotypics.tsv", sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
for c in ["Inattentive", "Hyper/Impulsive"]:
    ph[c + "_n"] = pd.to_numeric(ph[c], errors="coerce")
    ph.loc[ph[c + "_n"] == -999, c + "_n"] = np.nan
phmap = ph.set_index("ScanDir ID")[["Inattentive_n", "Hyper/Impulsive_n", "ADHD Measure"]]
gmap = dict(zip(ph["ScanDir ID"], ph["Gender"].fillna(0).astype(int)))
sex_all = np.array([gmap.get(s, 0) for s in ids_all], dtype=float)
mm = pd.read_csv(f"{BASE}/data/adhd200/adhd200_maxmotion.tsv", sep="\t")
mm["ScanDir ID"] = mm["ScanDir ID"].astype(str)
mmmap = dict(zip(mm["ScanDir ID"], mm["maxmotion_mm"]))
age_i = ages.copy(); age_i[np.isnan(age_i)] = np.nanmean(ages)
mot = np.array([float(mmmap.get(s, np.nan)) for s in ids_all]); mot[np.isnan(mot)] = np.nanmean(mot)

keep_T, keep_DX = [], []
for i, sid in enumerate(ids_all):
    if sid not in phmap.index: continue
    meas = str(phmap.loc[sid, "ADHD Measure"])
    if sites[i] in (3, 5, 6) and meas in ("2", "3") and not pd.isna(phmap.loc[sid, "Inattentive_n"]):
        keep_T.append(i)
    if sites[i] in (1, 3, 4, 5, 6) and str(dx[i]) in ("0", "1"):
        keep_DX.append(i)
keep_T = np.array(keep_T); keep_DX = np.array(keep_DX)

ALPHAS = [0.1, 1, 10, 100, 1000, 3000, 1e4, 3e4, 1e5, 3e5]
def site_demean(v, s, tr):
    out = v.copy()
    for st in np.unique(s[tr]):
        m = tr & (s == st)
        out[m] = v[m] - v[m].mean()
    return out
def rfp(Xtr, ytr, Xte, a):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-12
    Xz = (Xtr-mu)/sd
    w = np.linalg.solve(Xz.T@Xz + a*np.eye(Xz.shape[1]), Xz.T@ytr)
    return ((Xte-mu)/sd)@w
def loso_r(X, y, s):
    preds = np.full(len(y), np.nan)
    for st in np.unique(s):
        te = s == st; tr = ~te
        mask_tr = np.zeros(len(y), bool); mask_tr[tr] = True
        ytr_c = site_demean(y, s, mask_tr)
        best_a, best_s = ALPHAS[0], -1
        rng = np.random.RandomState(0)
        inner = np.array_split(rng.permutation(np.where(tr)[0]), 3)
        for a in ALPHAS:
            sc = []
            for iv in inner:
                tr2 = np.setdiff1d(np.where(tr)[0], iv)
                m2 = np.zeros(len(y), bool); m2[tr2] = True
                ytr2_c = site_demean(y, s, m2)
                p = rfp(X[tr2], ytr2_c[tr2], X[iv], a)
                sc.append(np.corrcoef(p, y[iv])[0,1] if np.std(p) > 1e-12 else 0)
            sv = np.mean(sc)
            if sv > best_s: best_s, best_a = sv, a
        preds[te] = rfp(X[tr], ytr_c[tr], X[te], best_a)
    preds_c = preds.copy(); y_c = y.copy()
    for st in np.unique(s):
        m = s == st
        preds_c[m] -= preds_c[m].mean(); y_c[m] -= y_c[m].mean()
    return float(np.corrcoef(preds_c, y_c)[0,1])

TARGETS = {
    "Inatt": (keep_T, np.array([float(phmap.loc[ids_all[i], "Inattentive_n"]) for i in keep_T]), sites[keep_T]),
    "Hyper": (keep_T, np.array([float(phmap.loc[ids_all[i], "Hyper/Impulsive_n"]) for i in keep_T]), sites[keep_T]),
    "DX": (keep_DX, dx[keep_DX].astype(float), sites[keep_DX]),
}
REAL = {}
for tname, (keep, y0, s) in TARGETS.items():
    CONF = np.column_stack([age_i[keep], sex_all[keep], mot[keep]])
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(y0)), CONF]), y0, rcond=None)[0]
    yr = y0 - np.column_stack([np.ones(len(y0)), CONF]) @ coef
    REAL[tname] = {}
    for k, Xf in FEAT.items():
        X = Xf[keep]
        ok = np.isfinite(X).all(1) & np.isfinite(yr)
        if ok.sum() < 50: continue
        r = loso_r(X[ok], yr[ok], s[ok])
        REAL[tname][k] = {"cv_r": r, "n": int(ok.sum())}
        print(f"[{tname:5s}] {k:10s} LOSO CVr={r:+.3f} n={ok.sum()}")

print("\nfamilywise null (200)...", flush=True)
rng = np.random.RandomState(20260917)
maxstats = []
for p_ in range(200):
    mx = 0.0
    for tname, (keep, y0, s) in TARGETS.items():
        CONF = np.column_stack([age_i[keep], sex_all[keep], mot[keep]])
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(y0)), CONF]), y0, rcond=None)[0]
        yr0 = y0 - np.column_stack([np.ones(len(y0)), CONF]) @ coef
        yp = rng.permutation(yr0)
        for k, Xf in FEAT.items():
            X = Xf[keep]
            ok = np.isfinite(X).all(1)
            if ok.sum() < 50: continue
            r = loso_r(X[ok], yp[ok], s[ok])
            if abs(r) > abs(mx): mx = r
    maxstats.append(mx)
    if (p_+1) % 50 == 0: print(f"  perm {p_+1} ({time.time()-t0:.0f}s)", flush=True)
maxstats = np.array(maxstats)
print(f"null q95={np.percentile(np.abs(maxstats),95):.3f}")
for tname in REAL:
    for k in REAL[tname]:
        r = REAL[tname][k]["cv_r"]
        fw_p = float((np.sum(np.abs(maxstats) >= abs(r)) + 1) / 201)
        REAL[tname][k]["fw_p"] = fw_p
        verdict = "ALIVE" if (fw_p < 0.05 and abs(r) >= 0.15) else "dead"
        REAL[tname][k]["GATE2_VERDICT"] = verdict
        print(f"[{tname:5s}] {k:10s} CVr={r:+.3f} fw_p={fw_p:.4f} -> {verdict}")

with open(f"{OUT}/batch14_results.json", "w") as f:
    json.dump(REAL, f, indent=2)
print(f"DONE {time.time()-t0:.0f}s")
