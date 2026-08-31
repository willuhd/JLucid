#!/usr/bin/env python3
"""
exp/37 BATCH 4 — PennLEAD INDEPENDENT DIRECTIONAL CONFIRMATION of W1 (preregistered).
Prereg: SIEVE_TABLE.md Batch-4 section, written BEFORE this run. NOT a screen — an
out-of-sample test on data never touched by the sieve.

Feature: W1 = Var_t[wrap(Θ_a − Θ_b)] over 28 Yeo network pairs, computed on PennLEAD
rest timeseries at PennLEAD TR=0.8s, f=0.05 Hz, 5 cycles, cap 60s (L=75 TRs).
Cortical 400 parcels (Schaefer400 Yeo-mapped); subcortical excluded from networks.
Targets: ESWAN adhd inattention total (primary), ESWAN total (secondary).
Stats: per-component Pearson+Spearman r (28 tests, Holm), sign-consistency vs the
ADHD-200 alpha=1000 full-refit loadings (w1_inatt_loadings_a1000.npy; binomial sign test
on comps with |loading| > 0.5*max|loading|), and the prereg composite test.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats as st

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/37_leida"
sys.path.insert(0, f"{BASE}/src")
from pennlead.datasets import load_pennlead

TR = 0.8
FREQ = 0.05
CYCLES = 5
CAP = 60.0
NETS = ['VIS', 'SOM', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']
UP = [(a, b) for a in range(7) for b in range(a, 7)]

def kernel_len(TR, f, cycles, cap):
    sigma = cycles / (2 * np.pi * f)
    L = int(6 * sigma / TR); L3 = int(3.0 / f / TR); L = max(L, L3)
    if L % 2 == 0: L += 1
    Lcap = int(cap / TR); Lcap = Lcap if Lcap % 2 == 1 else Lcap - 1
    if L > Lcap: L = Lcap
    return L
def morlet(TR, f, cycles, cap):
    L = kernel_len(TR, f, cycles, cap)
    t = (np.arange(L) - L // 2) * TR
    w = np.pi ** -0.25 * np.exp(1j * 2 * np.pi * f * t) * np.exp(-t ** 2 / (2 * (cycles / (2 * np.pi * f)) ** 2))
    return w / np.sqrt(np.sum(np.abs(w) ** 2) + 1e-12), L
def phases_of(ts, w, L):
    T, N = ts.shape
    n_fft = 1
    while n_fft < T + L: n_fft <<= 1
    return np.angle(np.fft.ifft(np.fft.fft(ts, n=n_fft, axis=0) * np.fft.fft(w, n=n_fft)[:, None], axis=0)[L // 2:L // 2 + T])
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

# Yeo mapping for PennLEAD (exp/23 mapping)
from nilearn.datasets import fetch_atlas_schaefer_2018
sch = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, data_dir=os.path.join(BASE, "data/atlas/schaefer_2018/nilearn"))
lab_names = [l.decode() if isinstance(l, bytes) else str(l) for l in sch.labels]
net_code = {"Vis": 0, "SomMot": 1, "DorsAttn": 2, "SalVentAttn": 3, "Limbic": 4, "Cont": 5, "Default": 6}

pts, plab, pmeta = load_pennlead(condition="rest", qc_fd_thresh=0.5)
Np = len(pts); NROI_P = pmeta["N"]
print(f"PennLEAD n={Np} N_rois={NROI_P}")
yeo_penn = np.full(NROI_P, -1)
for i in range(min(400, NROI_P)):
    l = lab_names[i + 1] if i + 1 < len(lab_names) else ""
    for nm, code in net_code.items():
        if f"_{nm}_" in l:
            yeo_penn[i] = code; break
n_cort = int((yeo_penn[:400] >= 0).sum())
print(f"cortical mapped: {n_cort}/400")

w, L = morlet(TR, FREQ, CYCLES, CAP)
print(f"kernel L={L} TRs = {L*TR:.1f}s")
W1 = np.full((Np, 28), np.nan)
for i, ts in enumerate(pts):
    T = ts.shape[0]
    if T < 60: continue
    ph = phases_of(ts, w, L)
    TH = np.zeros((T, 7))
    for k in range(7):
        idx = np.where(yeo_penn == k)[0]
        z = np.exp(1j * ph[:, idx]).sum(axis=1)
        TH[:, k] = np.angle(z)
    for kk, (a, b) in enumerate(UP):
        d = wrap(TH[:, a] - TH[:, b])
        W1[i, kk] = d.var(axis=0)
print("W1 built")

# targets
eswan_inatt = np.array(pmeta.get("eswan_scores", np.full(Np, np.nan))) if "eswan_scores" in pmeta else None
# loader puts eswan_total in meta; we need inattention specifically. Re-read cohort:
cohort = pd.read_csv(os.path.join(BASE, "data/pennlead/pheno/cohort.csv"))
cohort["participant_id"] = cohort["participant_id"].astype(str)
pids = pmeta["participant_ids"]
inatt_map = dict(zip(cohort["participant_id"], pd.to_numeric(cohort["eswan_adhd_inattention_total"], errors="coerce")))
total_map = dict(zip(cohort["participant_id"], pd.to_numeric(cohort["eswan_adhd_total_score"], errors="coerce")))
y_inatt = np.array([inatt_map.get(p, np.nan) for p in pids])
y_total = np.array([total_map.get(p, np.nan) for p in pids])
print(f"ESWAN inatt available: {np.isfinite(y_inatt).sum()}, total: {np.isfinite(y_total).sum()}")

# ADHD-200 loadings (direction from the discovery cohort)
w_adhd = np.load(f"{OUT}/w1_inatt_loadings_a1000.npy")
w_norm = w_adhd / (np.abs(w_adhd).max() + 1e-12)
big = np.abs(w_norm) > 0.5
print(f"components with |loading|>0.5: {big.sum()}")

results = {}
for tname, y in [("inatt", y_inatt), ("total", y_total)]:
    m = np.isfinite(y) & np.all(np.isfinite(W1), axis=1)
    Xs, ys = W1[m], y[m]
    comps = []
    for j in range(28):
        r_p = float(np.corrcoef(Xs[:, j], ys)[0, 1])
        r_s = float(st.spearmanr(Xs[:, j], ys)[0])
        a, b = UP[j]
        comps.append({"pair": f"{NETS[a]}-{NETS[b]}", "pearson": r_p, "spearman": r_s,
                      "adhd200_loading": float(w_adhd[j])})
    # Holm on pearson
    # two-sided t p
    pvals = []
    for c in comps:
        r = c["pearson"]; n = int(m.sum())
        t = r * np.sqrt((n-2)/(1-r*r+1e-15))
        pvals.append(2 * st.t.sf(abs(t), n-2))
    pvals = np.array(pvals)
    order = np.argsort(pvals)
    holm = np.empty(28); mrun = 0.0
    for rank, j in enumerate(order):
        holm[j] = min(1.0, max(mrun, (28 - rank) * pvals[j])); mrun = holm[j]
    for j in range(28):
        comps[j]["p_raw"] = float(pvals[j]); comps[j]["p_holm"] = float(holm[j])
    # sign consistency on big-loading comps
    big_idx = np.where(big)[0]
    same = sum(1 for j in big_idx if np.sign(comps[j]["pearson"]) == np.sign(w_adhd[j]))
    n_big = len(big_idx)
    sign_p = float(st.binomtest(same, n_big, 0.5).pvalue) if n_big > 0 else 1.0
    results[tname] = {"n": int(m.sum()), "comps": comps, "sign_same": int(same), "n_big": n_big,
                      "sign_p": sign_p}
    sig = [c for c in comps if c["p_holm"] < 0.05]
    print(f"\n[{tname}] n={m.sum()}  sign-consistency: {same}/{n_big} same sign (p={sign_p:.4f})")
    print("  Holm-significant components:")
    if sig:
        for c in sig:
            print(f"    {c['pair']:14s} r={c['pearson']:+.3f} p_holm={c['p_holm']:.4f} adhd200_w={c['adhd200_loading']:+.3f}")
    else:
        print("    none")
    top = sorted(comps, key=lambda c: abs(c["pearson"]), reverse=True)[:5]
    print("  top-5 |r|:")
    for c in top:
        print(f"    {c['pair']:14s} r={c['pearson']:+.3f} (sp {c['spearman']:+.3f}) p_holm={c['p_holm']:.3f} adhd200_w={c['adhd200_loading']:+.3f}")

with open(f"{OUT}/penn_confirm_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved penn_confirm_results.json")
