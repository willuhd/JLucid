#!/usr/bin/env python3
"""exp/37 Batch 12 — final family batch: 7 theorist-proposed hybrid variants.
Gate 0 fresh for t4 variants (CGATE/EDIFF/DYNCTRL/EMAP split-half SB), then
Gate 0 re-verify + Gate 1 oracle + Gate 2 honest battery for all 7.
PREREG: results/SIEVE_TABLE.md batch-12 section (written before fit).
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.linalg import solve_discrete_lyapunov
from sklearn.covariance import LedoitWolf

BASE = Path("/Volumes/thinkplus/Code/JLucid")
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "results/37_leida"
SEED = 42
N_PERM = 200
ALPHAS = [0.1, 1, 10, 100, 1000, 3e3, 1e4, 3e4, 1e5]
NETS = ["VIS", "SOM", "DAN", "SAL", "LIM", "FPN", "DMN"]
LAM = 0.1
T0 = time.time() if (time := __import__('time')) else None

import time as _t
T0 = _t.time()
def log(m): print(f"[{_t.time()-T0:7.1f}s] {m}", flush=True)

from controllability.datasets_cc200 import load_cc200
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites_all = np.array(meta_all["sites"])
ages_all = np.array(meta_all["ages"], float)
Ts_all = [t.shape[0] for t in ts_all]
TR_BY_SITE = {1: 2.0, 3: 2.5, 4: 2.0, 5: 2.0, 6: 2.5, 7: 1.5, 8: 2.5}
yeo_idx = json.load(open(BASE / "src/leida/atlas/yeo_true_idx.json"))
UP = [(a, b) for a in range(7) for b in range(a, 7)]

# ---------------- cohort ----------------
ph = pd.read_csv(BASE / "data/adhd200/adhd200_preprocessed_phenotypics.tsv", sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
for c in ["Inattentive", "Hyper/Impulsive"]:
    ph[c + "_n"] = pd.to_numeric(ph[c], errors="coerce")
    ph.loc[ph[c + "_n"] == -999, c + "_n"] = np.nan
phmap = ph.set_index("ScanDir ID")[["Inattentive_n", "Hyper/Impulsive_n", "ADHD Measure"]]
keep = []
for i, sid in enumerate(ids_all):
    if sites_all[i] not in (3, 5, 6) or sid not in phmap.index: continue
    if str(phmap.loc[sid, "ADHD Measure"]) not in ("2", "3"): continue
    if pd.isna(phmap.loc[sid, "Inattentive_n"]): continue
    keep.append(i)
keep = np.array(keep)
Nsub = len(keep)
y_inatt = np.array([float(phmap.loc[ids_all[i], "Inattentive_n"]) for i in keep])
y_hyper = np.array([float(phmap.loc[ids_all[i], "Hyper/Impulsive_n"]) for i in keep])
sites_tc = sites_all[keep]
ages_tc = ages_all[keep]
log(f"T-cohort n={Nsub}")

# IQ cohort
v = pd.to_numeric(ph["Full4 IQ"], errors="coerce"); v[v == -999] = np.nan
ph["f4"] = v
phm2 = ph.set_index("ScanDir ID")
keep_iq = []
for i, sid in enumerate(ids_all):
    if sid not in phm2.index or pd.isna(phm2.loc[sid, "f4"]): continue
    keep_iq.append(i)
keep_iq = np.array(keep_iq)
y_iq = np.array([float(phm2.loc[ids_all[i], "f4"]) for i in keep_iq])
log(f"IQ cohort n={len(keep_iq)}")

# ---------------- machinery ----------------
def l2_theta(X, lam=LAM):
    Xz = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-12)
    lw = LedoitWolf().fit(Xz)
    return np.linalg.inv(lw.covariance_ + lam * np.eye(X.shape[1]))

def theta_to_A(Theta):
    Nn = Theta.shape[0]
    d = np.sqrt(np.abs(np.diag(Theta))); d[d == 0] = 1e-12
    P = -Theta / np.outer(d, d); np.fill_diagonal(P, 0)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    Araw = np.abs(P)
    lmax = float(np.max(np.linalg.eigvalsh((Araw + Araw.T) / 2)))
    A = Araw / (1.0 + lmax + 1e-12) - np.eye(Nn)
    return (A + A.T) / 2

def avg_modal(A):
    A = (A + A.T) / 2
    vals, vecs = np.linalg.eigh(A)
    return (vecs ** 2) @ (1.0 / (-2.0 * vals)), (vecs ** 2) @ (1.0 - np.exp(vals))

def morlet_kernel(TRv, freq=0.05, cycles=5, kernel_secs=60.0):
    sigma = cycles / (2 * np.pi * freq)
    L = max(int(6 * sigma / TRv), 3, int(3 * (1 / freq) / TRv))
    if L % 2 == 0: L += 1
    max_L = int(kernel_secs / TRv)
    if L > max_L: L = max_L if max_L % 2 == 1 else max_L - 1
    t = (np.arange(L) - L // 2) * TRv
    w = (np.pi ** -0.25) * np.exp(1j * 2 * np.pi * freq * t) * np.exp(-t ** 2 / (2 * sigma ** 2))
    return w / np.sqrt(np.sum(np.abs(w) ** 2) + 1e-12)

def wavelet_phase_ts(tseries, TRv, freq=0.05, cycles=5, kernel_secs=60.0):
    T, N = tseries.shape
    kern = morlet_kernel(TRv, freq, cycles, kernel_secs)
    Lk = len(kern)
    n_fft = 1
    while n_fft < T + Lk: n_fft <<= 1
    conv = np.fft.ifft(np.fft.fft(tseries, n=n_fft, axis=0) * np.fft.fft(kern, n=n_fft)[:, None], axis=0)
    st = Lk // 2
    return np.angle(conv[st:st + T, :])

def lam1_series(phases):
    c = np.cos(phases); s = np.sin(phases)
    a = (c * c).sum(1); b = (c * s).sum(1); d = (s * s).sum(1)
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4 * b * b, 0.0))
    return (a + d + disc) / 2 / phases.shape[1]

def fc28_of(z):
    netm = np.zeros((7, 7))
    for a in range(7):
        for b in range(a, 7):
            ia = yeo_idx[NETS[a]]; ib = yeo_idx[NETS[b]]
            blk = z[np.ix_(ia, ib)]
            if a == b:
                iu = np.triu_indices(len(ia), k=1)
                netm[a, b] = blk[iu].mean() if len(iu[0]) else 0.0
            else:
                netm[a, b] = blk.mean()
    return np.array([netm[a, b] for a, b in UP])

# ---------------- t4 variant feature builders (per subject) ----------------
def cgate_feats(t, TRv):
    ph = wavelet_phase_ts(t, TRv)
    g = lam1_series(ph)
    H = g >= np.median(g)
    out = {}
    for nm, msk in [("H", H), ("L", ~H)]:
        tsb = t[msk]
        fc = np.corrcoef(tsb.T); fc = np.nan_to_num(fc)
        z = np.arctanh(np.clip(fc, -0.999999, 0.999999))
        out["fc28_" + nm] = fc28_of(z)
        Th = l2_theta(tsb)
        d = np.sqrt(np.abs(np.diag(Th))); P = np.abs(Th) / np.outer(d, d); np.fill_diagonal(P, 0)
        out["p28_" + nm] = fc28_of(P)
        A = theta_to_A(Th)
        phi, mu = avg_modal(A)
        out["phi7_" + nm] = np.array([phi[yeo_idx[NETS[k]]].mean() for k in range(7)])
        out["mu7_" + nm] = np.array([mu[yeo_idx[NETS[k]]].mean() for k in range(7)])
    x = np.concatenate([out["fc28_H"] - out["fc28_L"], out["p28_H"] - out["p28_L"],
                        out["phi7_H"] - out["phi7_L"], out["mu7_H"] - out["mu7_L"]])
    return x  # 70-dim

def ediff_feats(t, D28, xD):
    fc = np.corrcoef(t.T); fc = np.nan_to_num(fc)
    z = np.arctanh(np.clip(fc, -0.999999, 0.999999))
    z28 = fc28_of(z)
    f_load = float(z28 @ D28)
    Th = l2_theta(t)
    A = theta_to_A(Th)
    lam_max = float(np.max(np.abs(np.linalg.eigvalsh(A))))
    if lam_max >= 1.0: A = A / (lam_max + 1e-6) * 0.99
    W = solve_discrete_lyapunov(A, np.eye(A.shape[0]))
    Winv = np.linalg.inv(W)
    E_D = float(xD @ Winv @ xD)
    Ei = np.diag(Winv)
    e_rel = np.log(E_D + 1e-12) - np.mean(np.log(np.abs(Ei) + 1e-12))
    return np.array([f_load, np.log(E_D + 1e-12), e_rel])

def dynctrl_feats(t):
    T = t.shape[0]
    W = 30; stride = 10
    n_w = (T - W) // stride + 1
    a_net = np.zeros((n_w, 7))
    for wi in range(n_w):
        seg = t[wi * stride: wi * stride + W]
        Cw = np.corrcoef(seg.T); Cw = np.nan_to_num(Cw)
        Tw_ = np.linalg.inv(Cw + 0.1 * np.eye(190))
        d = np.sqrt(np.abs(np.diag(Tw_))); P = np.abs(Tw_) / np.outer(d, d); np.fill_diagonal(P, 0)
        s_row = P.mean(1)
        for k in range(7):
            a_net[wi, k] = s_row[yeo_idx[NETS[k]]].mean()
    mean7 = a_net.mean(0)
    cv7 = a_net.std(0) / (np.abs(a_net.mean(0)) + 1e-12)
    glob = a_net.mean(1)
    if len(glob) >= 3:
        persist1 = np.corrcoef(glob[:-1], glob[1:])[0, 1] if np.std(glob[:-1]) > 1e-12 and np.std(glob[1:]) > 1e-12 else 0.0
    else:
        persist1 = 0.0
    return np.concatenate([mean7, cv7, [persist1]])  # 15-dim

def emap_feats(t):
    Th = l2_theta(t)
    A = theta_to_A(Th)
    lam_max = float(np.max(np.abs(np.linalg.eigvalsh(A))))
    if lam_max >= 1.0: A = A / (lam_max + 1e-6) * 0.99
    W = solve_discrete_lyapunov(A, np.eye(A.shape[0]))
    Winv = np.linalg.inv(W)
    Ei = np.diag(Winv)
    self7 = np.array([np.mean(np.log(np.abs(Ei[yeo_idx[NETS[k]]]) + 1e-12)) for k in range(7)])
    tgt7 = []
    for k in range(7):
        xk = np.zeros(190); xk[yeo_idx[NETS[k]]] = 1.0
        xk /= np.linalg.norm(xk)
        tgt7.append(np.log(float(xk @ Winv @ xk) + 1e-12))
    xcon4 = []
    for (a, b) in [(2, 6), (3, 6), (3, 5), (5, 6)]:  # (DAN,DMN),(SAL,DMN),(SAL,FPN),(FPN,DMN)
        ua = np.zeros(190); ua[yeo_idx[NETS[a]]] = 1.0; ua /= np.linalg.norm(ua)
        ub = np.zeros(190); ub[yeo_idx[NETS[b]]] = 1.0; ub /= np.linalg.norm(ub)
        x = (ua - ub) / np.linalg.norm(ua - ub)
        xcon4.append(np.log(float(x @ Winv @ x) + 1e-12))
    return np.concatenate([self7, tgt7, xcon4])  # 18-dim

# ---------------- build all features (full run + halves for Gate 0) ----------------
log("building t4 features full+halves (this is the heavy pass) ...")
Z = np.load(BASE / "results/pilot_geometry/pilot7_d5_raw.npz")
CSF = Z["CSF"]; DDF = Z["DDF"]; occF = Z["occF"]; agesF = Z["ages"]

# EDIFF needs the ADHD-HC difference template D28 (in-sample for oracle; fold-local at Gate 2 per prereg)
F28_all = np.load(BASE / "results/F_STATIC.npy")[:, 0:28]
dx_all = labels_all
site_mean = {}
for s in np.unique(sites_all):
    site_mean[s] = F28_all[sites_all == s].mean(0)
F28c = F28_all - np.array([site_mean[s] for s in sites_all])
D28 = F28c[dx_all == 1].mean(0) - F28c[dx_all == 0].mean(0)
D28n = D28 / (np.linalg.norm(D28) + 1e-12)
xD_node = np.zeros(190)
for a in range(7):
    for b in range(7):
        if a <= b:
            pass
# node-space xD: row-sum of block-constant template
net_of_node = np.zeros(190, int)
for k, net in enumerate(NETS):
    for i in yeo_idx[net]: net_of_node[i] = k
xD = np.zeros(190)
for a in range(7):
    for b in range(7):
        if a <= b:
            k = UP.index((a, b))
            ia = np.array(yeo_idx[NETS[a]]); ib = np.array(yeo_idx[NETS[b]])
            xD[ia] += D28n[k]
            if a != b:
                xD[ib] += D28n[k]
xD /= np.linalg.norm(xD) + 1e-12

feats = {}
for name in ["CGATE", "EDIFF", "DYNCTRL", "EMAP"]:
    feats[name] = {"full": [], "h1": [], "h2": []}

for i in range(len(ts_all)):
    t = ts_all[i]; TRv = TR_BY_SITE[int(sites_all[i])]
    h = t.shape[0] // 2
    # CGATE on full + halves
    for tag, tt in [("full", t), ("h1", t[:h]), ("h2", t[h:])]:
        feats["CGATE"][tag].append(cgate_feats(tt, TRv))
        feats["EDIFF"][tag].append(ediff_feats(tt, D28n, xD))
        feats["EMAP"][tag].append(emap_feats(tt))
        if tag == "full":
            feats["DYNCTRL"][tag].append(dynctrl_feats(t))
        else:
            feats["DYNCTRL"][tag].append(dynctrl_feats(tt))
    if i % 100 == 0: log(f"  subj {i}/872")

np.savez(OUT / "batch12_feats.npz",
         **{f"{nm}_{tag}": np.array(feats[nm][tag]) for nm in feats for tag in feats[nm]},
         D28n=D28n, xD=xD)
log("t4 features built and saved")

# ---------------- Gate 0: split-half SB ----------------
log("Gate 0 (split-half SB) ...")
Ts_arr = np.array(Ts_all)
mT = Ts_arr >= 60
gate0 = {}
for nm in ["CGATE", "EDIFF", "DYNCTRL", "EMAP"]:
    X1 = np.array(feats[nm]["h1"])[mT]; X2 = np.array(feats[nm]["h2"])[mT]
    rhos = []
    for j in range(X1.shape[1]):
        if np.std(X1[:, j]) < 1e-12 or np.std(X2[:, j]) < 1e-12: rhos.append(0.0); continue
        r = np.corrcoef(X1[:, j], X2[:, j])[0, 1]
        rhos.append(2 * r / (1 + r) if r > -1 else 0.0)
    gate0[nm] = {"mean_SB": float(np.mean(rhos)), "min_SB": float(np.min(rhos)),
                 "frac_ge_030": float(np.mean(np.array(rhos) >= 0.30)),
                 "per_comp_SB": [round(float(v), 3) for v in rhos]}
    print(f"  {nm}: mean SB={gate0[nm]['mean_SB']:.3f} min={gate0[nm]['min_SB']:.3f} frac>=0.30={gate0[nm]['frac_ge_030']:.2f}")
json.dump(gate0, open(OUT / "batch12_gate0.json", "w"), indent=1)
DEAD0 = [nm for nm in gate0 if gate0[nm]["frac_ge_030"] < 0.5]
log(f"Gate 0 DEAD: {DEAD0}")
