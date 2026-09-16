#!/usr/bin/env python3
"""RECOVERED (transcribed) Penn builder /tmp/audit_b.py -> batch15c_penn_dispz.npz (+B3=item 5).

Source: fork-session testimony Round 4 2026-08-30 (turn after SyntaxError turn).
Adapted ONLY: BASE JLucid2->JLucid, OUT exp/37_leida_sieve/gate1_oracle->results,
src-pennlead->src/pennlead. SAVE REDIRECTED to exp_overnight_leida/00_recovered/
(never overwrite results/). LOGIC IDENTICAL: Morlet 0.05Hz/5cyc/60s cap, 7-net
phases, W1/H3v/Lnet/K1 blocks, non-ADHD normative (min 20), MAD+clip3,
comb=H3v+K1+Lnet_r, B3 mean-FD+age+sex lstsq (seed 31, 2000 perms),
B1 global-median low-motion (seed 32), B2 ESWAN residual-r.
Status: transcribed-pending-verification.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
VERIFY_OUT = f"{BASE}/exp_overnight_leida/00_recovered/verify_batch15c_penn_dispz.npz"
sys.path.insert(0, f"{BASE}/src")
from pennlead.datasets import load_pennlead
from nilearn.datasets import fetch_atlas_schaefer_2018
TR = 0.8
def kernel_len(TR, f, cycles, cap):
    sigma = cycles / (2*np.pi*f)
    L = int(6*sigma/TR); L3 = int(3.0/f/TR); L = max(L, L3)
    if L % 2 == 0: L += 1
    Lcap = int(cap/TR); Lcap = Lcap if Lcap % 2 == 1 else Lcap - 1
    if L > Lcap: L = Lcap
    return L
def morlet(TR, f, cycles, cap):
    L = kernel_len(TR, f, cycles, cap)
    t = (np.arange(L) - L//2) * TR
    w = np.pi**-0.25*np.exp(1j*2*np.pi*f*t)*np.exp(-t**2/(2*(cycles/(2*np.pi*f))**2))
    return w/np.sqrt(np.sum(np.abs(w)**2)+1e-12), L
def phases_of(ts, w, L):
    T, N = ts.shape
    n_fft = 1
    while n_fft < T + L: n_fft <<= 1
    return np.angle(np.fft.ifft(np.fft.fft(ts, n=n_fft, axis=0)*np.fft.fft(w, n=n_fft)[:, None], axis=0)[L//2:L//2+T])
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

def build_blocks(cond):
    pts, plab, pmeta = load_pennlead(condition=cond, qc_fd_thresh=0.5)
    pids = [str(x) for x in pmeta["participant_ids"]]
    sch = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, data_dir=f"{BASE}/data/atlas/schaefer_2018/nilearn")
    lab_names = [l.decode() if isinstance(l, bytes) else str(l) for l in sch.labels]
    net_code = {"Vis":0,"SomMot":1,"DorsAttn":2,"SalVentAttn":3,"Limbic":4,"Cont":5,"Default":6}
    yeo = np.full(pmeta["N"], -1)
    for i in range(min(400, pmeta["N"])):
        l = lab_names[i+1] if i+1 < len(lab_names) else ""
        for nm, code in net_code.items():
            if f"_{nm}_" in l: yeo[i] = code; break
    net_idx = [np.where(yeo==k)[0] for k in range(7)]
    w, L = morlet(TR, 0.05, 5, 60.0)
    n = len(pts)
    H3v = np.full((n, 8), np.nan); K1 = np.full((n, 14), np.nan)
    Lnet = np.full((n, 28), np.nan); W1 = np.full((n, 28), np.nan)
    UP = [(a,b) for a in range(7) for b in range(a,7)]
    for i, ts in enumerate(pts):
        T = ts.shape[0]
        if T < L + 20: continue
        ph = phases_of(ts, w, L)
        TH = np.zeros((T, 7))
        for k in range(7):
            z = np.exp(1j*ph[:, net_idx[k]]).sum(axis=1)
            TH[:, k] = np.angle(z)
        for kk, (a_, b_) in enumerate(UP):
            W1[i, kk] = wrap(TH[:, a_]-TH[:, b_]).var()
        U = np.zeros((T, 7))
        for t in range(T):
            cc = np.cos(TH[t, :, None]-TH[t, None, :])
            evv, evec = np.linalg.eigh(cc)
            u = evec[:, -1]
            if u.sum() > 0: u = -u
            U[t] = u
        var7 = U.var(axis=0)
        H3v[i] = np.concatenate([[var7.mean()], var7])
        Lnet[i] = ((U.T@U)/T)[np.triu_indices(7)]
        cort = np.concatenate(net_idx)
        Xz = (ts[:, cort]-ts[:, cort].mean(0))/(ts[:, cort].std(0)+1e-12)
        FC = np.nan_to_num(np.corrcoef(Xz.T))
        lam = np.linalg.eigvalsh((FC+FC.T)/2)
        A_s = FC/(1.0+float(lam.max())+1e-12) - np.eye(FC.shape[0])
        ev = np.linalg.eigvalsh((A_s+A_s.T)/2)
        if ev.max() >= 0: A_s = A_s - (ev.max()+1e-6)*np.eye(FC.shape[0])
        vals, vecs = np.linalg.eigh((A_s+A_s.T)/2)
        phi = (vecs**2)@(1.0/(-2.0*vals)); mu = (vecs**2)@(1.0-np.exp(vals))
        masks = [[j for j in range(len(cort)) if yeo[cort[j]]==k] for k in range(7)]
        K1[i] = np.concatenate([[phi[mk].mean() for mk in masks], [mu[mk].mean() for mk in masks]])
    return pts, pmeta, pids, {"H3v": H3v, "K1": K1, "Lnet_r": Lnet, "W1_r": W1}

cohort = pd.read_csv(f"{BASE}/data/pennlead/pheno/cohort.csv")
cohort["participant_id"] = cohort["participant_id"].astype(str)
cm = cohort.set_index("participant_id")

R = {}
for cond in ["rest", "nback"]:
    pts, pmeta, pids, B = build_blocks(cond)
    dxp = np.array([int(cm.loc[p, "dx_adhd_i"]) if p in cm.index and pd.notna(cm.loc[p, "dx_adhd_i"]) else np.nan for p in pids])
    ages = np.array(pmeta["ages"], dtype=float)
    sexes = np.array([1.0 if str(x).upper().startswith("M") else 0.0 for x in pmeta["sexes"]])
    fds = np.array(pmeta["fd_means"], dtype=float)
    valid = np.isfinite(dxp); na = valid & (dxp == 0)
    def norm_z_pl(X):
        n, d = X.shape
        Z = np.full_like(X, np.nan)
        A = np.column_stack([np.ones(na.sum()), ages[na], sexes[na]])
        for j in range(d):
            y = X[na, j]; ok = np.isfinite(y)
            if ok.sum() < 20: continue
            coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
            r = y[ok] - A[ok] @ coef
            mad = 1.4826*np.median(np.abs(r-np.median(r))) + 1e-12
            Aall = np.column_stack([np.ones(n), ages, sexes])
            Z[:, j] = (X[:, j] - Aall @ coef)/mad
        return np.clip(Z, -3, 3)
    Zs = {k: norm_z_pl(X) for k, X in B.items()}
    mz2 = {k: np.nanmean(Z**2, axis=1) for k, Z in Zs.items()}
    comb = np.nanmean(np.column_stack([mz2["H3v"], mz2["K1"], mz2["Lnet_r"]]), axis=1)
    R[cond] = dict(pids=pids, dx=dxp, ages=ages, sexes=sexes, fds=fds, mz2=mz2, comb=comb)

pids_r = R["rest"]["pids"]; pids_n = R["nback"]["pids"]
common = [p for p in pids_r if p in pids_n]
ir = [pids_r.index(p) for p in common]; inn = [pids_n.index(p) for p in common]
dcomb = R["nback"]["comb"][inn] - R["rest"]["comb"][ir]
dx_c = np.array([int(cm.loc[p, "dx_adhd_i"]) if p in cm.index and pd.notna(cm.loc[p, "dx_adhd_i"]) else np.nan for p in common])
fd_n = R["nback"]["fds"][inn]; fd_r = R["rest"]["fds"][ir]
ages_c = R["rest"]["ages"][ir]; sexes_c = R["rest"]["sexes"][ir]
m = np.isfinite(dcomb) & np.isfinite(dx_c)
obs = dcomb[m & (dx_c==1)].mean() - dcomb[m & (dx_c==0)].mean()
B_ = np.column_stack([np.ones(m.sum()), (fd_n+fd_r)[m]/2, ages_c[m], sexes_c[m]])
coef, *_ = np.linalg.lstsq(B_, dcomb[m], rcond=None)
resid = dcomb[m] - B_ @ coef
obs_r = resid[dx_c[m]==1].mean() - resid[dx_c[m]==0].mean()
rng = np.random.RandomState(31)
nulls = []
for _ in range(2000):
    yp = rng.permutation(dx_c[m])
    nulls.append(resid[yp==1].mean() - resid[yp==0].mean())
nulls = np.array(nulls)
p = (np.sum(np.abs(nulls) >= abs(obs_r))+1)/2001
p1 = (np.sum(nulls >= obs_r)+1)/2001
print(f"B3 WITHIN-SUBJECT rest->nback dispersion change (ADHD - non-ADHD):")
print(f"  raw obs={obs:+.4f}; conf-resid obs={obs_r:+.4f} two-sided p={p:.4f} one-sided p={p1:.4f} (nADHD={int(np.sum(m&(dx_c==1)))}, nHC={int(np.sum(m&(dx_c==0)))})")

fdn = R["nback"]["fds"]; valid_n = np.isfinite(R["nback"]["dx"])
comb_n = R["nback"]["comb"]
med_fd = np.median(fdn[valid_n])
lowm = valid_n & (fdn <= med_fd) & np.isfinite(comb_n)
dxn = R["nback"]["dx"]
obs_lo = comb_n[lowm & (dxn==1)].mean() - comb_n[lowm & (dxn==0)].mean()
rng = np.random.RandomState(32)
nl = []
for _ in range(2000):
    yp = rng.permutation(dxn[lowm])
    nl.append(comb_n[lowm][yp==1].mean() - comb_n[lowm][yp==0].mean())
nl = np.array(nl)
p_lo = (np.sum(nl >= obs_lo)+1)/2001
print(f"\nB1 nback LOW-MOTION half: obs={obs_lo:+.4f} one-sided p={p_lo:.4f} (nADHD={int(np.sum(lowm&(dxn==1)))}, nHC={int(np.sum(lowm&(dxn==0)))})")

np.savez_compressed(VERIFY_OUT,
                    pids_rest=np.array(R["rest"]["pids"]), pids_nback=np.array(R["nback"]["pids"]),
                    comb_rest=R["rest"]["comb"], comb_nback=R["nback"]["comb"],
                    dx_rest=R["rest"]["dx"], dx_nback=R["nback"]["dx"],
                    **{f"mz2_rest_{k}": v for k, v in R["rest"]["mz2"].items()},
                    **{f"mz2_nback_{k}": v for k, v in R["nback"]["mz2"].items()})
print("\nsaved VERIFY_OUT (exp folder; results/ untouched)")
z = np.load(f"{OUT}/batch15c_penn_dispz.npz")
for k in ["comb_rest", "comb_nback"]:
    a = np.load(VERIFY_OUT)[k]; b = z[k]
    print(f"VERIFY {k}: max|regen-saved| = {float(np.nanmax(np.abs(a-b))):.6f}")
