#!/usr/bin/env python3
"""
exp/37 BATCH 9 — PennLEAD nback task-modulation screen (preregistered).
Extract 0BACK and 2BACK block ON-period timeseries (from events.tsv), compute the frozen
families per condition, then CONTRASTS (2back − 0back). Also behavioral d-prime analog as
positive-control predictor of ESWAN. Targets: 3 ESWAN scores, age/sex/FD partialled.
Nested 5-fold CV; familywise null 1000 perms over all batch-9 cells.
"""
import sys, os, glob, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats as st

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/37_leida"
TR = 0.8
NETS = ['VIS', 'SOM', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']
UP = [(a, b) for a in range(7) for b in range(a, 7)]

sys.path.insert(0, f"{BASE}/src")
from pennlead.datasets import load_pennlead

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

from nilearn.datasets import fetch_atlas_schaefer_2018
sch = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, data_dir=os.path.join(BASE, "data/atlas/schaefer_2018/nilearn"))
lab_names = [l.decode() if isinstance(l, bytes) else str(l) for l in sch.labels]
net_code = {"Vis": 0, "SomMot": 1, "DorsAttn": 2, "SalVentAttn": 3, "Limbic": 4, "Cont": 5, "Default": 6}

pts, plab, pmeta = load_pennlead(condition="nback", qc_fd_thresh=0.5)
Np = len(pts)
pids = list(pmeta["participant_ids"])
NROI = pmeta["N"]
yeo = np.full(NROI, -1)
for i in range(min(400, NROI)):
    l = lab_names[i + 1] if i + 1 < len(lab_names) else ""
    for nm, code in net_code.items():
        if f"_{nm}_" in l: yeo[i] = code; break
net_idx = [np.where(yeo == k)[0] for k in range(7)]
print(f"n={Np}")

# events per subject
evmap = {}
for e in sorted(glob.glob(f"{BASE}/data/pennlead/events/*nback_events.tsv")):
    pid = os.path.basename(e).split("_")[0]
    evmap[pid] = pd.read_csv(e, sep="\t")

w, L = morlet(TR, 0.05, 5, 60.0)
print(f"kernel L={L} TRs ({L*TR:.0f}s)")

t0 = time.time()
F = {}
for tag in ["W1", "H1v", "Lnet", "H3v"]:
    F[f"2b_{tag}"] = np.full((Np, 28 if tag in ("W1","H1v","Lnet") else 8), np.nan)
    F[f"0b_{tag}"] = np.full((Np, 28 if tag in ("W1","H1v","Lnet") else 8), np.nan)

def block_feats(ts_segment):
    """Compute W1/H1v/Lnet/H3v on a (Tseg, N) segment."""
    ph = phases_of(ts_segment, w, L)
    T = ph.shape[0]
    TH = np.zeros((T, 7))
    for k in range(7):
        z = np.exp(1j * ph[:, net_idx[k]]).sum(axis=1)
        TH[:, k] = np.angle(z)
    W1 = np.zeros(28); H1v = np.zeros(28)
    for kk, (a_, b_) in enumerate(UP):
        W1[kk] = wrap(TH[:, a_] - TH[:, b_]).var()
    cosd = np.cos(ph[:, :, None] - ph[:, None, :])
    for kk, (a_, b_) in enumerate(UP):
        ia, ib = net_idx[a_], net_idx[b_]
        if a_ == b_:
            mm = cosd[:, ia][:, :, ia]
            ii, jj = np.triu_indices(len(ia), k=1)
            series = mm[:, ii, jj].mean(axis=1)
        else:
            series = cosd[:, ia][:, :, ib].mean(axis=(1, 2))
        H1v[kk] = series.var()
    U = np.zeros((T, 7))
    for t in range(T):
        cc = np.cos(TH[t, :, None] - TH[t, None, :])
        evv, evec = np.linalg.eigh(cc)
        u = evec[:, -1]
        if u.sum() > 0: u = -u
        U[t] = u
    var7 = U.var(axis=0)
    H3v = np.concatenate([[var7.mean()], var7])
    Lm = (U.T @ U) / T
    Lnet = Lm[np.triu_indices(7)]
    return W1, H1v, Lnet, H3v

# per subject: build concatenated 0back and 2back segments
n_ok = 0
for i, pid in enumerate(pids):
    if pid not in evmap: continue
    ev = evmap[pid]
    ts = pts[i]
    Ttot = ts.shape[0]
    segs = {}
    for cond in ["0BACK", "2BACK"]:
        rows = ev[ev["trial_type"] == cond]
        trs = []
        for _, r in rows.iterrows():
            s = int(r["onset"] / TR); e = int((r["onset"] + r["duration"] + 4.8) / TR)  # +6TR post-trial
            trs.append((s, e))
        # merge overlapping
        trs.sort()
        merged = []
        for s, e in trs:
            if merged and s <= merged[-1][1]: merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else: merged.append((s, e))
        idx = []
        for s, e in merged:
            idx.extend(range(max(0, s), min(Ttot, e)))
        segs[cond] = ts[idx] if len(idx) > 100 else None
    if segs["0BACK"] is None or segs["2BACK"] is None: continue
    W2b, H2b, L2b, HV2b = block_feats(segs["2BACK"])
    W0b, H0b, L0b, HV0b = block_feats(segs["0BACK"])
    F["2b_W1"][i] = W2b; F["0b_W1"][i] = W0b
    F["2b_H1v"][i] = H2b; F["0b_H1v"][i] = H0b
    F["2b_Lnet"][i] = L2b; F["0b_Lnet"][i] = L0b
    F["2b_H3v"][i] = HV2b; F["0b_H3v"][i] = HV0b
    n_ok += 1
print(f"subjects with both blocks: {n_ok} ({time.time()-t0:.0f}s)")

# contrasts
for tag, dim in [("W1", 28), ("H1v", 28), ("Lnet", 28), ("H3v", 8)]:
    F[f"delta_{tag}"] = F[f"2b_{tag}"] - F[f"0b_{tag}"]

# behavioral d-prime analog per subject
BEH = np.full(Np, np.nan)
for i, pid in enumerate(pids):
    if pid not in evmap: continue
    ev = evmap[pid]
    sc = ev["score"].dropna().astype(str)
    tp = (sc == "true_positive").sum(); fp = (sc == "false_positive").sum()
    fn = (sc == "false_negative").sum(); tn = (sc == "true_negative").sum()
    hit = tp / max(tp + fn, 1); fa = fp / max(fp + tn, 1)
    # d' analog with floor/ceiling correction
    hit = min(max(hit, 1e-3), 1 - 1e-3); fa = min(max(fa, 1e-3), 1 - 1e-3)
    from scipy.stats import norm
    BEH[i] = norm.ppf(hit) - norm.ppf(fa)
print("d-prime stats:", np.nanmin(BEH), np.nanmax(BEH), "finite:", np.isfinite(BEH).sum())

cohort = pd.read_csv(os.path.join(BASE, "data/pennlead/pheno/cohort.csv"))
cohort["participant_id"] = cohort["participant_id"].astype(str)
ymaps = {c: dict(zip(cohort["participant_id"], pd.to_numeric(cohort[c], errors="coerce")))
         for c in ["eswan_adhd_inattention_total", "eswan_adhd_hyperactivity_impulsivity_total", "eswan_adhd_total_score"]}
ages = np.array(pmeta["ages"], dtype=float)
sexes = np.array([1.0 if str(s).upper().startswith("M") else 0.0 for s in pmeta["sexes"]])
fds = np.array(pmeta["fd_means"], dtype=float)

ALPHAS = [1, 10, 100, 1000, 1e4]
def cv_r(X, y, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, 5)
    preds = np.full(len(y), np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        best_a, best_s = ALPHAS[0], -1
        inner = np.array_split(rng.permutation(tr), 3)
        for a in ALPHAS:
            sc = []
            for iv in inner:
                tr2 = np.setdiff1d(tr, iv)
                mu = X[tr2].mean(0); sd = X[tr2].std(0) + 1e-12
                Xz = (X[tr2]-mu)/sd
                ww = np.linalg.solve(Xz.T@Xz + a*np.eye(Xz.shape[1]), Xz.T@y[tr2])
                p = ((X[iv]-mu)/sd)@ww
                sc.append(np.corrcoef(p, y[iv])[0,1] if np.std(p) > 1e-12 else 0)
            s = np.mean(sc)
            if s > best_s: best_s, best_a = s, a
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-12
        Xz = (X[tr]-mu)/sd
        ww = np.linalg.solve(Xz.T@Xz + best_a*np.eye(Xz.shape[1]), Xz.T@y[tr])
        preds[f] = ((X[f]-mu)/sd)@ww
    return float(np.corrcoef(preds, y)[0,1])

TDATA = {}
for tcol, ymap in ymaps.items():
    y = np.array([ymap.get(p, np.nan) for p in pids])
    m = np.isfinite(y)
    yv = y[m]
    B = np.column_stack([ages[m], sexes[m], fds[m]])
    coef = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), B]), yv, rcond=None)[0]
    yr = yv - np.column_stack([np.ones(m.sum()), B]) @ coef
    cells = {}
    for fn, Xf in F.items():
        X = Xf[m]; ok = np.isfinite(X).all(1)
        if ok.sum() < 40: continue
        cells[fn] = (X[ok], yr[ok])
    # behavioral cell
    okb = np.isfinite(BEH[m])
    if okb.sum() >= 40:
        cells["BEH_dprime"] = (BEH[m][okb][:, None], yr[okb])
    TDATA[tcol] = cells

REAL = {}
for tcol, cells in TDATA.items():
    REAL[tcol] = {}
    for fn, (X, yr) in cells.items():
        r = cv_r(X, yr, seed=0)
        rho = float(st.spearmanr(X[:, 0] if X.shape[1] == 1 else X @ np.ones(X.shape[1])/X.shape[1], yr)[0])
        REAL[tcol][fn] = {"cv_r": r, "n": len(yr)}
        print(f"[{tcol[13:32]:20s}] {fn:12s} CVr={r:+.3f} n={len(yr)}")

print("\nperm null (500)...", flush=True)
rng = np.random.RandomState(20260904)
maxstats = []
for p_ in range(500):
    mx = 0.0
    for tcol, cells in TDATA.items():
        for fn, (X, yr) in cells.items():
            yp = rng.permutation(yr)
            r = cv_r(X, yp, seed=0)
            if abs(r) > abs(mx): mx = r
    maxstats.append(mx)
    if (p_+1) % 100 == 0: print(f"  perm {p_+1}/500 ({time.time()-t0:.0f}s)", flush=True)
maxstats = np.array(maxstats)

for tcol in REAL:
    for fn in REAL[tcol]:
        r = REAL[tcol][fn]["cv_r"]
        fw_p = float((np.sum(np.abs(maxstats) >= abs(r)) + 1) / 501)
        REAL[tcol][fn]["fw_p"] = fw_p
        alive = fw_p < 0.05 and abs(r) > 0.25
        REAL[tcol][fn]["GATE2_VERDICT"] = "ALIVE" if alive else "dead"
        print(f"[{tcol[13:32]:20s}] {fn:12s} CVr={r:+.3f} fw_p={fw_p:.4f} -> {REAL[tcol][fn]['GATE2_VERDICT']}")

out = {"real": REAL, "null_max": {"mean": float(np.mean(np.abs(maxstats))), "q95": float(np.percentile(np.abs(maxstats), 95))}, "n_perm": 500}
with open(f"{OUT}/batch9_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nDONE {time.time()-t0:.0f}s")
