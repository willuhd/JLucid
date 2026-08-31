#!/usr/bin/env python3
"""
PILOT 2 (reliability-only, Gate-0 previews). NO target touched. Pre-committed:
  - joint 2-band dictionary (concat V1 0.05+0.08 c3nat, single k-means k=5)
  - 3-band fused occ {0.05, 0.07, 0.09} (c3nat, Hungarian to band-1, z-fused)
  - 4-band fused occ {0.05, 0.065, 0.08, 0.095} (c3nat, Hungarian to band-1, z-fused)
  - lam1 closure check at 0.05c3nat and 0.08c3nat (subject mean/std of lambda1)
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass
sys.path.insert(0, "/Volumes/thinkplus/Code/JLucid/src")
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from controllability.datasets_cc200 import load_cc200

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/pilot_geometry"
TR = 2.0
K = 5
N_SUB_KM = 40000

BANDS = {  # name -> (freq, cycles, cap)
    "b050": (0.05, 3, 90.0),
    "b065": (0.065, 3, 90.0),
    "b070": (0.07, 3, 90.0),
    "b080": (0.08, 3, 90.0),
    "b090": (0.09, 3, 90.0),
    "b095": (0.095, 3, 90.0),
}

def kernel_len(TR, f, cycles, cap):
    sigma = cycles/(2*np.pi*f)
    L = int(6*sigma/TR); L3 = int(3.0/f/TR); L = max(L, L3)
    if L % 2 == 0: L += 1
    Lcap = int(cap/TR); Lcap = Lcap if Lcap % 2 == 1 else Lcap-1
    if L > Lcap: L = Lcap
    return L

def morlet(TR, f, cycles, cap):
    L = kernel_len(TR, f, cycles, cap)
    t = (np.arange(L) - L//2) * TR
    w = np.pi**-0.25 * np.exp(1j*2*np.pi*f*t) * np.exp(-t**2/(2*(cycles/(2*np.pi*f))**2))
    return w/np.sqrt(np.sum(np.abs(w)**2)+1e-12), L

def wavelet_phase(ts, w, L):
    T, N = ts.shape
    n_fft = 1
    while n_fft < T+L: n_fft <<= 1
    Fk = np.fft.fft(w, n=n_fft)
    Fd = np.fft.fft(ts, n=n_fft, axis=0)
    return np.angle(np.fft.ifft(Fd*Fk[:, None], axis=0)[L//2:L//2+T])

def v1_lam1(phases):
    c = np.cos(phases); s = np.sin(phases)
    a = (c*c).sum(1); b = (c*s).sum(1); d = (s*s).sum(1)
    disc = np.sqrt(np.maximum((a-d)**2+4*b*b, 0.0))
    lam1 = (a+d+disc)/2.0
    u0 = b; u1 = lam1-a
    norm = np.sqrt(u0*u0+u1*u1); degen = norm < 1e-12
    u0 = np.where(degen, 1.0, u0/np.where(degen,1.0,norm))
    u1 = np.where(degen, 0.0, u1/np.where(degen,1.0,norm))
    V1 = c*u0[:, None]+s*u1[:, None]
    V1 /= (np.linalg.norm(V1, axis=1, keepdims=True)+1e-12)
    flip = (V1 > 0).sum(1) > 0.5*V1.shape[1]
    V1[flip] = -V1[flip]
    return V1.astype(np.float32), (lam1/phases.shape[1]).astype(np.float32)

def worker(args):
    ts, f, cycles, cap = args
    w, L = morlet(TR, f, cycles, cap)
    ph = wavelet_phase(ts, w, L)
    V1, lam1 = v1_lam1(ph)
    return V1, lam1, L

def site_resid(X, sites):
    uniq = np.unique(sites)
    D = (sites[:, None] == uniq[None, :]).astype(float)[:, 1:]
    D = np.column_stack([np.ones(len(sites)), D])
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ coef

def sb(r): return 2*r/(1+abs(r)) if r > -1 else -1

def occ_h1h2(labs_list, k=K):
    of, o1, o2 = [], [], []
    for lab in labs_list:
        T = len(lab)
        of.append(np.bincount(lab, minlength=k)/max(T, 1))
        o1.append(np.bincount(lab[:T//2], minlength=k)/max(T//2, 1))
        o2.append(np.bincount(lab[T//2:], minlength=k)/max(T-T//2, 1))
    return np.array(of), np.array(o1), np.array(o2)

def sb_table(o1, o2, sites, k=K):
    valid = np.ones(len(o1), bool)
    per = []
    for j in range(k):
        x1, x2 = o1[:, j], o2[:, j]
        if np.std(x1) < 1e-9 or np.std(x2) < 1e-9: continue
        r = float(np.corrcoef(x1, x2)[0, 1])
        rr = float(np.corrcoef(site_resid(x1, sites), site_resid(x2, sites))[0, 1])
        per.append({"state": j, "r": r, "sb": sb(r), "sb_siteres": sb(rr)})
    return {"per_state": per, "mean_sb": float(np.mean([p["sb"] for p in per])),
            "mean_sb_siteres": float(np.mean([p["sb_siteres"] for p in per]))}

def km_labels(V_all):
    rng = np.random.default_rng(0)
    idx = rng.choice(V_all.shape[0], min(N_SUB_KM, V_all.shape[0]), replace=False)
    km = KMeans(n_clusters=K, n_init=10, random_state=0, max_iter=100).fit(V_all[idx])
    return km.predict(V_all).astype(np.int16)

t0 = time.time()
print("loading ...", flush=True)
ts_all, _, meta = load_cc200(qc_only=True)
sites = meta["sites"]; Ts = [t.shape[0] for t in ts_all]
print(f"loaded {len(ts_all)} in {time.time()-t0:.0f}s", flush=True)

need = ["b050", "b065", "b070", "b080", "b090", "b095"]
V1s, LAM1s, Ls = {}, {}, {}
for name in need:
    f, c, cap = BANDS[name]
    t1 = time.time()
    with mp.Pool(4) as pool:
        outs = pool.map(worker, [(ts_all[i], f, c, cap) for i in range(len(ts_all))], chunksize=16)
    V1s[name] = np.concatenate([o[0] for o in outs], 0)
    LAM1s[name] = np.concatenate([o[1] for o in outs], 0)
    Ls[name] = outs[0][2]
    print(f"  {name} f={f} L={outs[0][2]}TR ({outs[0][2]*TR:.0f}s) done {time.time()-t1:.0f}s", flush=True)

results = {}

# ---------- joint 2-band dictionary ----------
t1 = time.time()
J = np.concatenate([V1s["b050"], V1s["b080"]], axis=1)  # 380-dim, rows have sqrt(2) norm (equal band weight)
lab = km_labels(J)
labs_list, pos = [], 0
for T in Ts:
    labs_list.append(lab[pos:pos+T]); pos += T
of, o1, o2 = occ_h1h2(labs_list)
results["JOINT_2band_050_080_dict"] = sb_table(o1, o2, sites)
results["JOINT_2band_050_080_dict"]["L_secs"] = None
print(f"JOINT 2-band dict: mean SB={results['JOINT_2band_050_080_dict']['mean_sb']:.3f} "
      f"siteres={results['JOINT_2band_050_080_dict']['mean_sb_siteres']:.3f} "
      f"per-state {[round(p['sb'],3) for p in results['JOINT_2band_050_080_dict']['per_state']]} ({time.time()-t1:.0f}s)", flush=True)

# ---------- fused multi-band occupancy ----------
def build_band_occ(name):
    lab = km_labels(V1s[name])
    labs_list, pos = [], 0
    for T in Ts:
        labs_list.append(lab[pos:pos+T]); pos += T
    return occ_h1h2(labs_list)

def z(a): return (a - a.mean(0)) / (a.std(0) + 1e-12)

def fuse(band_names):
    ofs = {n: build_band_occ(n) for n in band_names}
    ref = band_names[0]
    A = ofs[ref][0]
    perms = {ref: np.arange(K)}
    for n in band_names[1:]:
        B = ofs[n][0]
        C = np.array([[abs(float(np.corrcoef(A[:, i], B[:, j])[0, 1])) for j in range(K)] for i in range(K)])
        C = np.nan_to_num(C)
        ri, ci = linear_sum_assignment(-C)
        perms[n] = ci
    def fused(parts):  # parts: dict name-> (of, o1, o2), use given part index 0/1/2
        acc_f = z(ofs[ref][parts[0]])
        acc_1 = z(ofs[ref][parts[1]])
        acc_2 = z(ofs[ref][parts[2]])
        for n in band_names[1:]:
            p = perms[n]
            acc_f = acc_f + z(ofs[n][parts[0]])[:, p]
            acc_1 = acc_1 + z(ofs[n][parts[1]])[:, p]
            acc_2 = acc_2 + z(ofs[n][parts[2]])[:, p]
        Bn = len(band_names)
        return acc_f/Bn, acc_1/Bn, acc_2/Bn
    ff, f1, f2 = fused((0, 1, 2))
    return sb_table(f1, f2, sites), {n: perms[n].tolist() for n in perms}

for key, bands in [("FUSED_3band_050_070_090", ["b050", "b070", "b090"]),
                   ("FUSED_4band_050_065_080_095", ["b050", "b065", "b080", "b095"])]:
    t1 = time.time()
    tab, perms = fuse(bands)
    results[key] = tab
    results[key]["perms"] = perms
    print(f"{key}: mean SB={tab['mean_sb']:.3f} siteres={tab['mean_sb_siteres']:.3f} "
          f"per-state {[round(p['sb'],3) for p in tab['per_state']]} ({time.time()-t1:.0f}s)", flush=True)

# ---------- lam1 closure check ----------
print("\nlam1 (eigen-leader coherence) subject mean/std split-half SB by geometry:", flush=True)
lam = {}
for name in ["b050", "b080"]:
    arr = LAM1s[name]
    segs, pos = [], 0
    for T in Ts:
        segs.append(arr[pos:pos+T]); pos += T
    m = np.array([s.mean() for s in segs]); sd = np.array([s.std() for s in segs])
    r_m = float(np.corrcoef(m[::1][:len(m)//2], m[::1][len(m)//2:])[0, 1]) if False else None
    # proper split-half: first vs second half of RUN per subject
    m1 = np.array([s[:len(s)//2].mean() for s in segs]); m2 = np.array([s[len(s)//2:].mean() for s in segs])
    s1 = np.array([s[:len(s)//2].std() for s in segs]); s2 = np.array([s[len(s)//2:].std() for s in segs])
    rm = float(np.corrcoef(m1, m2)[0, 1]); rsd = float(np.corrcoef(s1, s2)[0, 1])
    rm_r = float(np.corrcoef(site_resid(m1, sites), site_resid(m2, sites))[0, 1])
    lam[name] = {"mean_sb": sb(rm), "std_sb": sb(rsd), "mean_sb_siteres": sb(rm_r), "r_mean": rm}
    print(f"  {name}: lam1.mean SB={sb(rm):.3f} (siteres {sb(rm_r):.3f}), lam1.std SB={sb(rsd):.3f}", flush=True)
results["LAM1_check"] = lam

with open(f"{OUT}/pilot2_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
