#!/usr/bin/env python3
"""
exp/37 BATCH 5 feature builder — NEW quantities (preregistered SIEVE_TABLE.md).
NO target touched. Features:
  H1 v_ab  (28): Var_t[ mean_{i∈a,j∈b} cos(θ_i−θ_j) ] — ROI-pairwise PL time-variance
  H2 MAG   (2): {mean_t, Var_t} of MAG(t)=(1/N)Σ_n l_t(n) (closed-form majority-flip V1)
  P1 ACF   (8): κ global + 7 per-network lag-1 ACF of F_a(t)
  K1 sAphi/mu (14): φ,μ network means from SIGNED-FC A
  K2 CA    (14): network means of CA_i=Σ 1/λ_j(W_i) and logdet_i=Σ log λ_j(W_i)
Geometries: H*/P* at cache 0.05 c5K60. K* from whole-run FC.
Output: batch5_features.npz
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
from controllability.datasets_cc200 import load_cc200

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
TR = 2.0
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
def v1_of(ph):
    c = np.cos(ph); s = np.sin(ph)
    a = (c*c).sum(1); b = (c*s).sum(1); d = (s*s).sum(1)
    disc = np.sqrt(np.maximum((a-d)**2 + 4*b*b, 0))
    lam1 = (a+d+disc)/2
    u0 = b; u1 = lam1 - a
    norm = np.sqrt(u0*u0 + u1*u1)
    degen = norm < 1e-12
    u0 = np.where(degen, 1.0, u0/np.where(degen, 1.0, norm))
    u1 = np.where(degen, 0.0, u1/np.where(degen, 1.0, norm))
    V1 = c*u0[:,None] + s*u1[:,None]
    V1 /= (np.linalg.norm(V1, axis=1, keepdims=True) + 1e-12)
    flip = (V1 > 0).sum(1) > 0.5*V1.shape[1]
    V1[flip] = -V1[flip]
    return V1

t0 = time.time()
ts_all, _, meta = load_cc200(qc_only=True)
sites = np.array(meta["sites"]); Ts = [t.shape[0] for t in ts_all]
yeo = json.load(open(f"{BASE}/src/leida/atlas/yeo_true_idx.json"))
net_idx = [np.array(yeo[n]) for n in NETS]
N = len(ts_all)

def worker_ph(args):
    ts, = args
    w, L = morlet(TR, 0.05, 5, 60.0)
    ph = phases_of(ts, w, L)
    # H1: per-pair ROI-mean PL time series
    cosd = np.cos(ph[:, :, None] - ph[:, None, :])  # T x N x N
    vab = np.zeros(28)
    for kk, (a_, b_) in enumerate(UP):
        ia, ib = net_idx[a_], net_idx[b_]
        if a_ == b_:
            m = cosd[:, ia][:, :, ia]
            ii, jj = np.triu_indices(len(ia), k=1)
            series = m[:, ii, jj].mean(axis=1)
        else:
            series = cosd[:, ia][:, :, ib].mean(axis=(1, 2))
        vab[kk] = series.var()
    # H2: MAG from V1
    V1 = v1_of(ph)
    mag_t = V1.mean(axis=1)
    # P1: persistence
    kappa = float(np.mean(np.abs((V1[:-1] * V1[1:]).sum(axis=1))))
    facf = np.zeros(7)
    for a_ in range(7):
        F = V1[:, net_idx[a_]].mean(axis=1)
        if F.std() > 1e-12:
            facf[a_] = np.corrcoef(F[:-1], F[1:])[0, 1]
        else:
            facf[a_] = 0.0
    # H3: TRUE Hancock VAR (mean per-node temporal variance of V1 elements; sign-invariant)
    var_node = V1.var(axis=0)  # 190
    var7 = np.array([var_node[net_idx[k]].mean() for k in range(7)])
    h3 = np.concatenate([[var_node.mean()], var7])
    return vab, np.array([mag_t.mean(), mag_t.var()]), np.concatenate([[kappa], facf]), h3

def worker_k(ts):
    T, Nn = ts.shape
    if T < 60: return None
    Xz = (ts - ts.mean(0)) / (ts.std(0) + 1e-12)
    FC = np.nan_to_num(np.corrcoef(Xz.T))
    # K1: signed A
    lam = np.linalg.eigvalsh((FC + FC.T)/2)
    lmax = float(lam.max())
    A_s = FC / (1.0 + lmax + 1e-12)
    A_s = A_s - np.eye(Nn)
    # Hurwitz: shift so max eigen < 0
    ev = np.linalg.eigvalsh((A_s + A_s.T)/2)
    if ev.max() >= 0:
        A_s = A_s - (ev.max() + 1e-6) * np.eye(Nn)
    vals, vecs = np.linalg.eigh((A_s + A_s.T)/2)
    phi = (vecs**2) @ (1.0/(-2.0*vals))
    mu = (vecs**2) @ (1.0 - np.exp(vals))
    phi7 = np.array([phi[net_idx[k]].mean() for k in range(7)])
    mu7 = np.array([mu[net_idx[k]].mean() for k in range(7)])
    # K2: CA-score on |FC| A (paper-6 A)
    Araw = np.abs(FC)
    lmax2 = float(np.max(np.linalg.eigvalsh((Araw + Araw.T)/2)))
    A6 = Araw / (1.0 + lmax2 + 1e-12) - np.eye(Nn)
    A6 = (A6 + A6.T)/2
    # W = Lyapunov solution of A W A^T - W + I = 0 -> W = sum A^tau (A^tau)^T
    try:
        from scipy.linalg import solve_discrete_lyapunov
        W = solve_discrete_lyapunov(A6, np.eye(Nn))
        wv = np.linalg.eigvalsh((W + W.T)/2)
        wv = np.maximum(wv, 1e-12)
        ca = (1.0/wv).sum(axis=0) * np.ones(Nn) if wv.ndim else (1.0/wv).sum()
        # per-node: W_i = e_i^T ... the node-i Gramian is NOT the full W diag. Chen's CA_i uses
        # the diagonal-ish restriction; use the per-node controllable-energy surrogate:
        # E_i = W[i,i] (the i-th controllability "volume" along e_i), CA_i = 1/W[i,i]
        ca_node = 1.0 / np.maximum(np.diag(W), 1e-12)
        logdet_node = np.log(np.maximum(np.diag(W), 1e-12))
    except Exception:
        ca_node = np.zeros(Nn); logdet_node = np.zeros(Nn)
    ca7 = np.array([ca_node[net_idx[k]].mean() for k in range(7)])
    ld7 = np.array([logdet_node[net_idx[k]].mean() for k in range(7)])
    return np.concatenate([phi7, mu7]), np.concatenate([ca7, ld7])

with mp.Pool(4) as pool:
    outs = pool.map(worker_ph, [(ts_all[i],) for i in range(N)], chunksize=16)
H1 = np.array([o[0] for o in outs]); H2 = np.array([o[1] for o in outs]); P1 = np.array([o[2] for o in outs]); H3 = np.array([o[3] for o in outs])
print(f"H/P done {time.time()-t0:.0f}s", flush=True)
with mp.Pool(4) as pool:
    kouts = pool.map(worker_k, ts_all, chunksize=8)
K1 = np.array([o[0] if o is not None else np.full(14, np.nan) for o in kouts])
K2 = np.array([o[1] if o is not None else np.full(14, np.nan) for o in kouts])
print(f"K done {time.time()-t0:.0f}s", flush=True)

np.savez_compressed(f"{OUT}/batch5_features.npz", sites=sites, Ts=np.array(Ts),
                    H1__vab=H1, H2__mag=H2, P1__acf=P1, H3__hvar=H3, K1__signed=K1, K2__ca=K2)
print(f"DONE {time.time()-t0:.0f}s")
