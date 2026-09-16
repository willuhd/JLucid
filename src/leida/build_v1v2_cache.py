#!/usr/bin/env python3
"""Build V1/V2 wavelet eigenvector caches (locked g050c5K60 geometry)."""
import time
import numpy as np
from controllability.datasets_cc200 import load_cc200

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
TR = 2.0
F, CYC, CAP = 0.05, 5, 60.0


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
    om = 2 * np.pi * f
    w = np.pi ** -0.25 * np.exp(1j * om * t) * np.exp(-t ** 2 / (2 * (cycles / om) ** 2))
    return w / np.sqrt(np.sum(np.abs(w) ** 2) + 1e-12), L

def phases_of(ts, w, L):
    T, N = ts.shape
    n_fft = 1
    while n_fft < T + L: n_fft <<= 1
    Fw = np.fft.fft(w, n=n_fft)[:, None]
    Fd = np.fft.fft(ts, n=n_fft, axis=0)
    return np.angle(np.fft.ifft(Fd * Fw, axis=0)[L // 2:L // 2 + T])


def v1v2_block(ph):
    c = np.cos(ph); s = np.sin(ph)
    a = (c * c).sum(1); b = (c * s).sum(1); d = (s * s).sum(1)
    G = np.empty((ph.shape[0], 2, 2))
    G[:, 0, 0] = a; G[:, 0, 1] = b; G[:, 1, 0] = b; G[:, 1, 1] = d
    _, U = np.linalg.eigh(G)
    outs = []
    for k in (1, 0):
        u0 = U[:, 0, k]; u1 = U[:, 1, k]
        norm = np.sqrt(u0 * u0 + u1 * u1)
        degen = norm < 1e-12
        u0 = np.where(degen, 1.0, u0 / np.where(degen, 1.0, norm))
        u1 = np.where(degen, 0.0, u1 / np.where(degen, 1.0, norm))
        V = c * u0[:, None] + s * u1[:, None]
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        flip = (V > 0).sum(1) > 0.5 * V.shape[1]
        V[flip] = -V[flip]
        outs.append(V.astype(np.float64))
    return outs[0], outs[1]


t0 = time.time()
print("loading cc200 ...", flush=True)
ts_all, _, meta = load_cc200(qc_only=True)
Ts = [t.shape[0] for t in ts_all]
print("loaded done", flush=True)
w, L = morlet(TR, F, CYC, CAP)
print("kernel ok", flush=True)
V1p, V2p = [], []
t1 = time.time()
for i, ts in enumerate(ts_all):
    v1, v2 = v1v2_block(phases_of(np.asarray(ts, float), w, L))
    V1p.append(v1); V2p.append(v2)
    if (i + 1) % 100 == 0:
        print("did some", flush=True)
V1 = np.concatenate(V1p, axis=0)
V2 = np.concatenate(V2p, axis=0)
print("stacked", flush=True)
np.save(f"{OUT}/V1_adhd200_wavelet.npy", V1)
np.save(f"{OUT}/V2_adhd200_wavelet.npy", V2)
print("saved", flush=True)
