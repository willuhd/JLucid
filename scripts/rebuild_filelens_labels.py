#!/usr/bin/env python
"""Rebuild the file-lens k=5 labels from retained raw CC200 runs.

Ported from .agents/scripts/reconstruct_file_lens.py: same LEiDA pipeline,
k=5, n_init=10, seed 0, fitted on ALL sfn runs. Changes vs the archived
original are paths only: repo-relative data/ instead of a hardcoded absolute
ROOT, outputs to gitignored data/ (npz/npy are ignored by rule).
Writes:
  data/e123_adhd200_filelens_recon_C.npy
  data/e123_adhd200_filelens_recon.npz
"""
import numpy as np, pandas as pd, glob, os
from sklearn.cluster import KMeans

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def hilbert_phase(x):
    T = x.shape[0]; F = np.fft.fft(x, axis=0)
    h = np.zeros(T); h[0] = 1; h[T // 2] = 1; h[1:T // 2] = 2
    return np.angle(np.fft.ifft(F * h[:, None], axis=0))


def leida(ts):
    T, N = ts.shape
    z = (ts - ts.mean(0)) / np.where(ts.std(0) == 0, 1e-12, ts.std(0))
    ph = hilbert_phase(z)
    out = np.empty((T, N))
    for t in range(T):
        c = np.cos(ph[t]); s = np.sin(ph[t])
        G = np.array([[c @ c, c @ s], [c @ s, s @ s]])
        _, V = np.linalg.eigh(G); v = V[:, -1]
        x = v[0] * c + v[1] * s; n = np.linalg.norm(x)
        if n > 0: x = x / n
        k = int(np.argmax(np.abs(x)))
        if x[k] < 0: x = -x
        out[t] = x
    return out


fs = sorted([f for f in glob.glob(f'{ROOT}/data/adhd200/*/*/*cc200_TCs.1D') if 'sfnwmrda' in os.path.basename(f)])
print('sfn files', len(fs))
Xs = []
for f in fs:
    try:
        a = np.loadtxt(f, skiprows=1, usecols=range(2, 192))
    except Exception:
        continue
    if a.shape[0] < 120 or a.shape[1] != 190:
        continue
    Xs.append(leida(a))
Xall = np.concatenate(Xs, axis=0)
print('frames', Xall.shape)
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xall)
C = km.cluster_centers_
np.save(f'{ROOT}/data/e123_adhd200_filelens_recon_C.npy', C)
res = {}
for f in fs:
    try:
        a = np.loadtxt(f, skiprows=1, usecols=range(2, 192))
    except Exception:
        continue
    if a.shape[0] < 120 or a.shape[1] != 190:
        continue
    X = leida(a)
    lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1).astype(np.int32)
    sid = os.path.basename(os.path.dirname(f))
    res[f'{sid}||{os.path.basename(f)}'] = lab
np.savez_compressed(f'{ROOT}/data/e123_adhd200_filelens_recon.npz', **res)
print('saved labels', len(res), 'cluster fractions', np.round(np.bincount(km.labels_, minlength=5) / len(km.labels_), 3))
