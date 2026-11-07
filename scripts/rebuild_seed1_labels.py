#!/usr/bin/env python
"""Rebuild an independent seed-1 k=5 lens from retained raw CC200 runs.

Source-only replacement for the rescued e123_adhd200_seed1_labels.npz bytes:
no producer script for those bytes exists anywhere in the archive (only
loaders reference them), so this file re-derives an equivalent lens from
retained data instead of copying frozen bytes.

Procedure mirrors .agents/scripts/denovo_refit.py with an independent seed:
per-frame LEiDA vectors (lean src/leida_lib, argmax gauge), k-means k=5 fit
on a 1/3-run subsample (rng seed 1, k-means seed 1), Euclidean labeling of
ALL runs. The fresh fit's clusters are then Hungarian-matched (|cos|) to the
denovo reference centroids and permuted into denovo order, so literal index k
here means "the pattern matched to denovo-k" -- in particular the o4-matched
pattern lands on index 4, and src/o4_occupancy.py tests seed1 index 4.
Relabeling changes neither the partition nor any occupancy value. The match
table is printed for verifiability.

Writes (gitignored *.npz/*.npy):
  data/e123_adhd200_seed1_labels.npz
  data/e123_adhd200_seed1_C.npy
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import leida_lib as L

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    fs = L.cc200_paths("sfnwmrda")
    Xs, keys = [], []
    for i, f in enumerate(fs):
        x = L.load_cc200(f)
        if x is None:
            continue
        X, _ = L.leida_v1(x)
        X = L.apply_gauge(X, "argmax")
        Xs.append(X.astype(np.float32))
        keys.append(f"{L.subject_of(f)}||{i}")
    print(f"sfn runs={len(Xs)} frames={sum(len(v) for v in Xs)}", flush=True)
    rng = np.random.default_rng(1)
    pick = rng.choice(len(Xs), size=max(1, len(Xs) // 3), replace=False)
    km = KMeans(n_clusters=5, n_init=10, random_state=1).fit(
        np.concatenate([Xs[i] for i in pick]))
    Cn = km.cluster_centers_.astype(np.float64)
    Cref = np.load(os.path.join(ROOT, "data", "e123_adhd200_denovo_C.npy"))
    Cn_n = Cn / np.linalg.norm(Cn, axis=1, keepdims=True)
    Cr_n = Cref / np.linalg.norm(Cref, axis=1, keepdims=True)
    M = np.abs(Cn_n @ Cr_n.T)
    ri, ci = linear_sum_assignment(-M)
    order = np.empty(5, dtype=int)  # order[denovo_k] = new_idx
    for a, b in zip(ri, ci):
        order[int(b)] = int(a)
    print("match to denovo (denovo_k<-new_idx |cos|): " +
          " ".join(f"{int(b)}<-{int(a)}:{M[int(a), int(b)]:.3f}"
                   for a, b in zip(ri, ci)))
    inv = np.empty(5, dtype=int)  # inv[old_idx] = denovo_idx
    for dk in range(5):
        inv[order[dk]] = dk
    Cp = Cn[order]
    np.save(os.path.join(ROOT, "data", "e123_adhd200_seed1_C.npy"), Cp)
    res = {}
    for k, X in zip(keys, Xs):
        lab = ((X[:, None, :] - Cn[None]) ** 2).sum(-1).argmin(1).astype(np.int32)
        res[k] = inv[lab]
    np.savez_compressed(os.path.join(ROOT, "data", "e123_adhd200_seed1_labels.npz"), **res)
    print(f"saved labels {len(res)}, denovo-4 pattern at index 4")


if __name__ == "__main__":
    main()
