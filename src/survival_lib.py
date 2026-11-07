"""Does anything survive?  The phenotype-free laws under gauge, k, and a
carrier-only null.

The project's non-ADHD results (E1 dwell-routing, E3 Markov order, JNT joint
dwell+order, T2 transition geometry) all rest on the SAME k=5 label sequence
from data/e123_hbn_labels.npz, produced by the same invalid sign gauge that
Tier 1 showed collapses 5 states to 3 projective modes.  So they are not
automatically safe.  Three questions, in order of severity:

Q1  CARRIER NULL (decisive).  If the field is a rotating cut direction plus
    independent per-parcel phase noise -- i.e. the project's own carrier law --
    does the label sequence show the same "second-order memory" and
    "dwell-conditioned routing"?  If yes, E3/JNT measure the carrier and the
    quantisation, not state structure.

Q2  GAUGE.  Recompute E3/JNT from labels produced with the Paper-7 gauge
    (majority negative) instead of the project's argmax gauge.

Q3  k.  Recompute at k=3 (the number of projective modes Tier 1 found) and k=8.

Definitions taken verbatim from docs/PHENOTYPE_FREE_PORTFOLIO.md:
  E3  L1 = sum_{ij} |obs2_ij - (P^2)_ij| ; null = homogeneous Markov chains
      matched in length.  Reported: L1=0.8452 vs null 0.0484 +- 0.0096.
  JNT held-out dloglik/TR of dwell-binned model M2 over homogeneous M1.
      Reported: +0.01285 nats (SE .00100, z=+12.9).
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leida_lib as L

OUT = os.path.join(L.ROOT, "outputs", "turn3")
K = 5


# ------------------------------------------------------------------ metrics --
def trans1(seqs, K=K):
    C = np.zeros((K, K))
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            C[a, b] += 1
    P = C / np.maximum(C.sum(1, keepdims=True), 1)
    return C, P


def trans2(seqs, K=K):
    C = np.zeros((K, K))
    for s in seqs:
        for a, b in zip(s[:-2], s[2:]):
            C[a, b] += 1
    P = C / np.maximum(C.sum(1, keepdims=True), 1)
    return C, P


def e3_stat(seqs, K=K):
    _, P1 = trans1(seqs, K)
    _, P2 = trans2(seqs, K)
    return float(np.abs(P2 - P1 @ P1).sum())


def markov_null(seqs, K=K, nsim=200, seed=0):
    rng = np.random.default_rng(seed)
    _, P = trans1(seqs, K)
    # guard: a state that never occurs as an origin gets a uniform row
    for i in range(K):
        if P[i].sum() <= 0:
            P[i] = np.ones(K) / K
        P[i] = P[i] / P[i].sum()
    lens = [len(s) for s in seqs]
    stats = []
    for _ in range(nsim):
        sims = []
        for T in lens:
            s = np.empty(T, dtype=int)
            s[0] = rng.integers(K)
            for t in range(1, T):
                s[t] = rng.choice(K, p=P[s[t - 1]])
            sims.append(s)
        stats.append(e3_stat(sims, K))
    return float(np.mean(stats)), float(np.std(stats))


def dwell_bins(seq):
    """origin-dwell bin per transition: 1 -> 0, 2-3 -> 1, 4+ -> 2."""
    out = []
    t = 0
    T = len(seq)
    while t < T - 1:
        s = seq[t]
        d = 1
        while t + d < T and seq[t + d] == s:
            d += 1
        b = 0 if d == 1 else (1 if d <= 3 else 2)
        if t + d < T:
            out.append((s, b, seq[t + d]))
        t += d
    return out


def jnt_dloglik(train_seqs, test_seqs, K=K):
    """Held-out nats/TR: dwell-binned model M2 vs homogeneous M1."""
    # M1
    C1 = np.zeros((K, K))
    for s in train_seqs:
        for a, b in zip(s[:-1], s[1:]):
            C1[a, b] += 1
    P1 = (C1 + 1e-6) / (C1.sum(1, keepdims=True) + 1e-6 * K)
    # M2: dwell-binned
    C2 = np.zeros((K, 3, K))
    for s in train_seqs:
        for a, b, c in dwell_bins(s):
            C2[a, b, c] += 1
    P2 = (C2 + 1e-6) / (C2.sum(2, keepdims=True) + 1e-6 * K)
    ll1 = ll2 = 0.0
    n = 0
    for s in test_seqs:
        for a, b, c in dwell_bins(s):
            ll1 += np.log(P1[a, c])
            ll2 += np.log(P2[a, b, c])
            n += 1
    return (ll2 - ll1) / max(n, 1), n


# ------------------------------------------------------- carrier simulation --
def carrier_field(T, N, omega_hz=0.04, tr=2.0, noise=1.0, seed=0, coupling=0.0):
    """A rotating cut direction + independent per-parcel phase noise.

    theta_i(t) = phi(t) + delta_i + noise,  phi(t) = 2*pi*omega*t*tr
    This is exactly the structure the project's carrier law describes.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T) * tr
    phi = 2 * np.pi * omega_hz * t
    delta = rng.uniform(-np.pi, np.pi, N)
    ph = phi[:, None] + delta[None, :] + rng.normal(0, noise, (T, N))
    return np.angle(np.exp(1j * ph))


def label_from_phases(ph, C, mode="majneg"):
    X, _ = L.leida_v1_from_phases(ph) if hasattr(L, "leida_v1_from_phases") else (None, None)
    raise NotImplementedError


def labels_from_phases(ph, C, mode="majneg"):
    """Build v1 from given phases, apply gauge, assign to centroids."""
    c = np.cos(ph)
    s = np.sin(ph)
    cc = (c * c).sum(1)
    ss = (s * s).sum(1)
    cs = (c * s).sum(1)
    disc = np.sqrt(np.maximum((cc - ss) ** 2 + 4 * cs * cs, 0))
    l1 = 0.5 * (cc + ss + disc)
    e0 = cs.copy()
    e1 = l1 - cc
    nrm = np.hypot(e0, e1)
    nrm[nrm == 0] = 1
    e0 /= nrm
    e1 /= nrm
    X = e0[:, None] * c + e1[:, None] * s
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = L.apply_gauge(X, mode)
    lab = np.empty(len(X), dtype=np.int32)
    for i in range(0, len(X), 20000):
        ch = X[i:i + 20000]
        lab[i:i + 20000] = ((ch[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    return lab


def main():
    Z = np.load(os.path.join(L.ROOT, "data", "e123_hbn_labels.npz"), allow_pickle=True)
    keys = [k for k in Z.files if k.endswith("||rest_run-1")]
    seqs = [np.asarray(Z[k]).astype(int) for k in keys]
    lens = np.array([len(s) for s in seqs])
    print(f"HBN rest sequences: {len(seqs)}  length {lens.min()}-{lens.max()} "
          f"(median {int(np.median(lens))})")

    # ---- hygiene: reject degenerate sequences -------------------------------
    ok = []
    for s in seqs:
        u = np.unique(s)
        if len(u) < 3:
            continue
        ok.append(s)
    print(f"  after rejecting <3-distinct-label sequences: {len(ok)}")
    seqs = ok

    # ================= Q1: the project's own numbers =======================
    print("\n" + "=" * 74)
    print("Q1a  reproduce E3 on the shipped labels")
    print("=" * 74)
    e3_obs = e3_stat(seqs)
    m, sd = markov_null(seqs, nsim=200)
    print(f"  E3 L1(obs2 - P^2) = {e3_obs:.4f}")
    print(f"  homogeneous-Markov null: {m:.4f} +- {sd:.4f}  -> z = {(e3_obs-m)/sd:+.1f}")
    print(f"  doc reports L1=0.8452, null 0.0484 +- 0.0096")

    print("\n" + "=" * 74)
    print("Q1b  reproduce JNT on the shipped labels (subject-split CV)")
    print("=" * 74)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(seqs))
    half = len(seqs) // 2
    tr, te = [seqs[i] for i in idx[:half]], [seqs[i] for i in idx[half:]]
    d, n = jnt_dloglik(tr, te)
    print(f"  held-out dloglik/TR = {d:+.5f} nats over {n} transitions")
    print(f"  doc reports +0.01285")

    # ================= Q1c: the carrier null ===============================
    print("\n" + "=" * 74)
    print("Q1c  CARRIER NULL: a rotating cut + independent phase noise")
    print("      (i.e. the project's own carrier law, with NO discrete states)")
    print("=" * 74)
    C = np.load(os.path.join(L.ROOT, "data", "e123_adhd200_denovo_C.npy"))
    # use the HBN-fitted lens, which is what these labels came from
    Ch = np.load(os.path.join(L.ROOT, "pilot", "out", "leida_centroids_190.npy"))
    print(f"  centroids: HBN lens {Ch.shape}, norms {np.round(np.linalg.norm(Ch,axis=1),3)}")
    print(f"  {'noise':>6s} {'E3 L1':>9s} {'null':>9s} {'z':>7s} {'JNT':>10s}")
    for noise in (0.6, 1.0, 1.5):
        sims = []
        for i in range(len(seqs)):
            T = int(lens[i]) if i < len(lens) else 375
            ph = carrier_field(T, 190, noise=noise, seed=i)
            sims.append(labels_from_phases(ph, Ch, "majneg"))
        e = e3_stat(sims)
        mm, ss = markov_null(sims, nsim=60)
        tr2, te2 = sims[:half], sims[half:]
        dd, nn = jnt_dloglik(tr2, te2)
        print(f"  {noise:6.1f} {e:9.4f} {mm:9.4f} {(e-mm)/max(ss,1e-9):+7.1f} {dd:+10.5f}")
    print("  -> if E3/JNT effects appear here with NO state structure, the laws")
    print("     describe the carrier and the quantisation, not discrete states.")

    # ================= Q2: gauge ===========================================
    print("\n" + "=" * 74)
    print("Q2  GAUGE: recompute from raw ROI data under both gauges")
    print("=" * 74)
    MOV = os.path.join(L.ROOT, "data", "hbn_movie_fmri")
    sys.path.insert(0, os.path.join(L.ROOT, "verify_movie_fmri"))
    try:
        from isc_core import read_roi
    except Exception as e:
        print(f"  cannot import isc_core: {e}")
        return
    sids = sorted(set(k.split("||")[0] for k in keys))[:180]
    print(f"  regenerating labels for {len(sids)} subjects x rest_run-1")
    MISSING = [2, 5, 11, 25, 34, 68, 82, 95, 126, 191]
    KEEP = [i for i in range(200) if (i + 1) not in MISSING]
    out = {}
    for mode in ("argmax", "majneg"):
        ss = []
        for sid in sids:
            X = read_roi(os.path.join(MOV, f"{sid}__rest_run-1__roi_stats.csv"))
            if X is None or X.shape[1] != 200:
                continue
            _, ph = L.leida_v1(X[:, KEEP])
            ss.append(labels_from_phases(ph, Ch, mode))
        out[mode] = ss
        e = e3_stat(ss)
        mm, sd2 = markov_null(ss, nsim=120)
        tr3, te3 = ss[:len(ss)//2], ss[len(ss)//2:]
        dd, nn = jnt_dloglik(tr3, te3)
        print(f"  {mode:7s}: n={len(ss)} E3={e:.4f} null={mm:.4f}+-{sd2:.4f} "
              f"z={(e-mm)/max(sd2,1e-9):+.1f}  JNT={dd:+.5f}")
    # agreement between the two label streams
    if len(out["argmax"]) == len(out["majneg"]):
        agr = [np.mean(a == b) for a, b in zip(out["argmax"], out["majneg"])]
        print(f"  label agreement argmax vs majneg: mean {np.mean(agr):.3f} "
              f"(min {np.min(agr):.3f}, max {np.max(agr):.3f})")


if __name__ == "__main__":
    main()
