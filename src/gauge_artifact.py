"""The decisive test: is E3's "second-order memory" an artifact of the gauge?

Argument under test
-------------------
RP^(N-1) has NO continuous global gauge: every choice of representative
x or -x is discontinuous on a codimension-1 set. Two conventions are in use:
  argmax  : flip so the largest-|entry| is positive  -> jumps whenever the
            argmax parcel changes (essentially every frame)
  majneg  : flip so most entries are negative        -> jumps whenever the
            positive fraction crosses 0.5
Both therefore inject spurious discontinuities into the label sequence. If those
jumps are what E3 measures, then:
  (i)  label switches should coincide with gauge flips far above chance;
  (ii) a GAUGE-FREE partition (projective k-means) should lose the memory.

Also re-tests the carrier null with the noise level matched to the real data
(matched on R1, the first-harmonic order parameter), so the comparison is fair.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leida_lib as L
from survival_lib import (e3_stat, markov_null, jnt_dloglik,
                                    labels_from_phases, carrier_field, K)

OUT = os.path.join(L.ROOT, "outputs", "turn3")


def v1_from_phases(ph):
    c, s = np.cos(ph), np.sin(ph)
    cc, ss, cs = (c * c).sum(1), (s * s).sum(1), (c * s).sum(1)
    disc = np.sqrt(np.maximum((cc - ss) ** 2 + 4 * cs * cs, 0))
    l1 = 0.5 * (cc + ss + disc)
    e0, e1 = cs.copy(), l1 - cc
    n = np.hypot(e0, e1)
    n[n == 0] = 1
    e0 /= n
    e1 /= n
    X = e0[:, None] * c + e1[:, None] * s
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def main():
    pass  # read_roi inlined below
    def read_roi(path):
        try:
            d = pd.read_csv(path, sep='\t', comment='#', header=None)
        except Exception:
            return None
        d = d.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        X = d.values.astype(float); X = X[np.isfinite(X).all(1)]
        return X if X.shape[0] > 30 and X.shape[1] > 50 else None
    Z = np.load(os.path.join(L.ROOT, "data", "e123_hbn_labels.npz"), allow_pickle=True)
    keys = [k for k in Z.files if k.endswith("||rest_run-1")]
    sids = sorted(set(k.split("||")[0] for k in keys))[:180]
    MOV = os.path.join(L.ROOT, "data", "hbn_movie_fmri")
    MISSING = [2, 5, 11, 25, 34, 68, 82, 95, 126, 191]
    KEEP = [i for i in range(200) if (i + 1) not in MISSING]
    Ch = np.load(os.path.join(L.ROOT, "pilot", "out", "leida_centroids_190.npy"))

    Xs, phs = [], []
    for sid in sids:
        X = read_roi(os.path.join(MOV, f"{sid}__rest_run-1__roi_stats.csv"))
        if X is None or X.shape[1] != 200:
            continue
        _, ph = L.leida_v1(X[:, KEEP])
        Xs.append(v1_from_phases(ph))
        phs.append(ph)
    print(f"subjects with usable rest_run-1: {len(Xs)}")

    # ---------------- (i) gauge flip rate and coincidence -------------------
    print("\n" + "=" * 74)
    print("(i) ARE LABEL SWITCHES EXPLAINED BY GAUGE FLIPS?")
    print("=" * 74)
    flip_arg, flip_maj, sw_rate, co_arg, co_maj = [], [], [], [], []
    for X in Xs:
        k = np.argmax(np.abs(X), axis=1)
        s_arg = np.where(X[np.arange(len(X)), k] < 0, -1.0, 1.0)
        s_maj = np.where((X > 0).sum(1) > X.shape[1] / 2, -1.0, 1.0)
        fa = (np.diff(s_arg) != 0)
        fm = (np.diff(s_maj) != 0)
        flip_arg.append(fa.mean())
        flip_maj.append(fm.mean())
        # label switches under each gauge
        la = labels_from_phases(phs[len(flip_arg) - 1], Ch, "argmax")
        lm = labels_from_phases(phs[len(flip_arg) - 1], Ch, "majneg")
        sa = (np.diff(la) != 0)
        sm = (np.diff(lm) != 0)
        sw_rate.append((sa.mean(), sm.mean()))
        # conditional: P(switch | gauge flipped) vs P(switch | not flipped)
        co_arg.append((sa[fa].mean() if fa.sum() else np.nan,
                       sa[~fa].mean() if (~fa).sum() else np.nan, fa.sum()))
        co_maj.append((sm[fm].mean() if fm.sum() else np.nan,
                       sm[~fm].mean() if (~fm).sum() else np.nan, fm.sum()))
    print(f"  gauge flip rate: argmax {np.mean(flip_arg):.3f}  majneg {np.mean(flip_maj):.3f}")
    print(f"  label switch rate: argmax {np.mean([s[0] for s in sw_rate]):.3f}  "
          f"majneg {np.mean([s[1] for s in sw_rate]):.3f}")
    ca = np.array([(a, b) for a, b, n in co_arg], float)
    cm = np.array([(a, b) for a, b, n in co_maj], float)
    print(f"  P(switch | gauge flipped) vs P(switch | not flipped):")
    print(f"    argmax: {np.nanmean(ca[:,0]):.3f} vs {np.nanmean(ca[:,1]):.3f}")
    print(f"    majneg: {np.nanmean(cm[:,0]):.3f} vs {np.nanmean(cm[:,1]):.3f}")

    # ---------------- (ii) gauge-free partition ----------------------------
    print("\n" + "=" * 74)
    print("(ii) GAUGE-FREE PARTITION (projective k-means on RP^189)")
    print("=" * 74)
    Xall = np.concatenate(Xs, 0)
    lens = [len(X) for X in Xs]
    for kk in (3, 5, 8):
        lab_all, C, obj = L.projective_kmeans(Xall.astype(np.float64), kk, n_init=4, seed=0)
        seqs_p, o = [], 0
        for T in lens:
            seqs_p.append(lab_all[o:o + T])
            o += T
        e = e3_stat(seqs_p, K=kk)
        m, sd = markov_null(seqs_p, K=kk, nsim=80)
        tr, te = seqs_p[:len(seqs_p)//2], seqs_p[len(seqs_p)//2:]
        dd, nn = jnt_dloglik(tr, te, K=kk)
        print(f"  k={kk}: E3={e:.4f} null={m:.4f}+-{sd:.4f} z={(e-m)/max(sd,1e-9):+.1f}  "
              f"JNT={dd:+.5f}")

    # ---------------- carrier null with matched noise ----------------------
    print("\n" + "=" * 74)
    print("(iii) CARRIER NULL, noise matched to the data on R1")
    print("=" * 74)
    R1_real = np.mean([float(np.abs(np.exp(1j * p).mean(1)).mean()) for p in phs])
    print(f"  real R1 = {R1_real:.4f}")
    best = None
    for noise in np.arange(0.3, 3.01, 0.1):
        ph = carrier_field(375, 190, noise=float(noise), seed=0)
        r1 = float(np.abs(np.exp(1j * ph).mean(1)).mean())
        if best is None or abs(r1 - R1_real) < best[0]:
            best = (abs(r1 - R1_real), float(noise), r1)
    print(f"  matched noise = {best[1]:.2f} (gives R1={best[2]:.4f})")
    sims = [labels_from_phases(carrier_field(375, 190, noise=best[1], seed=i), Ch, "majneg")
            for i in range(len(Xs))]
    e = e3_stat(sims)
    m, sd = markov_null(sims, nsim=80)
    dd, nn = jnt_dloglik(sims[:len(sims)//2], sims[len(sims)//2:])
    print(f"  carrier-only: E3={e:.4f} null={m:.4f}+-{sd:.4f} z={(e-m)/max(sd,1e-9):+.1f} "
          f"JNT={dd:+.5f}")
    print(f"  real labels : E3={e3_stat([labels_from_phases(p, Ch, 'majneg') for p in phs]):.4f}")

    # ---------------- what the labels actually are -------------------------
    print("\n" + "=" * 74)
    print("(iv) SANITY: state counts and occupancy entropy per gauge")
    print("=" * 74)
    for mode in ("argmax", "majneg"):
        alllab = np.concatenate([labels_from_phases(p, Ch, mode) for p in phs])
        cnt = np.bincount(alllab, minlength=5) / len(alllab)
        H = -np.sum(cnt[cnt > 0] * np.log(cnt[cnt > 0]))
        print(f"  {mode:7s}: occupancy {np.round(cnt,3)}  entropy={H:.3f} / {np.log(5):.3f} "
              f"({H/np.log(5):.1%} of max)")


if __name__ == "__main__":
    main()
