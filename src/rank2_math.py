"""THE MATHEMATICS OF BOTH OPERATORS — what they are, structurally.

The question: can the whole methodology be stated in math terms? And is
"Jacobian" relevant, given Paper 6 explicitly builds on one?

Claim to test: BOTH operators are degenerate as dynamical systems, and the
degeneracy is a theorem, not an empirical accident.

  Paper 6:  A = |FC|/(c + lam_max(|FC|)) - I   is SYMMETRIC by construction.
            Hence A = V diag(lam) V' with real lam < 0.
            The linear system x' = Ax is then a GRADIENT FLOW:
                x' = -grad V(x),  V(x) = -0.5 x'Ax
            Gradient flows admit NO periodic orbits (a standard theorem).
            So Paper 6's "state transitions" cannot be oscillatory; every
            trajectory is a monotone decay to the origin.
            Corollary: "control energy" is the cost of pushing a decaying
            system, not the cost of a brain state transition.

  LEiDA:    C = cos(th_i - th_j) = u u' + v v'  is exactly rank 2 (T1).
            The leading eigenvector is v1 = cos(th - phi)/sqrt(lam1) (T2).
            So the entire per-frame object is ONE angle phi plus one
            coherence R2.  The "state sequence" is a quantisation of phi.
            The dynamics of phi is, empirically, rotation at the band
            carrier (measured ratio 1.00-1.16 in PennLEAD, 1.16-1.73 elsewhere).
            A uniform rotation on S^1 has no attractors, no bifurcations,
            no metastability: its only invariant is the rotation number.

Both statements are checkable. This script checks them.

New result to establish (the strongest one):
  If the spectrum of A is concentrated near -1 (which it is: A = |FC|/(c+lam_max)
  - I with |FC| having small eigenvalues), then W(T) is near-isotropic, so
  control energy E = xT' W^-1 xT is approximately a scalar times ||xT||^2.
  Then "control energy" carries NO network information beyond the norm of the
  target vector, and Paper 6's network differences are differences in ||xT||.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leida_lib as L

OUT = os.path.join(L.ROOT, "outputs", "turn3")
T_H = 1.0


def paper6_A(ts, c=1.0):
    FC = np.corrcoef(ts.T)
    M = np.abs(FC)
    lmax = np.linalg.eigvalsh(M).max()
    return M / (c + lmax) - np.eye(len(M)), lmax


def gramian(A, T=T_H):
    lam, V = np.linalg.eigh(A)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(np.abs(lam) > 1e-12, (np.exp(2 * lam * T) - 1) / (2 * lam), T)
    return (V * f) @ V.T


def main():
    fs = L.cc200_paths("sfnwmrda")[:60]
    print("=" * 78)
    print("1  IS PAPER 6's A A GRADIENT FLOW?  (symmetry + spectral check)")
    print("=" * 78)
    sym, lam_lo, lam_hi, iso, cond = [], [], [], [], []
    for f in fs:
        x = L.load_cc200(f)
        if x is None:
            continue
        A, lmax = paper6_A(x)
        sym.append(float(np.abs(A - A.T).max()))
        lam = np.linalg.eigvalsh(A)
        lam_lo.append(float(lam.min()))
        lam_hi.append(float(lam.max()))
        # how isotropic is the Gramian?
        W = gramian(A)
        ev = np.linalg.eigvalsh(W)
        iso.append(float(ev.min() / ev.max()))       # 1.0 = perfectly isotropic
        cond.append(float(ev.max() / ev.min()))
    print(f"  runs: {len(sym)}")
    print(f"  max |A - A'|            = {max(sym):.3e}   (symmetric to machine precision)")
    print(f"  eigenvalue range        = [{np.mean(lam_lo):.4f}, {np.mean(lam_hi):.4f}]")
    print(f"  spread of the spectrum  = {np.mean(lam_hi)-np.mean(lam_lo):.4f}")
    print(f"  Gramian eigenvalue ratio (min/max) = {np.mean(iso):.4f}  "
          f"(1.0 would be perfectly isotropic)")
    print(f"  Gramian condition number           = {np.mean(cond):.3f}")
    print()
    print("  A is symmetric -> all eigenvalues real -> x'=Ax is a gradient flow")
    print("  -> NO periodic orbits (theorem). Paper 6's dynamics cannot oscillate.")

    print("\n" + "=" * 78)
    print("2  IS 'CONTROL ENERGY' JUST THE TARGET NORM?")
    print("=" * 78)
    # the seven-network targets, as in paper 6
    mp = pd.read_csv(os.path.join(L.ROOT, "datasets", "hbn_t2", "cc200_yeo_mapping.csv"))
    miss = [2, 5, 11, 25, 34, 68, 82, 95, 126, 191]
    keep = [i for i in range(200) if (i + 1) not in miss]
    pos = {p: j for j, p in enumerate(keep)}
    nets = {}
    for net, g in mp.groupby("network"):
        if net == "unmapped":
            continue
        idx = [pos[int(p)] for p in g.parcel if int(p) in pos]
        if len(idx) >= 5:
            nets[net] = np.array(idx)
    net_names = sorted(nets)

    rows = []
    for f in fs[:40]:
        x = L.load_cc200(f)
        if x is None:
            continue
        A, _ = paper6_A(x)
        W = gramian(A)
        Winv = np.linalg.inv(W)
        rec = {}
        for nm in net_names:
            xt = np.zeros(A.shape[0])
            xt[nets[nm]] = 1.0
            rec[f"E_{nm}"] = float(xt @ Winv @ xt)
            rec[f"n_{nm}"] = float(xt @ xt)          # = number of nodes in network
        rows.append(rec)
    E = pd.DataFrame(rows)
    print(f"  {'network':20s} {'E mean':>10s} {'||xT||^2':>9s} {'E/||xT||^2':>11s}")
    ratios = []
    for nm in net_names:
        m = E[f"E_{nm}"].mean()
        n = E[f"n_{nm}"].mean()
        r = m / n
        ratios.append(r)
        print(f"  {nm:20s} {m:10.3f} {n:9.1f} {r:11.3f}")
    print(f"\n  spread of E/||xT||^2 across networks = {max(ratios)/min(ratios):.2f}x")
    print("  (if ~1.0, energy is fully explained by the target's norm)")

    # direct test: regress E on ||xT||^2 across networks
    from scipy import stats
    ee = np.concatenate([E[f"E_{nm}"].values.ravel() for nm in net_names])
    nn = np.concatenate([E[f"n_{nm}"].values.ravel() for nm in net_names])
    r, pval = stats.pearsonr(nn, ee)
    print(f"\n  r(E, ||xT||^2) across all network-subject cells = {float(r):+.4f} (p={float(pval):.2e})")
    print(f"  R^2 = {float(r)**2:.4f}")
    print("  => if R^2 ~ 1, Paper 6's control energy is a restatement of how many")
    print("     nodes the target network contains.")

    # how much of E's variance is between-network vs between-subject?
    print("\n  variance decomposition of E:")
    long = []
    for nm in net_names:
        for v in E[f"E_{nm}"].values:
            long.append((nm, v))
    Lf = pd.DataFrame(long, columns=["net", "E"])
    gm = Lf.E.mean()
    ssb = ((Lf.groupby("net").E.mean() - gm) ** 2 *
           Lf.groupby("net").E.size()).sum()
    sst = ((Lf.E - gm) ** 2).sum()
    print(f"    between-network share of variance = {ssb/sst:.4f}")
    print(f"    within-network  share             = {1-ssb/sst:.4f}")

    print("\n" + "=" * 78)
    print("3  LEiDA: IS THE STATE SPACE REALLY ONE ANGLE?")
    print("=" * 78)
    err_v1, err_rank, ratios_carrier = [], [], []
    for f in fs[:30]:
        x = L.load_cc200(f)
        if x is None:
            continue
        X, ph = L.leida_v1(x)
        c, s = np.cos(ph), np.sin(ph)
        cc = (c * c).sum(1)
        ss = (s * s).sum(1)
        cs = (c * s).sum(1)
        tr = cc + ss
        dd = np.sqrt(np.maximum((cc - ss) ** 2 + 4 * cs * cs, 0))
        l1 = 0.5 * (tr + dd)
        phi = 0.5 * np.angle(np.exp(2j * ph).sum(1))
        pred = np.cos(ph - phi[:, None]) / np.sqrt(l1)[:, None]
        err_v1.append(float(np.abs(np.abs((X * pred).sum(1)) - 1).max()))
        E2 = np.exp(2j * ph)
        R2 = np.abs(E2.mean(1))
        err_rank.append(float(np.abs(l1 / ph.shape[1] - (1 + R2) / 2).max()))
    print(f"  v1 = cos(theta - phi)/sqrt(lam1):  max err = {max(err_v1):.3e}")
    print(f"  lam1/N = (1+R2)/2:                 max err = {max(err_rank):.3e}")
    print("  => the per-frame object is EXACTLY (phi, R2). Two numbers.")
    print("     The K-state label is a quantisation of phi alone.")

    res = {"sym_err": max(sym), "lam_range": (np.mean(lam_lo), np.mean(lam_hi)),
           "gramian_iso": np.mean(iso), "E_vs_norm_r2": float(r) ** 2,
           "between_net_var": ssb / sst, "v1_err": max(err_v1),
           "rank_identity_err": max(err_rank)}
    pd.Series(res).to_csv(os.path.join(OUT, "math_operator_facts.csv"))
    print(f"\nwrote {OUT}/math_operator_facts.csv")


if __name__ == "__main__":
    main()
