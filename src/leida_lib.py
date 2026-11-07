"""Turn-3 shared library: LEiDA operator, sign gauges, projective clustering.

Every function here is validated against the project's own loop implementation
(scripts/denovo_refit.py, scripts/e123a_labels.py) by `python -m scripts.t3_00_lib`.

Conventions
-----------
ts      : (T, N) raw parcel timeseries
X       : (T, N) unit leading eigenvectors of the rank-2 phase-coherence matrix
ph      : (T, N) Hilbert phases of the per-parcel z-scored signal

Gauges (all are deterministic choices of one representative from {x, -x}):
  'argmax' : flip so the largest-|entry| component is positive  (project code)
  'majneg' : flip so the majority of entries are negative       (Paper 7, L480-484)
  'none'   : leave the closed-form sign as produced

Projective clustering operates on RP^(N-1) with chordal distance
  d(x,y)^2 = 2 - 2|<x,y>| ,
which is invariant under x -> -x and is therefore the only gauge-free way to
partition these states.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ADHD200 = os.path.join(ROOT, "data", "adhd200")


# ---------------------------------------------------------------- operator --
def hilbert_phase(z: np.ndarray) -> np.ndarray:
    """FFT Hilbert phase, identical convention to scripts/e123a_labels.py:17-22."""
    T = z.shape[0]
    F = np.fft.fft(z, axis=0)
    h = np.zeros(T)
    h[0] = 1.0
    if T % 2 == 0:
        h[T // 2] = 1.0
        h[1:T // 2] = 2.0
    else:
        h[1:(T + 1) // 2] = 2.0
    return np.angle(np.fft.ifft(F * h[:, None], axis=0))


def leida_v1(ts: np.ndarray):
    """Vectorised LEiDA leading eigenvector.

    Returns (X, ph) with X the unit leading eigenvector of
    C(t) = cos(ph_i - ph_j) per frame, obtained from the exact 2x2 Gram matrix
    G = [[c.c, c.s], [c.s, s.s]] (rank-2 identity, T1 in docs/THEORY_MEASUREMENT_LAW.md).
    """
    T, N = ts.shape
    sd = ts.std(0)
    z = (ts - ts.mean(0)) / np.where(sd == 0, 1e-12, sd)
    ph = hilbert_phase(z)
    c = np.cos(ph)
    s = np.sin(ph)
    cc = (c * c).sum(1)
    ss = (s * s).sum(1)
    cs = (c * s).sum(1)
    disc = np.sqrt(np.maximum((cc - ss) ** 2 + 4.0 * cs * cs, 0.0))
    l1 = 0.5 * (cc + ss + disc)
    e0 = cs.copy()
    e1 = l1 - cc
    zero = np.abs(cs) < 1e-14
    if zero.any():
        take_c = zero & (cc >= ss)
        take_s = zero & ~(cc >= ss)
        e0[take_c] = 1.0
        e1[take_c] = 0.0
        e0[take_s] = 0.0
        e1[take_s] = 1.0
    nrm = np.hypot(e0, e1)
    nrm[nrm == 0] = 1.0
    e0 /= nrm
    e1 /= nrm
    X = e0[:, None] * c + e1[:, None] * s
    nx = np.linalg.norm(X, axis=1)
    nx[nx == 0] = 1.0
    return X / nx[:, None], ph


def leida_v1_loop(ts: np.ndarray) -> np.ndarray:
    """Verbatim loop form from scripts/denovo_refit.py:11-26 (validation only)."""
    T, N = ts.shape
    z = (ts - ts.mean(0)) / np.where(ts.std(0) == 0, 1e-12, ts.std(0))
    ph = hilbert_phase(z)
    out = np.empty((T, N))
    for t in range(T):
        c = np.cos(ph[t])
        s = np.sin(ph[t])
        G = np.array([[c @ c, c @ s], [c @ s, s @ s]])
        _, V = np.linalg.eigh(G)
        v = V[:, -1]
        x = v[0] * c + v[1] * s
        n = np.linalg.norm(x)
        if n > 0:
            x = x / n
        k = int(np.argmax(np.abs(x)))
        if x[k] < 0:
            x = -x
        out[t] = x
    return out


# ------------------------------------------------------------------ gauges --
def apply_gauge(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return X.copy()
    if mode == "argmax":
        k = np.argmax(np.abs(X), axis=1)
        sgn = np.where(X[np.arange(len(X)), k] < 0, -1.0, 1.0)
        return X * sgn[:, None]
    if mode == "majneg":
        pos = (X > 0).sum(1) > (X.shape[1] / 2.0)
        sgn = np.where(pos, -1.0, 1.0)
        return X * sgn[:, None]
    raise ValueError(mode)


def gauge_disagreement(X: np.ndarray) -> float:
    """Fraction of frames where the two published gauges pick opposite signs."""
    a = apply_gauge(X, "argmax")
    b = apply_gauge(X, "majneg")
    return float((np.sign(np.einsum("ij,ij->i", a, b)) < 0).mean())


# --------------------------------------------------------------- clustering --
def spherical_kmeans(X, k, n_init=10, seed=0, sample=None):
    """Plain k-means on unit vectors (what both papers do, after gauge fixing)."""
    from sklearn.cluster import KMeans
    fit = X if sample is None else X[sample]
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(fit)
    lab = ((X[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1).argmin(1)
    return lab, km.cluster_centers_, km.inertia_


def projective_kmeans(X, k, n_init=10, seed=0, iters=200):
    """k-means on RP^(N-1): objective sum_i max_j |<x_i, c_j>|.

    Gauge-free. Equivalent to k-means with chordal distance on projective space.
    """
    n, d = X.shape
    best = (-np.inf, None, None)
    for init in range(n_init):
        rng = np.random.default_rng(seed + init)
        C = X[rng.choice(n, k, replace=False)].copy()
        for _ in range(iters):
            S = X @ C.T
            A = np.abs(S)
            lab = A.argmax(1)
            sgn = np.sign(S[np.arange(n), lab])
            newC = np.empty_like(C)
            for j in range(k):
                m = lab == j
                if m.sum() == 0:
                    newC[j] = X[rng.integers(n)]
                else:
                    v = (X[m] * sgn[m, None]).sum(0)
                    nv = np.linalg.norm(v)
                    newC[j] = v / nv if nv > 0 else C[j]
            if np.allclose(newC, C, atol=1e-10):
                C = newC
                break
            C = newC
        S = X @ C.T
        A = np.abs(S)
        lab = A.argmax(1)
        obj = float(A[np.arange(n), lab].sum())
        if obj > best[0]:
            best = (obj, C.copy(), lab.copy())
    obj, C, lab = best
    return lab, C, obj


def mode_spectrum(X: np.ndarray, n_blocks: int = 20):
    """Gauge-invariant mode count: eigenvalues of E[x x^T].

    For frames drawn from M directions c_m with weights pi_m,
    E[xx^T] ~ sum_m pi_m c_m c_m^T, so the eigenvalues estimate the projective
    occupancy of each mode. tr = 1 because x is unit.
    """
    M = (X.T @ X) / len(X)
    lam = np.sort(np.linalg.eigvalsh(M))[::-1]
    pr = (lam.sum() ** 2) / (lam ** 2).sum()
    return lam, float(pr)


# ------------------------------------------------------------------- loading --
def cc200_paths(substr: str = "sfnwmrda") -> list[str]:
    fs = sorted(glob.glob(os.path.join(ADHD200, "*", "*", "*cc200_TCs.1D")))
    if substr:
        fs = [f for f in fs if substr in os.path.basename(f)]
    return fs


def load_cc200(path: str, min_T: int = 120):
    try:
        x = pd.read_csv(path, sep=r"\s+", skiprows=1, usecols=range(2, 192),
                        header=None).to_numpy(float)
    except Exception:
        return None
    if x.ndim != 2 or x.shape[1] != 190 or x.shape[0] < min_T:
        return None
    if not np.isfinite(x).all() or x.std(0).min() == 0:
        return None
    return x


def subject_of(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


# ---------------------------------------------------------------- validation --
def _validate():
    fs = cc200_paths()[:5]
    errs = []
    dis = []
    for f in fs:
        x = load_cc200(f)
        if x is None:
            continue
        X, _ = leida_v1(x)
        Xl = leida_v1_loop(x)
        errs.append(float(np.abs(np.abs((X * Xl).sum(1)) - 1.0).max()))
        dis.append(gauge_disagreement(X))
    print(f"validated on {len(errs)} runs")
    print(f"  max |cos(v1_vectorised, v1_loop) - 1| = {max(errs):.3e}   (falsifier: >1e-10)")
    print(f"  gauge disagreement per run: {np.round(dis, 3)}")
    assert max(errs) < 1e-10, "vectorised operator does not match the loop implementation"
    print("  OPERATOR VALIDATED")


if __name__ == "__main__":
    _validate()
