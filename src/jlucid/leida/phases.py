"""Hilbert phase -> dPC -> leading eigenvector (Paper 7, section 2.4)."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def phase_from_bold(x: np.ndarray) -> np.ndarray:
    """Analytic-signal phase of bandpassed BOLD, x: (T, N) -> phase (T, N)."""
    x = np.asarray(x, dtype=float)
    return np.angle(hilbert(x, axis=0))


def dpc_matrix(phase_t: np.ndarray) -> np.ndarray:
    """Dynamic phase coherence at one time point: (N,) -> (N, N)."""
    phase_t = np.asarray(phase_t, dtype=float)
    return np.cos(phase_t[:, None] - phase_t[None, :])


def leading_two(dpc: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Leading two eigenvectors of a symmetric dPC by |eigenvalue|.

    Both vectors get the Paper-7 majority-negative sign convention.
    Returns (v1, v2, lam1, lam2).
    """
    w, v = np.linalg.eigh(np.asarray(dpc, dtype=float))
    order = np.argsort(np.abs(w))[::-1]
    v1 = v[:, order[0]].copy()
    v2 = v[:, order[1]].copy()
    if np.sum(v1 > 0) > v1.size / 2:
        v1 = -v1
    if np.sum(v2 > 0) > v2.size / 2:
        v2 = -v2
    return v1, v2, float(w[order[0]]), float(w[order[1]])


def leading_eigenvector(dpc: np.ndarray) -> tuple[np.ndarray, float]:
    """Eigenvector of the symmetric dPC with the largest |eigenvalue|.

    Sign convention (Paper 7): if the majority of elements are positive, the
    vector is multiplied by -1 so the leading vector is majority-negative.
    """
    w, v = np.linalg.eigh(dpc)  # ascending eigenvalues
    idx = int(np.argmax(np.abs(w)))
    v1 = v[:, idx].copy()
    if np.sum(v1 > 0) > v1.size / 2:
        v1 = -v1
    return v1, float(w[idx])


def v1_series(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """V1(t) and eigenvalue(t) series for a (T, N) BOLD matrix."""
    phase = phase_from_bold(x)
    T, N = phase.shape
    V1 = np.empty((T, N), dtype=np.float64)
    lam = np.empty(T, dtype=np.float64)
    for t in range(T):
        V1[t], lam[t] = leading_eigenvector(dpc_matrix(phase[t]))
    return V1, lam


def order_parameter(phase: np.ndarray) -> np.ndarray:
    """Global Kuramoto order parameter S(t) = |(1/N) sum_n exp(i theta_n)|."""
    phase = np.asarray(phase, dtype=float)
    return np.abs(np.mean(np.exp(1j * phase), axis=1))
