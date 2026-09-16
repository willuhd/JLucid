"""Stabilized system matrix from FC (Paper 6 eq. 4, D2-3)."""
from __future__ import annotations

import numpy as np

from ..config import C_STAB


def stabilize(fc: np.ndarray, c: float = C_STAB) -> np.ndarray:
    """A = |FC| / (c + lambda_max(|FC|)) - I (Paper 6 eq. 4).

    Spectral normalization of |FC| followed by a left shift by I guarantees
    all eigenvalues are strictly negative (Hurwitz), so the infinite-horizon
    controllability Gramian converges.
    """
    fc = np.asarray(fc, dtype=float)
    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError(f"expected square matrix, got {fc.shape}")
    a = np.abs(fc)
    lam_max = float(np.linalg.eigvalsh(a).max())
    n = fc.shape[0]
    return a / (c + lam_max) - np.eye(n)


def assert_hurwitz(a: np.ndarray, tol: float = 1e-8) -> None:
    """Raise if any real eigenvalue is not strictly negative."""
    ev = np.linalg.eigvals(a).real
    if ev.max() >= -tol:
        raise ValueError(f"system matrix not Hurwitz (max real eig = {ev.max():.3e})")


def hurwitz_shift(a: np.ndarray, margin: float = 1e-3) -> np.ndarray:
    """Make a square matrix Hurwitz by symmetrize + spectral shift.

    ``S = (A + Aᵀ) / 2``, then ``A_h = S − (λ_max(S) + margin) I``.
    All eigenvalues of ``A_h`` are real and strictly less than 0.

    This is **not** Paper 6's FC ``stabilize()`` of ``|FC|``.  Do not call
    ``stabilize`` here: J-space ``A_J`` is a projected Jacobian, not an
    absolute functional-connectivity matrix.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"expected square matrix, got {a.shape}")
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")
    n = a.shape[0]
    sym = 0.5 * (a + a.T)
    lam_max = float(np.linalg.eigvalsh(sym).max())
    return sym - (lam_max + float(margin)) * np.eye(n)
