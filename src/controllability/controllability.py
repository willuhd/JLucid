"""Core network control: Eq.4, Eq.9, Eq.10 — strictly paper 6.
Implements on torch 2.13 CPU with float64 eigh for stability (Intel Mac).

Paper 6 Eq.4:  A = |FC| / (c + λmax(|FC|)) − I
Paper 6 Eq.9:  φ_i = Σ_j v_ij² / (−2 λ_j)            (average controllability)
Paper 6 Eq.10: μ_i = Σ_j (1 − e^{λ_j}) v_ij²         (modal controllability)
All eigenvalues λ_j of A are <0 (Hurwitz), so denominators are well-defined.
"""
from __future__ import annotations
import numpy as np
import torch

def build_A(FC: np.ndarray, c: float = 1.0, use_abs: bool = True) -> np.ndarray:
    """Build stabilized system matrix A per Eq.4.
    Args:
        FC: symmetric N×N (Pearson correlation)
        c: stabilization constant (paper default 1.0; sweep 0.5,1,2,5)
        use_abs: if True use |FC|, else raw FC (robustness per paper 6 §2.7)
    Returns:
        A: symmetric N×N with all eigenvalues <0
    """
    FC = np.asarray(FC, dtype=np.float64)
    M = np.abs(FC) if use_abs else FC
    # M is symmetric; largest eigenvalue (real) — use float64
    # For robustness with non-symmetric due to norm variants, symmetrize:
    M = (M + M.T)/2.0
    # eigvalsh is stable for symmetric
    try:
        vals = np.linalg.eigvalsh(M)
    except np.linalg.LinAlgError:
        vals = np.linalg.eigvals(M).real
    lmax = float(np.max(vals))
    # Avoid division by zero if lmax is 0 (should not happen after abs)
    denom = c + lmax
    if denom == 0:
        denom = c + 1e-12
    A = M / denom - np.eye(M.shape[0], dtype=np.float64)
    # Ensure symmetry after division
    A = (A + A.T)/2.0
    return A

def _eigh_torch(A: np.ndarray):
    """Eigendecompose symmetric A via torch.linalg.eigh (float64, CPU)."""
    tA = torch.as_tensor(A, dtype=torch.float64)
    # eigh returns vals ascending, vecs columns are eigenvectors
    vals, vecs = torch.linalg.eigh(tA)  # vals (N,), vecs (N,N)
    return vals, vecs  # torch tensors

def avg_modal_controllability(A: np.ndarray):
    """Compute node-level average and modal controllability per Eq.9/10.
    Returns:
        avg:  (N,) average controllability φ_i
        modal: (N,) modal controllability μ_i
        vals: (N,) eigenvalues λ_j (numpy)
        vecs: (N,N) eigenvectors V (numpy, columns = eigenvectors)
    """
    A = np.asarray(A, dtype=np.float64)
    A = (A + A.T)/2.0
    vals_t, vecs_t = _eigh_torch(A)
    vals = vals_t.numpy()  # (N,)
    vecs = vecs_t.numpy()  # (N,N) columns eigenvectors
    # Numerical guard: ensure eigenvalues are negative (Hurwitz)
    # If any >=0 due to numerical edge, clamp to small negative
    vals = np.where(vals >= 0, -1e-9, vals)
    # avg: sum_j v_ij^2 / (-2 λ_j)
    # vecs[i,j] is element i row, j col
    inv_neg2lam = 1.0 / (-2.0 * vals)  # (N,)
    avg = (vecs**2) @ inv_neg2lam  # (N,) = Σ_j v_ij² * inv
    # modal: sum_j (1 - exp(λ_j)) * v_ij²
    one_minus_exp = 1.0 - np.exp(vals)  # (N,)
    modal = (vecs**2) @ one_minus_exp
    return avg, modal, vals, vecs

def subject_controllability(FC: np.ndarray, c: float = 1.0, use_abs: bool = True):
    """One-shot per subject: FC → A → avg/modal (node & individual mean).
    Returns dict with node-level and scalar means.
    """
    A = build_A(FC, c=c, use_abs=use_abs)
    avg, modal, vals, vecs = avg_modal_controllability(A)
    return {
        "A": A,
        "eigvals": vals,
        "avg_node": avg,        # (N,)
        "modal_node": modal,    # (N,)
        "avg_individual": float(np.mean(avg)),
        "modal_individual": float(np.mean(modal)),
    }

def check_hurwitz(A: np.ndarray, tol: float = 1e-9) -> bool:
    vals = np.linalg.eigvalsh((A+A.T)/2)
    return bool(np.all(vals < tol))

# Optional: control energy stub (deferred per advisor)
def optimal_control_energy_stub():
    """Placeholder for Eq.11-14 — not executed in Phase 1.
    Advisor says: '先看看可控性指标，先不管控制能量吧'
    Returns a description for future implementation.
    """
    return {
        "note": "Control energy (Eq.11-14) deferred to Phase 2; structure: min ∫[(xT−x)ᵀS(xT−x)+ρuᵀu]dt s.t. ẋ=Ax+Bu, B=I, T=1, dt=0.001, ρ=1, S diagonal on target network",
        "target_states": "zero-init → one-hot per network (VIS/SOM/DAN/SAL/LIM/FPN/DMN)",
        "solver": "future: scipy.integrate + LQR via torch or nistats",
    }
