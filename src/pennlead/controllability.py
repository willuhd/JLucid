"""Core network control: Eq.4, Eq.9, Eq.10 — strictly Paper 6.
Isolated copy for src-pennlead (identical to src-controllability).
Torch 2.13 CPU float64 eigh, Intel Mac.
Paper 6 Eq.4:  A = |FC| / (c + λmax(|FC|)) − I
Paper 6 Eq.9:  φ_i = Σ_j v_ij² / (−2 λ_j)
Paper 6 Eq.10: μ_i = Σ_j (1 − e^{λ_j}) v_ij²
"""
from __future__ import annotations
import numpy as np
import torch

def build_A(FC: np.ndarray, c: float = 1.0, use_abs: bool = True) -> np.ndarray:
    FC = np.asarray(FC, dtype=np.float64)
    M = np.abs(FC) if use_abs else FC
    M = (M + M.T)/2.0
    try:
        vals = np.linalg.eigvalsh(M)
    except np.linalg.LinAlgError:
        vals = np.linalg.eigvals(M).real
    lmax = float(np.max(vals))
    denom = c + lmax
    if denom == 0:
        denom = c + 1e-12
    A = M / denom - np.eye(M.shape[0], dtype=np.float64)
    A = (A + A.T)/2.0
    return A

def _eigh_torch(A: np.ndarray):
    tA = torch.as_tensor(A, dtype=torch.float64)
    vals, vecs = torch.linalg.eigh(tA)
    return vals, vecs

def avg_modal_controllability(A: np.ndarray):
    A = np.asarray(A, dtype=np.float64)
    A = (A + A.T)/2.0
    vals_t, vecs_t = _eigh_torch(A)
    vals = vals_t.numpy()
    vecs = vecs_t.numpy()
    vals = np.where(vals >= 0, -1e-9, vals)
    inv_neg2lam = 1.0 / (-2.0 * vals)
    avg = (vecs**2) @ inv_neg2lam
    one_minus_exp = 1.0 - np.exp(vals)
    modal = (vecs**2) @ one_minus_exp
    return avg, modal, vals, vecs

def subject_controllability(FC: np.ndarray, c: float = 1.0, use_abs: bool = True):
    A = build_A(FC, c=c, use_abs=use_abs)
    avg, modal, vals, vecs = avg_modal_controllability(A)
    return {
        "A": A,
        "eigvals": vals,
        "avg_node": avg,
        "modal_node": modal,
        "avg_individual": float(np.mean(avg)),
        "modal_individual": float(np.mean(modal)),
    }

def check_hurwitz(A: np.ndarray, tol: float = 1e-9) -> bool:
    vals = np.linalg.eigvalsh((A+A.T)/2)
    return bool(np.all(vals < tol))

def optimal_control_energy_stub():
    return {
        "note": "Control energy (Eq.11-14) deferred to Phase 2; min ∫[(xT−x)ᵀS(xT−x)+ρuᵀu]dt s.t. ẋ=Ax+Bu, B=I, T=1, dt=0.001, ρ=1, S diagonal",
        "target_states": "zero-init → one-hot per network",
        "solver": "future: scipy.integrate + LQR via torch",
    }
