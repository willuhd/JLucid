"""Average and modal controllability (Paper 6 eqs. 9-10, D2-6)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.athena import load_network_membership


def average_controllability(a: np.ndarray) -> np.ndarray:
    """Node-level average controllability phi_i = sum_j v_ij^2 / (-2 lambda_j).

    A must be Hurwitz (strictly negative eigenvalues); see stabilize.py.
    """
    a = np.asarray(a, dtype=float)
    w, v = np.linalg.eigh(a)
    return (v**2 / (-2.0 * w)).sum(axis=1)


def modal_controllability(a: np.ndarray) -> np.ndarray:
    """Node-level modal controllability psi_i = sum_j (1 - e^{lambda_j}) v_ij^2."""
    a = np.asarray(a, dtype=float)
    w, v = np.linalg.eigh(a)
    return ((1.0 - np.exp(w)) * v**2).sum(axis=1)


def network_masks(membership: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    """Map each network group to a boolean ROI mask (AAL-90 state order).

    The membership CSV is indexed by AAL idx (1..90); the state vector is in
    the same AAL order, so ROI i in state space corresponds to idx i+1.
    """
    if membership is None:
        membership = load_network_membership()
    groups = ["Subcortical"] + list(membership.network_group.unique())
    groups = [g for g in groups if g != "Subcortical"] + ["Subcortical"]
    masks: dict[str, np.ndarray] = {}
    for g in dict.fromkeys(groups):
        idx = membership.loc[membership.network_group == g, "idx"].astype(int).values
        m = np.zeros(len(membership), dtype=bool)
        m[idx - 1] = True
        masks[g] = m
    return masks


def aggregate_controllability(
    phi: np.ndarray,
    psi: np.ndarray,
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate node metrics to individual (whole-brain mean) and network
    means (Yeo-7 + Subcortical).  Returns long-form rows.
    """
    masks = network_masks(membership)
    rows = []
    rows.append({"metric": "avg", "level": "individual", "group": "whole_brain",
                 "value": float(phi.mean())})
    rows.append({"metric": "modal", "level": "individual", "group": "whole_brain",
                 "value": float(psi.mean())})
    for g, m in masks.items():
        if m.sum() == 0:
            continue
        rows.append({"metric": "avg", "level": "network", "group": g,
                     "value": float(phi[m].mean())})
        rows.append({"metric": "modal", "level": "network", "group": g,
                     "value": float(psi[m].mean())})
    return pd.DataFrame(rows)
