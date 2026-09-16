"""Control targets: network activation/suppression templates and state
centroid targets (step-2 Phase D, D2-9/D2-14)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import YEO_NETWORKS
from .controllability import network_masks


def _to_masks(membership: pd.DataFrame | None) -> dict[str, np.ndarray]:
    masks = network_masks(membership)
    # deterministic group order for reproducibility
    ordered = ["Subcortical"] + YEO_NETWORKS
    return {g: masks[g] for g in ordered if g in masks}


def activation_template(
    network_groups: list[str],
    membership: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """x_T = 1 on the union of the given networks, 0 elsewhere;
    S = diag(1) on those nodes (Paper 6 eq. 12; D2-9)."""
    masks = _to_masks(membership)
    sel = np.zeros(len(next(iter(masks.values()))), dtype=bool)
    for g in network_groups:
        if g not in masks:
            raise KeyError("unknown network " + g)
        sel |= masks[g]
    x_t = sel.astype(float)
    return x_t, np.diag(sel.astype(float))


def suppression_template(
    network_group: str,
    membership: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """x_T = -1 on the network (DMN suppression), 0 elsewhere; S on those nodes."""
    masks = _to_masks(membership)
    if network_group not in masks:
        raise KeyError("unknown network " + network_group)
    sel = masks[network_group]
    x_t = -sel.astype(float)
    return x_t, np.diag(sel.astype(float))


def centroid_template(centroid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """State-centroid target: unit max-abs normalization, S = I (D2-9)."""
    centroid = np.asarray(centroid, dtype=float).copy()
    m = np.max(np.abs(centroid)) if centroid.size else 0.0
    if m > 0:
        centroid /= m
    n = centroid.size
    return centroid, np.eye(n)


def cohort_mean_v1(v1_arrays: list[np.ndarray]) -> np.ndarray:
    """Mean V1 pattern across pooled frames, unit max-abs normalized."""
    if not v1_arrays:
        raise ValueError("empty V1 list")
    stacked = np.vstack(v1_arrays)
    mean_v = stacked.mean(axis=0)
    m = np.max(np.abs(mean_v))
    if m > 0:
        mean_v = mean_v / m
    return mean_v


def dominant_state_by_network(
    loadings_csv: Path,
    network: str,
) -> int:
    """State index whose dominant network is network; fall back to the state
    with the highest loading in that network (data-driven state mapping)."""
    df = pd.read_csv(loadings_csv).set_index("state")
    net_cols = ["Subcortical"] + YEO_NETWORKS
    net_cols = [c for c in net_cols if c in df.columns]
    if network not in df.columns:
        raise KeyError(f"network {network} not in loadings columns")
    dom = df[net_cols].idxmax(axis=1)
    candidates = df.index[dom == network]
    if len(candidates):
        return int(candidates[df.loc[candidates, network].argmax()])
    return int(df[network].idxmax())


def empirical_frame_match(v1: np.ndarray, target: np.ndarray) -> float:
    """Max |cosine| between a target template and observed V1 frames (Paper 6
    supplementary; D2-9 empirical frame-match)."""
    v1 = np.asarray(v1, dtype=float)
    target = np.asarray(target, dtype=float)
    tnorm = np.linalg.norm(target)
    if tnorm == 0:
        return 0.0
    vnorm = np.linalg.norm(v1, axis=1)
    denom = vnorm * tnorm
    cos = np.abs(v1 @ target) / np.maximum(denom, 1e-12)
    return float(np.max(cos[denom > 1e-12])) if np.any(denom > 1e-12) else 0.0
