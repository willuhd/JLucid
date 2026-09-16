"""AAL-90 -> Yeo-7 network labeling of LEiDA state centroids."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ENERGY_PRIMARY_SYSTEM, SYSTEM_GLOBAL, YEO_NETWORKS, system_name
from ..io.athena import load_network_membership


def state_network_loadings(centroids: np.ndarray, membership: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-state network loading: L_net = mean_{r in net} |V_c[r]| (plan Phase 8).

    Returns DataFrame indexed by network_group with one row per state.
    """
    if membership is None:
        membership = load_network_membership()
    groups = ["Subcortical"] + YEO_NETWORKS
    rows = []
    for c, v in enumerate(centroids):
        row = {"state": c}
        for g in groups:
            mask = membership.network_group == g
            row[g] = float(np.mean(np.abs(v[mask.values]))) if mask.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("state")


def label_states(loadings: pd.DataFrame, groups: list[str] | None = None) -> list[str]:
    """Name each state by its dominant network(s).

    A secondary network within 0.15 of the dominant loading is appended
    (e.g. 'FPN-DAN') to reflect mixed states.
    """
    if groups is None:
        groups = ["Subcortical"] + YEO_NETWORKS
    names = []
    for _, row in loadings.iterrows():
        vals = row[groups].astype(float)
        order = vals.sort_values(ascending=False)
        top = order.index[0]
        if len(order) > 1 and order.iloc[0] - order.iloc[1] < 0.15:
            names.append(f"{top}-{order.index[1]}")
        else:
            names.append(top)
    return names


def dmn_like_state_index(
    centroids: np.ndarray,
    membership: pd.DataFrame | None = None,
) -> int:
    """Index of the state with the largest mean |V| on DMN nodes.

    Same rule as ``scripts/11_transition_energy.py`` (``d_state``).
    """
    if membership is None:
        membership = load_network_membership()
    dmn = membership.network_group == "DMN"
    idx = membership.loc[dmn, "idx"].astype(int).to_numpy() - 1
    if idx.size == 0:
        raise ValueError("membership has no DMN rows")
    scores = np.abs(np.asarray(centroids, dtype=float)[:, idx]).mean(axis=1)
    return int(np.argmax(scores))


def primary_energy_system(
    transition: str,
    centroids: np.ndarray,
    membership: pd.DataFrame | None = None,
) -> str:
    """Headline A-system for a T1–T4 energy test (F1)."""
    spec = ENERGY_PRIMARY_SYSTEM[transition]
    if spec == SYSTEM_GLOBAL or spec == "global":
        return SYSTEM_GLOBAL
    if spec == "dmn_like":
        return system_name(dmn_like_state_index(centroids, membership))
    return spec
