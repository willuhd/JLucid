import numpy as np
import pytest

from jlucid.leida.networks import (
    dmn_like_state_index,
    label_states,
    primary_energy_system,
    state_network_loadings,
)


def test_state_network_loadings_shape():
    # tiny membership: 4 ROIs in 2 groups + subcortical group empty
    import pandas as pd

    mem = pd.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "label": ["a", "b", "c", "d"],
            "mask_code": [1, 2, 3, 4],
            "yeo_VIS": [0, 0, 0, 0],
            "yeo_SMN": [0, 0, 0, 0],
            "yeo_DAN": [0, 0, 0, 0],
            "yeo_VAN": [0, 0, 0, 0],
            "yeo_LIMBIC": [0, 0, 0, 0],
            "yeo_FPN": [0, 0, 0, 0],
            "yeo_DMN": [0, 0, 0, 0],
            "coverage": [0.9] * 4,
            "dominant": ["DMN", "DMN", "VIS", "VIS"],
            "network_group": ["DMN", "DMN", "VIS", "VIS"],
        }
    )
    centroids = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    loadings = state_network_loadings(centroids, membership=mem)
    assert loadings.shape == (2, 8)  # Subcortical + 7 Yeo networks
    assert loadings.loc[0, "DMN"] == pytest.approx(1.0)
    names = label_states(loadings)
    assert names == ["DMN", "VIS"]


def test_static_membership_assets_load():
    from jlucid.io.athena import load_aal_labels, load_network_membership

    labels = load_aal_labels()
    mem = load_network_membership()
    assert len(labels) == 90
    assert len(mem) == 90
    assert mem.network_group.nunique() >= 6


def test_dmn_like_state_and_primary_energy_system():
    from jlucid.io.athena import load_network_membership

    mem = load_network_membership()
    n = len(mem)
    cents = np.zeros((3, n))
    dmn_idx = mem.loc[mem.network_group == "DMN", "idx"].astype(int).values - 1
    cents[2, dmn_idx] = 1.0
    cents[0] = 0.01
    cents[1] = 0.02
    assert dmn_like_state_index(cents, mem) == 2
    assert primary_energy_system("T1", cents, mem) == "global"
    assert primary_energy_system("T2", cents, mem) == "state2"
    assert primary_energy_system("T3", cents, mem) == "state2"
    assert primary_energy_system("T4", cents, mem) == "global"
