import numpy as np
import pandas as pd
import pytest

from jlucid.control.targets import (
    activation_template,
    centroid_template,
    cohort_mean_v1,
    dominant_state_by_network,
    empirical_frame_match,
    suppression_template,
)


def _membership(n=90):
    rng = np.random.default_rng(0)
    groups = ["DMN", "FPN", "DAN", "SMN", "VIS", "VAN", "LIMBIC", "Subcortical"]
    return pd.DataFrame({"idx": np.arange(1, n + 1),
                         "network_group": [groups[i % len(groups)] for i in range(n)]})


def test_activation_template():
    mem = _membership()
    x_t, s = activation_template(["FPN", "DAN"], mem)
    assert set(np.unique(x_t)) <= {0.0, 1.0}
    assert x_t.sum() == (mem.network_group.isin(["FPN", "DAN"])).sum()
    assert np.allclose(s, np.diag(x_t))


def test_suppression_template():
    mem = _membership()
    x_t, s = suppression_template("DMN", mem)
    assert set(np.unique(x_t)) == {-1.0, 0.0}
    assert s.sum() == (mem.network_group == "DMN").sum()


def test_centroid_template():
    c = np.array([0.0, 2.0, -1.0])
    x_t, s = centroid_template(c)
    assert x_t[1] == pytest.approx(1.0) and x_t[2] == pytest.approx(-0.5)
    assert np.allclose(s, np.eye(3))


def test_cohort_mean_v1():
    v1s = [np.ones((5, 3)), 2 * np.ones((5, 3))]
    m = cohort_mean_v1(v1s)
    assert m[0] == pytest.approx(1.0)


def test_dominant_state_by_network():
    df = pd.DataFrame({
        "state": [0, 1],
        "DMN": [0.1, 0.4],
        "FPN": [0.3, 0.2],
    })
    df.to_csv("/tmp/loadings_test.csv", index=False)
    assert dominant_state_by_network("/tmp/loadings_test.csv", "DMN") == 1
    assert dominant_state_by_network("/tmp/loadings_test.csv", "FPN") == 0


def test_empirical_frame_match():
    v1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert empirical_frame_match(v1, np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert empirical_frame_match(v1, np.array([0.0, 1.0])) == pytest.approx(1.0)
