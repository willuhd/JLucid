import numpy as np
import pytest

from jlucid.control.state_fc import compute_subject_state_fc, ledoit_fc, pearson_fc


def test_pearson_fc_recovers_planted_blocks():
    rng = np.random.default_rng(0)
    n = 6
    t = 200
    base = rng.normal(size=(t, 2))
    x = np.column_stack([base[:, 0], base[:, 0], base[:, 1], base[:, 1],
                         rng.normal(size=(t, 2))])
    fc = pearson_fc(x)
    assert fc.shape == (n, n)
    assert fc[0, 1] > 0.9
    assert fc[2, 3] > 0.9
    assert abs(fc[0, 4]) < 0.3


def test_min_frames_rule():
    rng = np.random.default_rng(1)
    tc = rng.normal(size=(40, 4))
    labels = np.array([0] * 15 + [1] * 25)
    state_fc, global_fc = compute_subject_state_fc(tc, labels, 2, min_frames=20)
    assert state_fc[0][0] is None and state_fc[0][1] == 15
    assert state_fc[1][0] is not None and state_fc[1][1] == 25
    assert global_fc[0] is not None


def test_censored_frames_excluded():
    rng = np.random.default_rng(2)
    tc = rng.normal(size=(30, 4))
    labels = np.zeros(30, dtype=int)
    mask = np.ones(30, dtype=bool)
    mask[5:10] = False
    state_fc, _ = compute_subject_state_fc(tc, labels, 1, mask=mask, min_frames=20)
    assert state_fc[0][1] == 25


def test_global_fc_none_with_few_frames():
    rng = np.random.default_rng(3)
    tc = rng.normal(size=(2, 4))
    labels = np.zeros(2, dtype=int)
    mask = np.array([True, False])
    _, global_fc = compute_subject_state_fc(tc, labels, 1, mask=mask, min_frames=1)
    assert global_fc[0] is None


def test_ledoit_fc_shaped():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(100, 5))
    fc = ledoit_fc(x)
    assert fc.shape == (5, 5)
