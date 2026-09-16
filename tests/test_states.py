import numpy as np
import pytest

from jlucid.leida.states import (
    lifetime,
    occurrence_probability,
    state_metrics_subject,
    tpm,
    tpm_export,
)


def test_occurrence_probability():
    labels = np.array([0, 0, 1, 1, 1, 0, -1, -1, 0])
    occ = occurrence_probability(labels, 2)
    # 7 valid frames: 3 zeros + 3 ones + 1 zero = 4 / 7
    assert occ[0] == pytest.approx(4 / 7)
    assert occ[1] == pytest.approx(3 / 7)


def test_lifetime_runs():
    labels = np.array([0, 0, 1, 1, 1, 0, -1, 0, 0])
    life = lifetime(labels, 2)
    assert life[0] == pytest.approx(np.mean([2, 1, 2]))
    assert life[1] == pytest.approx(3)


def test_tpm_normalized_and_censored():
    labels = np.array([0, 1, 1, -1, 1, 0])
    tp = tpm(labels, 2)
    # observed transitions: 0->1, 1->1, (gap), 1->0
    assert tp[0, 1] == 1.0
    assert tp[1, 0] == pytest.approx(0.5)
    assert tp[1, 1] == pytest.approx(0.5)
    assert np.allclose(tp.sum(axis=1)[tp.sum(axis=1) > 0], 1.0)


def test_state_metrics_subject():
    labels = np.array([0, 0, 1, 1])
    phase = np.zeros((4, 3))
    met = state_metrics_subject(labels, phase, 2)
    assert met["n_frames"] == 4
    assert met["order_param_mean"] == pytest.approx(1.0)
    assert met["n_states_visited"] == 2


def test_tpm_export_keeps_true_zeros():
    # visited 0, never went to 1
    labels = np.array([0, 0, 0])
    tp = tpm_export(labels, 2)
    assert tp[0, 0] == pytest.approx(1.0)
    assert tp[0, 1] == pytest.approx(0.0)
    # 0 -> 1 -> 1: visited 1, never 1 -> 0
    labels = np.array([0, 1, 1])
    tp = tpm_export(labels, 2)
    assert tp[1, 1] == pytest.approx(1.0)
    assert tp[1, 0] == pytest.approx(0.0)


def test_tpm_export_nan_if_from_state_never_visited():
    labels = np.array([0, 0, 0])
    tp = tpm_export(labels, 2)
    assert tp[0, 0] == pytest.approx(1.0)
    assert np.isnan(tp[1]).all()


def test_tpm_export_nan_if_no_outgoing_pair():
    # state 1 occurs once at the end: occurrence > 0 but TPM undefined
    labels = np.array([0, 0, 1])
    tp = tpm_export(labels, 2)
    assert tp[0, 0] == pytest.approx(0.5)
    assert tp[0, 1] == pytest.approx(0.5)
    assert np.isnan(tp[1]).all()


def test_tpm_export_censoring_breaks_chain():
    labels = np.array([0, 1, -1, 1])
    tp = tpm_export(labels, 2)
    assert tp[0, 1] == pytest.approx(1.0)
    # the isolated trailing 1 has no outgoing pair
    assert np.isnan(tp[1]).all()
