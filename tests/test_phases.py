import numpy as np
import pytest

from jlucid.leida.phases import (
    dpc_matrix,
    leading_eigenvector,
    leading_two,
    order_parameter,
    phase_from_bold,
    v1_series,
)


def test_phase_of_sine_is_linear():
    t = np.linspace(0, 10, 100)
    x = np.sin(2 * np.pi * 0.5 * t)[:, None]
    phase = phase_from_bold(x)
    assert phase.shape == (100, 1)
    # unwrapped phase should be close to linear in t
    unwrapped = np.unwrap(phase[:, 0])
    slope = np.polyfit(t, unwrapped, 1)[0]
    assert slope == pytest.approx(2 * np.pi * 0.5, rel=0.05)


def test_dpc_is_symmetric_cosine():
    ph = np.array([0.3, 1.1, -0.4])
    d = dpc_matrix(ph)
    assert d.shape == (3, 3)
    assert np.allclose(d, d.T)
    assert np.allclose(d[0, 1], np.cos(ph[0] - ph[1]))


def test_leading_eigenvector_majority_negative():
    # 2x2 with a known leading vector
    d = np.array([[1.0, 0.9], [0.9, 1.0]])
    v, lam = leading_eigenvector(d)
    assert lam == pytest.approx(1.9)
    assert np.sum(v > 0) <= v.size / 2  # majority-negative convention
    # eigenvector with both positive entries must be flipped
    assert np.all(v <= 0) or np.abs(v[0]) > np.abs(v[1])


def test_leading_two_orthogonal_and_signed():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 8))
    dpc = dpc_matrix(np.angle(x[0] + 1j * x[1]))
    v1, v2, l1, l2 = leading_two(dpc)
    assert abs(v1 @ v2) < 1e-6
    assert np.sum(v1 > 0) <= v1.size / 2
    assert np.sum(v2 > 0) <= v2.size / 2
    assert abs(l1) >= abs(l2)


def test_v1_series_shape_and_unit_norm():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8))
    # bandpass-ish smooth signal
    x = np.apply_along_axis(lambda s: np.convolve(s, np.ones(3) / 3, mode="same"), 0, x)
    v1, lam = v1_series(x)
    assert v1.shape == (40, 8)
    assert lam.shape == (40,)
    norms = np.linalg.norm(v1, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_order_parameter_bounds():
    phase = np.column_stack([np.zeros(20), np.zeros(20)])
    assert np.allclose(order_parameter(phase), 1.0)
    phase2 = np.column_stack([np.zeros(20), np.full(20, np.pi)])
    assert np.allclose(order_parameter(phase2), 0.0, atol=1e-12)
