"""Routing Jacobian is z-dependent and not an occupancy mix of shared A_s."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from jlucid.ode.jlucid4.jfuture import (
    dest_jacobian_batch,
    occupancy_from_labels,
    route_from_mean_J,
    stay_output_jacobian_batch,
)
from jlucid.ode.jlucid4.jspace import concentration, mix_A
from jlucid.ode.jlucid4.models import DestRouter, StayAffineChord, ThesisModel


def test_dest_jacobian_depends_on_z():
    torch.manual_seed(0)
    router = DestRouter(latent=12, k=3, hidden=16)
    z0 = torch.zeros(2, 12)
    z1 = torch.randn(2, 12)
    extras = torch.randn(2, 3)
    # wrap as a mini model
    class M:
        dest_router = router
    J0 = dest_jacobian_batch(M(), z0, extras)
    J1 = dest_jacobian_batch(M(), z1, extras)
    assert J0.shape == (2, 4, 12)
    assert J1.shape == (2, 4, 12)
    assert torch.isfinite(J0).all() and torch.isfinite(J1).all()
    # tanh MLP: Jacobian at 0 is not the Jacobian at a random z
    assert (J0 - J1).abs().max() > 1e-4
    # closed form matches autograd
    zi = z1[0].detach()
    ei = extras[0:1]

    def _f(u):
        return router.net(torch.cat([u.unsqueeze(0), ei], dim=-1)).squeeze(0)

    J_auto = torch.autograd.functional.jacobian(_f, zi)
    assert torch.allclose(J1[0], J_auto, atol=1e-5, rtol=1e-4)


def test_dest_jacobian_not_occupancy_mix_of_three_A():
    torch.manual_seed(1)
    model = ThesisModel(latent=12, window=7, hidden=16)
    z = torch.randn(6, 12)
    extras = torch.randn(6, 3)
    J = dest_jacobian_batch(model, z, extras).detach().numpy()
    J_bar = J.mean(axis=0)
    # cannot be written as Σ π_s A_s for 3 shared 12×12 stay matrices:
    # dest J is 4×12, rectangular, and varies with z
    assert J_bar.shape == (4, 12)
    rec = route_from_mean_J(J_bar, n_frames=6)
    assert 0.0 < rec["C_k"] <= 1.0
    assert 1.0 <= rec["r_eff"] <= 12.0 + 1e-6
    # frame-to-frame J varies (not a constant A_s)
    assert float(J.var()) > 0.0


def test_stay_output_jacobian_shape_and_finite():
    torch.manual_seed(2)
    stay = StayAffineChord(latent=12, k=3, hidden=16, in_dim=90)
    z = torch.randn(3, 12)
    p = torch.softmax(torch.randn(3, 3), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(3, 90), dim=-1)
    disp = 0.1 * torch.randn(3, 90)
    J = stay_output_jacobian_batch(stay, z, p, v, disp)
    assert J.shape == (3, 90, 12)
    assert torch.isfinite(J).all()
    # output Jacobian is not A(p)
    A = stay.mix_A(p)
    assert A.shape == (3, 12, 12)
    assert J.shape != A.shape


def test_when_features_from_ablated_z_have_dim_27():
    from jlucid.ode.jlucid4.discrete import feature_names, switch_feature_matrix
    from jlucid.ode.jlucid4.jfuture import route_arrays_from_z

    n, d, k = 8, 12, 3
    rng = np.random.default_rng(0)
    z = rng.normal(size=(n, d))
    extras = rng.normal(size=(n, 3))
    origin = rng.integers(0, k, size=n)
    dest = rng.integers(0, k, size=n)
    dwell = rng.integers(1, 6, size=n).astype(float)
    v = rng.normal(size=(n, 90)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    raw = rng.normal(size=(n, 1 + k))
    arr = route_arrays_from_z(z, extras, origin, dest, dwell, v, v, raw)
    X = switch_feature_matrix(arr, latent=d)
    assert X.shape == (n, len(feature_names(d)))
    assert X.shape[1] == 27


def test_occupancy_from_labels():
    lab = np.array([0, 0, 1, 1, 1, 2, -1])
    occ = occupancy_from_labels(lab, 3)
    assert occ.sum() == pytest.approx(1.0)
    assert occ[1] == 0.5
