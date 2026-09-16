import numpy as np
import torch

from jlucid.ode.jlucid23.eval import cosine_rows, hard_compose, regime
from jlucid.ode.jlucid23.jacobians import concentration, effective_rank, j_dyn
from jlucid.ode.jlucid23.models import (
    StaySplitModel,
    flip_target,
    split_loss,
    stay_mask,
    velocity_persist,
)


def test_velocity_persist_continues_displacement():
    v = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)
    prev = torch.nn.functional.normalize(torch.tensor([[0.9, 0.1, 0.0]]), dim=-1)
    vel = velocity_persist(v, prev)
    assert vel.shape == (1, 3)
    assert torch.allclose(vel.norm(dim=-1), torch.ones(1), atol=1e-5)
    # prev→now moved toward −y; the velocity persist continues that heading
    assert float(vel[0, 1]) < 0.0


def test_forward_unit_and_hard_gate():
    torch.manual_seed(0)
    m = StaySplitModel(latent=8, window=5)
    windows = torch.nn.functional.normalize(torch.randn(6, 5, 90), dim=-1)
    now = windows[:, -1]
    p0 = torch.zeros(6, 3)
    p0[:, 1] = 1
    out = m(windows, now, p0, hard_tau=2.0, last_cos_max=1.0)
    assert torch.allclose(out["v1_hat"], out["stay"], atol=1e-5)
    out2 = m(windows, now, p0, hard_tau=-1.0, last_cos_max=1.0)
    assert torch.allclose(out2["v1_hat"], out2["invert"], atol=1e-5)
    assert torch.allclose(out["stay"].norm(dim=-1), torch.ones(6), atol=1e-5)
    assert torch.allclose(out["z_dot"], out["z1"] - out["z0"], atol=1e-6)


def test_stay_mask_and_loss_finite():
    now = torch.nn.functional.normalize(torch.randn(5, 90), dim=-1)
    nxt = now.clone()
    nxt[0] = -now[0]
    lab0 = torch.zeros(5, dtype=torch.long)
    lab1 = torch.zeros(5, dtype=torch.long)
    lab1[1] = 1
    assert bool(flip_target(now, nxt)[0])
    sm = stay_mask(now, nxt, lab0, lab1)
    assert not bool(sm[0]) and not bool(sm[1]) and bool(sm[2])
    m = StaySplitModel(latent=8, window=5)
    windows = now.unsqueeze(1).repeat(1, 5, 1)
    p0 = torch.zeros(5, 3)
    p0[:, 0] = 1
    out = m(windows, now, p0)
    ls = split_loss(out, nxt, now, lab0, lab1)
    for k in ("total", "stay_fit", "pull", "bce", "vel_fit"):
        assert torch.isfinite(ls[k])


def test_j_dyn_shape_and_finite():
    torch.manual_seed(0)
    m = StaySplitModel(latent=8, window=5)
    z = torch.randn(3, 8)
    p = torch.zeros(3, 3)
    p[:, 2] = 1
    J = j_dyn(m, z, p)
    assert J.shape == (3, 8, 8)
    assert torch.isfinite(J).all()


def test_hard_compose_and_concentration():
    stay = np.eye(3)[:2]
    vnow = stay.copy()
    g = np.array([0.1, 0.9])
    last = np.array([0.8, -0.6])
    yhat, called = hard_compose(stay, vnow, g, last, tau=0.5, last_cos_max=0.0)
    assert not called[0] and called[1]
    assert np.allclose(yhat[1], -vnow[1])
    s = np.array([4.0, 2.0, 1.0, 1.0])
    assert concentration(s, 2) == 20.0 / 22.0
    assert effective_rank(s) > 1.0
    fut = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    r = regime(vnow, fut, np.array([0, 0]), np.array([0, 1]))
    assert list(r) == ["stay", "flip"]


def test_cosine_rows():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(cosine_rows(a, a), [1.0, 1.0])
