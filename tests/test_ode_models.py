import numpy as np
import pytest
import torch

from jlucid.ode.jlucid2.models import (
    LatentODEModel,
    encode_no_ode,
    kl_divergence,
    model_loss,
    reparameterize,
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return LatentODEModel(latent=8, window=5, solver="euler")


def test_forward_shapes(model):
    v1 = torch.randn(25, 90)
    p = torch.zeros(25, 3)
    p[:, 0] = 1
    out = model(v1, p, sample=False)
    assert out["z_hat"].shape == (25, 8)
    assert out["v1_hat"].shape == (25, 90)
    assert out["state_logits"].shape == (25, 3)
    assert out["z_enc"].shape == (25, 8)
    assert torch.isfinite(out["z_hat"]).all()


def test_loss_masks_censored(model):
    v1 = torch.randn(25, 90)
    p = torch.zeros(25, 3)
    labels = torch.zeros(25, dtype=torch.long)
    mask = torch.ones(25, dtype=torch.bool)
    mask[10] = False
    wmask = torch.zeros(25, dtype=torch.bool)
    wmask[4:] = True
    out = model(v1, p, sample=False)
    ls = model_loss(model, out, v1, labels, mask, wmask)
    for k in ("total", "recon", "kl", "state_ce", "smooth", "consist"):
        assert torch.isfinite(ls[k])
        assert ls[k].item() >= 0


def test_kl_and_reparameterize():
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    assert kl_divergence(mu, logvar) == pytest.approx(0.0, abs=1e-5)
    z = reparameterize(mu, logvar)
    assert z.shape == (4, 8)


def test_encode_no_ode(model):
    v1 = torch.randn(20, 90)
    z = encode_no_ode(model, v1)
    assert z.shape == (20 - model.window + 1, 8)


def test_integrate_matches_manual_euler(model):
    v1 = torch.randn(10, 90)
    p = torch.zeros(10, 3)
    p[:, 2] = 1
    out = model(v1, p, sample=False)
    z0 = out["z_hat"][0]
    # manual one-step
    z1 = z0 + model.dynamics(z0, p[0])
    assert torch.allclose(out["z_hat"][1], z1, atol=1e-5)


def test_forward_shots_shapes_residual():
    torch.manual_seed(0)
    model = LatentODEModel(latent=8, window=5, solver="euler",
                           predict_residual=True, condition="p0",
                           unit_sphere=True)
    b, w, h = 4, 5, 3
    windows = torch.randn(b, w, 90)
    windows = windows / windows.norm(dim=-1, keepdim=True)
    p0 = torch.zeros(b, 3)
    p0[:, 1] = 1
    out = model.forward_shots(windows, p0, n_steps=h, sample=False)
    assert out["v1_hat"].shape == (h + 1, b, 90)
    assert out["z"].shape == (h + 1, b, 8)
    assert torch.allclose(out["v1_hat"].norm(dim=-1), torch.ones(h + 1, b), atol=1e-5)


def test_shot_loss_finite():
    from jlucid.ode.jlucid2.models import shot_loss
    torch.manual_seed(0)
    model = LatentODEModel(latent=8, window=5, solver="euler",
                           predict_residual=True)
    b, w, h = 3, 5, 2
    windows = torch.nn.functional.normalize(torch.randn(b, w, 90), dim=-1)
    p0 = torch.zeros(b, 3)
    p0[:, 0] = 1
    v1_fut = torch.nn.functional.normalize(torch.randn(b, h, 90), dim=-1)
    lab = torch.zeros(b, h, dtype=torch.long)
    out = model.forward_shots(windows, p0, n_steps=h, sample=False)
    ls = shot_loss(out, v1_fut, lab, windows[:, -1])
    for k in ("total", "pred", "now", "kl", "state_ce", "smooth", "cos1"):
        assert torch.isfinite(ls[k])


def test_integrate_steps_hold_p_batched():
    torch.manual_seed(0)
    model = LatentODEModel(latent=8, window=5, solver="euler")
    z0 = torch.randn(6, 8)
    p0 = torch.zeros(6, 3)
    p0[:, 2] = 1
    z = model.integrate_steps(z0, p0, n_steps=4, hold_p=True)
    assert z.shape == (5, 6, 8)
    z1 = z0 + model.dynamics(z0, p0)
    assert torch.allclose(z[1], z1, atol=1e-5)

