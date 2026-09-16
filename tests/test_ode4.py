"""JLucid 4: identifiable A(p)+chord stay, packed dest/when, no invert."""

import numpy as np
import torch
import torch.nn.functional as F

from jlucid.ode.jlucid4.discrete import compose_route
from jlucid.ode.jlucid4.eval import half_step_on_index
from jlucid.ode.jlucid4.models import (
    StayAffineChord,
    ThesisModel,
    autograd_field_jacobian,
    horizon_weights,
    stay_loss_weighted,
    stay_safe_penalty,
)
from jlucid.ode.jlucid4.shots import dwell_index, stay_index


def test_stay_unit_and_shapes():
    torch.manual_seed(0)
    m = ThesisModel(latent=8, window=5, decode_mode="recursive")
    b, w = 3, 5
    win = F.normalize(torch.randn(b, w, 90), dim=-1)
    p0 = torch.zeros(b, 3)
    p0[:, 0] = 1
    out = m(win, p0, win[:, -1], win[:, -1], torch.ones(b), n_steps=3)
    assert out["v_stay"].shape == (b, 3, 90)
    assert torch.allclose(out["v_stay"].norm(dim=-1), torch.ones(b, 3), atol=1e-5)
    assert out["gamma"].shape == (b, 1)
    assert out["z0"].shape == (b, 8)
    assert out["state_logits"].shape == (b, 3)


def test_gamma_init_is_half():
    m = ThesisModel(latent=8, window=5, gamma_mode="learned")
    z = torch.randn(5, 8)
    assert torch.allclose(m.stay.gamma(z), torch.full((5, 1), 0.5), atol=1e-5)


def test_persist_init_is_identity_latent_step():
    stay = StayAffineChord(latent=12, k=3, small_decode_init=False)
    z = torch.randn(5, 12)
    p = torch.eye(3).repeat(2, 1)[:5]
    assert torch.allclose(stay.step(z, p), z)
    assert torch.allclose(stay.field_jacobian(z, p), torch.zeros(5, 12, 12))


def test_no_eps_residual_module():
    stay = StayAffineChord(latent=8, k=3)
    assert stay.eps == 0.0
    assert getattr(stay, "residual", None) is None
    z = torch.randn(2, 8)
    p = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with torch.no_grad():
        stay.A[0].copy_(0.1 * torch.eye(8))
        stay.b[0].zero_()
    expect = z[0] @ stay.A[0].T + stay.b[0]
    assert torch.allclose(stay.field(z, p)[0], expect, atol=1e-5)
    assert torch.allclose(stay.step(z, p)[0], z[0] + expect, atol=1e-5)


def test_jacobian_equals_A_and_is_finite():
    torch.manual_seed(1)
    d, k, b = 12, 3, 4
    stay = StayAffineChord(latent=d, k=k)
    with torch.no_grad():
        stay.A.copy_(0.05 * torch.randn(k, d, d))
        stay.b.copy_(0.01 * torch.randn(k, d))
    z = torch.randn(b, d)
    p = torch.zeros(b, k)
    p[0, 0] = p[1, 1] = p[2, 2] = 1
    p[3] = torch.tensor([0.2, 0.3, 0.5])
    J_auto = autograd_field_jacobian(stay.field, z, p)
    A_mix = stay.mix_A(p)
    assert J_auto.shape == (b, d, d)
    assert torch.isfinite(J_auto).all()
    assert not torch.allclose(J_auto, torch.zeros_like(J_auto))
    assert torch.allclose(J_auto, A_mix, atol=1e-5, rtol=1e-4)
    assert torch.allclose(stay.field_jacobian(z, p), A_mix, atol=1e-5, rtol=1e-4)


def test_recursive_matches_from_now_at_h1_then_diverges():
    torch.manual_seed(2)
    m = ThesisModel(latent=8, window=5, decode_mode="recursive")
    win = F.normalize(torch.randn(4, 5, 90), dim=-1)
    p0 = torch.zeros(4, 3)
    p0[:, 0] = 1
    rec = m(win, p0, win[:, -1], win[:, -1], torch.ones(4), n_steps=3,
            decode_mode="recursive")["v_stay"]
    now = m(win, p0, win[:, -1], win[:, -1], torch.ones(4), n_steps=3,
            decode_mode="from_now")["v_stay"]
    assert torch.allclose(rec[:, 0], now[:, 0], atol=1e-5)
    assert (rec[:, 2] - now[:, 2]).abs().mean() > 1e-6


def test_dec_zero_gamma_half_is_half_step():
    stay = StayAffineChord(latent=4, k=3, in_dim=6, gamma_mode="fixed",
                           gamma_value=0.5, decode_mode="recursive",
                           small_decode_init=False)
    with torch.no_grad():
        for p in stay.decoder.parameters():
            p.zero_()
    z0 = torch.zeros(2, 4)
    p0 = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    v = F.normalize(torch.randn(2, 6), dim=-1)
    prev = F.normalize(torch.randn(2, 6), dim=-1)
    hat = stay.rollout(z0, p0, v, n_steps=1, last_disp=v - prev)[:, 0]
    expect = F.normalize(v + 0.5 * (v - prev), dim=-1)
    assert torch.allclose(hat, expect, atol=1e-5)


def test_stay_does_not_invert():
    stay = StayAffineChord(latent=4, k=3, in_dim=6, gamma_mode="none",
                           small_decode_init=False)
    with torch.no_grad():
        for p in stay.decoder.parameters():
            p.zero_()
    z0 = torch.zeros(3, 4)
    hat = stay.rollout(z0, torch.eye(3), F.normalize(torch.randn(3, 6), dim=-1),
                       n_steps=1)[:, 0]
    v = F.normalize(torch.randn(3, 6), dim=-1)
    hat = stay.rollout(z0, torch.eye(3), v, n_steps=1)[:, 0]
    assert torch.allclose(hat, v, atol=1e-5)
    assert (hat * v).sum(-1).min() > 0.99


def test_horizon_weights_and_safe_penalty():
    w = horizon_weights(3)
    assert torch.allclose(w.sum(), torch.tensor(1.0))
    now = F.normalize(torch.randn(6, 90), dim=-1)
    assert float(stay_safe_penalty(now.unsqueeze(1), now.unsqueeze(1), now)) < 1e-6
    assert float(stay_safe_penalty((-now).unsqueeze(1), now.unsqueeze(1), now)) > 0.5
    v = F.normalize(torch.randn(4, 3, 90), dim=-1)
    assert float(stay_loss_weighted(v, v, w)) < 1e-6


def test_dwell_index_still_rejects_flips():
    t = 16
    v1 = np.zeros((t, 4), np.float32)
    v1[:, 0] = 1.0
    v1[8] = np.array([-1, 0, 0, 0], np.float32)
    bundle = {"b": {
        "v1": v1, "labels": np.zeros(t, np.int64), "mask": np.ones(t, bool),
        "p": np.zeros((t, 3), np.float32), "v2": np.zeros_like(v1),
        "gap": np.zeros(t, np.float32),
    }}
    stay = {tt for _, tt in stay_index(bundle, window=3, horizon=3)}
    dwell = {tt for _, tt in dwell_index(bundle, window=3, horizon=3)}
    assert 6 in stay and 6 not in dwell
    assert 10 in dwell


def test_compose_route_is_label_not_invert():
    pred = compose_route(
        np.array([0, 1, 2, 1]), np.array([1, 2, 0, 0]),
        np.array([False, True, False, True]))
    assert pred.tolist() == [0, 2, 2, 0]


def test_half_step_baseline_matches_unit_chord():
    v1 = np.zeros((8, 3), np.float32)
    for i in range(8):
        ang = 0.1 * i
        v1[i] = np.array([np.cos(ang), np.sin(ang), 0.0], np.float32)
    out = half_step_on_index({"s": {"v1": v1}}, [("s", 3)], horizon=1, gamma=0.5)
    cur, prev = v1[3], v1[2]
    expect = cur + 0.5 * (cur - prev)
    expect = expect / np.linalg.norm(expect)
    assert abs(out["model"][1][0] - float(expect @ v1[4])) < 1e-5


def test_stay_params_exclude_dest_and_when():
    m = ThesisModel(latent=8, window=5)
    stay_ids = {id(p) for p in m.stay_params()}
    for p in list(m.dest_encoder.parameters()) + list(m.dest_router.parameters()) + list(m.when.parameters()):
        assert id(p) not in stay_ids
