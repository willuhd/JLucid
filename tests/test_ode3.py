import numpy as np
import torch

from jlucid.leida.phases import dpc_matrix, leading_two
from jlucid.ode.jlucid3.models import ThesisModel, flip_target, stay_loss
from jlucid.ode.jlucid3.shots import stay_index


def test_stay_field_unit_and_shapes():
    torch.manual_seed(0)
    m = ThesisModel(latent=8, window=5)
    b, w = 3, 5
    win = torch.nn.functional.normalize(torch.randn(b, w, 90), dim=-1)
    p0 = torch.zeros(b, 3); p0[:, 0] = 1
    now = win[:, -1]
    v2 = torch.nn.functional.normalize(torch.randn(b, 90), dim=-1)
    gap = torch.ones(b)
    out = m(win, p0, now, v2, gap, n_steps=2)
    assert out["v_stay"].shape == (b, 2, 90)
    assert torch.allclose(out["v_stay"].norm(dim=-1), torch.ones(b, 2), atol=1e-5)
    assert out["flip_logit"].shape == (b,)
    assert out["state_logits"].shape == (b, 3)


def test_stay_loss_zero_on_identity():
    v = torch.nn.functional.normalize(torch.randn(4, 90), dim=-1)
    assert float(stay_loss(v, v)) < 1e-6


def test_flip_target():
    v = torch.nn.functional.normalize(torch.randn(2, 90), dim=-1)
    assert not bool(flip_target(v[0:1], v[0:1])[0])
    assert bool(flip_target(v[0:1], -v[0:1])[0])


def test_stay_index_drops_switches_and_flips():
    t = 20
    v1 = np.zeros((t, 4), np.float32)
    v1[:, 0] = 1
    v1[10] = np.array([-1, 0, 0, 0], np.float32)  # flip vs 9 and 11
    lab = np.zeros(t, np.int64)
    lab[5:8] = 1  # switch block
    mask = np.ones(t, bool)
    bundle = {"a": {
        "v1": v1, "labels": lab, "mask": mask,
        "p": np.zeros((t, 3), np.float32),
        "v2": np.zeros_like(v1), "gap": np.zeros(t, np.float32),
    }}
    idx = stay_index(bundle, window=3, horizon=1)
    ts = {t for _, t in idx}
    assert 4 not in ts  # 4->5 switches 0->1
    assert 9 not in ts  # 9->10 is a flip
    assert 6 in ts      # inside the state-1 run


def test_leading_two_matches_eigh_order():
    ph = np.array([0.0, 0.4, 1.2, -0.3])
    v1, v2, l1, l2 = leading_two(dpc_matrix(ph))
    assert abs(v1 @ v2) < 1e-6
    assert abs(l1) >= abs(l2)
