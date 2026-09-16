"""Unit tests for JLucid 4 J-space math (array-level; no full-cohort tables)."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from jlucid.control.stabilize import hurwitz_shift, stabilize
from jlucid.ode.jlucid4.jspace import (
    apply_ablate,
    apply_swap,
    apply_zero,
    assemble_subject,
    concentration,
    covariates_from_table,
    decoder_jacobian,
    dmn_contamination,
    effective_rank,
    h3_group_stats,
    jspace_coords,
    map_axes_decoder,
    mix_A,
    pattern_network_loadings,
    project_jacobian,
    random_orthonormal,
    svd_right,
    swap_donors,
)
from jlucid.ode.jlucid4.models import StayAffineChord


def test_concentration_and_effective_rank_known_spectrum():
    s = np.array([4.0, 2.0, 1.0, 1.0])
    assert concentration(s, 2) == pytest.approx(20.0 / 22.0)
    assert concentration(s, 3) == pytest.approx(21.0 / 22.0)
    assert effective_rank(s) == pytest.approx(64.0 / 22.0)
    assert np.isnan(concentration(np.zeros(12), 3))
    assert np.isnan(effective_rank(np.zeros(12)))
    one = np.array([3.0, 0.0, 0.0, 0.0])
    assert concentration(one, 3) == pytest.approx(1.0)
    assert effective_rank(one) == pytest.approx(1.0)
    flat = np.ones(12)
    assert effective_rank(flat) == pytest.approx(12.0)
    assert 0.0 < concentration(s, 3) <= 1.0
    assert 1.0 <= effective_rank(s) <= 4.0


def test_svd_right_vector_convention_square_and_rectangular():
    rng = np.random.default_rng(0)
    v = rng.normal(size=12)
    v = v / np.linalg.norm(v)
    u11 = rng.normal(size=11)
    u11 = u11 / np.linalg.norm(u11)
    J_rect = 3.0 * np.outer(u11, v)
    U, s, V = svd_right(J_rect)
    assert U.shape == (11, 11)
    assert V.shape == (12, 11)
    assert abs(float(V[:, 0] @ v)) == pytest.approx(1.0, abs=1e-6)
    z = rng.normal(size=12)
    V_k = V[:, :3]
    zJ = jspace_coords(z, V_k)
    assert zJ.shape == (3,)
    assert np.allclose(zJ, V_k.T @ z)
    # reconstructed projection lives in the right space
    z_hat = V_k @ zJ
    assert np.allclose(z_hat, V_k @ (V_k.T @ z))

    J_sq = rng.normal(size=(12, 12))
    U2, s2, V2 = svd_right(J_sq)
    recon = (U2 * s2) @ V2.T
    assert np.allclose(recon, J_sq)


def test_field_jacobian_is_closed_form_Ap():
    torch.manual_seed(4)
    d, k, b = 12, 3, 5
    stay = StayAffineChord(latent=d, k=k)
    with torch.no_grad():
        stay.A.copy_(0.07 * torch.randn(k, d, d))
        stay.b.copy_(0.02 * torch.randn(k, d))
    z = torch.randn(b, d)
    p = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.2, 0.3, 0.5],
        [0.5, 0.5, 0.0],
    ])
    J_closed = stay.field_jacobian(z, p).detach().cpu().numpy()
    A_np = mix_A(stay.A.detach().cpu().numpy(), p.numpy())
    assert J_closed.shape == (b, d, d)
    assert np.isfinite(J_closed).all()
    assert np.allclose(J_closed, A_np, atol=1e-6, rtol=1e-5)
    # A(p) does not depend on z
    z2 = torch.randn(b, d)
    assert np.allclose(
        stay.field_jacobian(z2, p).detach().cpu().numpy(), A_np, atol=1e-6)


def test_AJ_projection():
    rng = np.random.default_rng(1)
    J = rng.normal(size=(12, 12))
    _, _, V = svd_right(J)
    V_k = V[:, :3]
    A_J = project_jacobian(J, V_k)
    assert A_J.shape == (3, 3)
    assert np.allclose(A_J, V_k.T @ J @ V_k)
    # subspace action: V_k.T J (V_k a) = A_J a
    a = rng.normal(size=3)
    assert np.allclose(V_k.T @ J @ (V_k @ a), A_J @ a)


def test_hurwitz_shift_not_fc_stabilize():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(6, 6))
    H = hurwitz_shift(A, margin=1e-3)
    S = stabilize(A, c=1.0)
    assert not np.allclose(H, S)
    ev = np.linalg.eigvals(H).real
    assert ev.max() < 0.0
    # exact recipe: symmetrize + shift
    sym = 0.5 * (A + A.T)
    lam = float(np.linalg.eigvalsh(sym).max())
    expect = sym - (lam + 1e-3) * np.eye(6)
    assert np.allclose(H, expect)
    # stabilize is the Paper-6 |FC| recipe, a different matrix
    absA = np.abs(A)
    stab_expect = absA / (1.0 + float(np.linalg.eigvalsh(absA).max())) - np.eye(6)
    assert np.allclose(S, stab_expect)
    src = inspect.getsource(hurwitz_shift)
    # the helper must not *call* FC stabilize (docstring may name it)
    code_lines = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("''"):
            continue
        if '"""' in line or "'''" in line:
            continue
        code_lines.append(stripped)
    assert not any(l.startswith("stabilize(") or "= stabilize(" in l or l.startswith("return stabilize(")
                   for l in code_lines)


def test_hurwitz_then_controllability_finite():
    from jlucid.ode.jlucid4.jspace import control_from_AJ

    rng = np.random.default_rng(3)
    J = rng.normal(size=(8, 8))
    _, _, V = svd_right(J)
    A_J = project_jacobian(J, V[:, :3])
    rec = control_from_AJ(A_J)
    assert rec["max_real_eig"] < 0.0
    assert rec["avg"].shape == (3,)
    assert rec["modal"].shape == (3,)
    assert np.isfinite(rec["avg"]).all() and np.isfinite(rec["modal"]).all()
    assert rec["avg_mean"] > 0.0


def test_dmn_contamination_formula():
    loadings = {"DMN": 2.0, "FPN": 1.0, "DAN": 3.0, "VIS": 9.0}
    assert dmn_contamination(loadings, eps=1e-8) == pytest.approx(1.0)
    masks = {
        "DMN": np.array([True, True, False, False, False]),
        "FPN": np.array([False, False, True, False, False]),
        "DAN": np.array([False, False, False, True, False]),
        "VIS": np.array([False, False, False, False, True]),
    }
    pattern = np.array([4.0, 2.0, 1.0, 3.0, 8.0])
    L = pattern_network_loadings(pattern, masks)
    assert L["DMN"] == pytest.approx(3.0)
    assert L["FPN"] == pytest.approx(1.0)
    assert L["DAN"] == pytest.approx(3.0)
    assert dmn_contamination(L) == pytest.approx(3.0 / 2.0)


def test_decoder_map_and_assemble():
    rng = np.random.default_rng(5)
    d, k_ax = 6, 2
    R = rng.normal(size=(90, d))
    V_k = np.eye(d)[:, :k_ax]
    W = map_axes_decoder(R, V_k)
    assert W.shape == (90, 2)
    assert np.allclose(W, R[:, :2])
    # planted rank-1 J: only first right vector
    v = np.zeros(d); v[0] = 1.0
    u = np.zeros(d); u[0] = 1.0
    J = 5.0 * np.outer(u, v)
    masks = {
        "DMN": np.zeros(90, dtype=bool),
        "FPN": np.zeros(90, dtype=bool),
        "DAN": np.zeros(90, dtype=bool),
    }
    masks["DMN"][:10] = True
    masks["FPN"][10:20] = True
    masks["DAN"][20:30] = True
    rec = assemble_subject(J, R=R, masks=masks, k=k_ax, n_frames=7)
    assert rec["C_k"] == pytest.approx(1.0)
    assert rec["r_eff"] == pytest.approx(1.0)
    assert rec["A_J"].shape == (2, 2)
    assert rec["n"] == 7
    assert np.isfinite(rec["dmn_contamination"])


def test_three_ablation_maps_on_constructed_z():
    d, k = 6, 2
    V_k = np.eye(d)[:, :k]
    z = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    z0 = apply_zero(z, V_k)
    assert np.allclose(z0[:2], 0.0)
    assert np.allclose(z0[2:], z[2:])
    assert np.allclose(jspace_coords(z0, V_k), 0.0)

    Q = np.eye(d)[:, 2:4]
    zr = apply_ablate(z, Q)
    assert np.allclose(zr[2:4], 0.0)
    assert np.allclose(zr[:2], z[:2])
    assert not np.allclose(zr, z0)

    donor = np.array([10.0, 20.0])
    zs = apply_swap(z, V_k, donor)
    assert np.allclose(zs[:2], donor)
    assert np.allclose(zs[2:], z[2:])
    assert np.allclose(jspace_coords(zs, V_k), donor)

    # batch + random frame
    Z = np.stack([z, 2 * z], axis=0)
    Qr = random_orthonormal(d, k, seed=42)
    assert np.allclose(Qr.T @ Qr, np.eye(k), atol=1e-8)
    Zr = apply_ablate(Z, Qr)
    assert Zr.shape == (2, d)
    assert np.allclose(Zr @ Qr, 0.0, atol=1e-8)


def test_swap_donors_no_fixed_points():
    pair = swap_donors(8, seed=42)
    assert pair.shape == (8,)
    assert set(pair.tolist()) == set(range(8))
    assert not np.any(pair == np.arange(8))


def test_h3_drives_real_permutation_on_planted_groups():
    rng = np.random.default_rng(8)
    n_a, n_t = 24, 24
    rows = []
    for i in range(n_a):
        rows.append({
            "dx_group": "ADHD", "C_k": 0.4 + 0.02 * rng.normal(),
            "r_eff": 6.0 + 0.1 * rng.normal(),
            "dmn_contamination": 1.0 + 0.05 * rng.normal(),
            "age": 10.0, "gender": 1, "mean_fd": 0.1, "site": "NYU",
        })
    for i in range(n_t):
        rows.append({
            "dx_group": "TDC", "C_k": 0.8 + 0.02 * rng.normal(),
            "r_eff": 3.0 + 0.1 * rng.normal(),
            "dmn_contamination": 1.0 + 0.05 * rng.normal(),
            "age": 10.0, "gender": 1, "mean_fd": 0.1, "site": "NYU",
        })
    df = pd.DataFrame(rows)
    out = h3_group_stats(df, n_perm=199, seed=42)
    assert set(out.metric) == {"C_k", "r_eff", "dmn_contamination"}
    ck = out[out.metric == "C_k"].iloc[0]
    re = out[out.metric == "r_eff"].iloc[0]
    assert ck.p_raw < 0.05 and re.p_raw < 0.05
    assert ck.p_fdr <= 1.0
    # null metric stays large
    dmn = out[out.metric == "dmn_contamination"].iloc[0]
    assert dmn.p_raw > 0.05


def test_covariates_match_step2_class():
    df = pd.DataFrame({
        "age": [10.0, 12.0],
        "gender": [1, 0],
        "mean_fd": [0.1, 0.2],
        "site": ["NYU", "KKI"],
    })
    cov = covariates_from_table(df)
    assert "age" in cov.columns and "sex" in cov.columns and "motion" in cov.columns
    site_cols = [c for c in cov.columns if c.startswith("site_")]
    assert len(site_cols) == 2


def test_reserved_artifact_names_and_official_ckpt_guard():
    from pathlib import Path
    from jlucid.config import (
        ABLATION_JSON,
        STEP3_AXES_NPZ,
        STEP3_CONTROL_CSV,
        STEP3_ENERGY_CSV,
        STEP3_GROUP_STATS_CSV,
        STEP3_JACOBIANS_H5,
        STEP3_METRICS_CSV,
        STEP3_VALIDATION_JSON,
        STEP3_VALIDATION_MD,
        STEP3_REPORT_MD,
        STEP3_REPORT_HTML,
    )
    from jlucid.ode.jlucid4.jspace import official_checkpoint_ok

    assert STEP3_JACOBIANS_H5.name == "jacobians.h5"
    assert STEP3_AXES_NPZ.name == "jspace_axes.npz"
    assert STEP3_METRICS_CSV.name == "jspace_metrics.csv"
    assert STEP3_CONTROL_CSV.name == "jspace_control.csv"
    assert STEP3_ENERGY_CSV.name == "latent_energy.csv"
    assert STEP3_GROUP_STATS_CSV.name == "jspace_group_stats.csv"
    assert ABLATION_JSON.name == "ablation_results.json"
    assert STEP3_VALIDATION_JSON.name == "validation_results.json"
    assert STEP3_VALIDATION_MD.name == "validation_report.md"
    assert STEP3_REPORT_MD.name == "step3_report.md"
    assert STEP3_REPORT_HTML.name == "step3_report.html"
    bad = Path("data/processed/step3/ode4/packonly_ode39/checkpoint.pt")
    ok, msg = official_checkpoint_ok(bad)
    assert ok is False and "pack-only" in msg


def test_decoder_jacobian_matches_linear_layer():
    torch.manual_seed(0)
    lin = nn.Linear(4, 7, bias=True)
    z = np.array([0.2, -0.1, 0.3, 0.0], dtype=np.float32)
    R = decoder_jacobian(lin, z)
    W = lin.weight.detach().cpu().numpy()
    assert R.shape == (7, 4)
    assert np.allclose(R, W, atol=1e-6)
