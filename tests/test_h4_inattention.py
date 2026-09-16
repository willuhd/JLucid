"""H4 symptom hygiene and Spearman evaluation (does not touch QC files)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from jlucid.stats.symptoms import (
    clean_sentinels,
    clean_symptom_frame,
    evaluate_h4,
    h4_block,
    rank_within,
    spearman_rho,
    steiger_t,
)


def test_clean_sentinels_maps_missing_codes_only():
    x = np.array([-999.0, 32.0, 999.0, np.nan, 71.0, -999, 0.0])
    y = clean_sentinels(x)
    assert np.isnan(y[0]) and np.isnan(y[2]) and np.isnan(y[3]) and np.isnan(y[5])
    assert y[1] == pytest.approx(32.0)
    assert y[4] == pytest.approx(71.0)
    assert y[6] == pytest.approx(0.0)


def test_clean_symptom_frame_is_a_copy():
    raw = pd.DataFrame({
        "inattentive": [-999, 28, 70],
        "hyper_impulsive": [21, -999, 66],
        "adhd_index": [48, 999, 71],
    })
    out = clean_symptom_frame(raw)
    assert raw["inattentive"].tolist() == [-999, 28, 70]
    assert np.isnan(out.loc[0, "inattentive"])
    assert out.loc[1, "inattentive"] == 28
    assert np.isnan(out.loc[1, "hyper_impulsive"])
    assert np.isnan(out.loc[1, "adhd_index"])


def test_rank_within_measure_equalizes_scales():
    inatt = pd.Series([20.0, 35.0, 51.0, 90.0])
    meas = pd.Series([1, 1, 2, 2])
    r = rank_within(inatt, meas)
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(2.0)
    assert r[2] == pytest.approx(1.0)
    assert r[3] == pytest.approx(2.0)


def test_spearman_rho_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = -0.8 * x + rng.normal(size=40) * 0.2
    got = spearman_rho(x, y)
    exp_r, exp_p = spearmanr(x, y)
    assert got["n"] == 40
    assert got["rho"] == pytest.approx(float(exp_r))
    assert got["p"] == pytest.approx(float(exp_p))
    assert np.isnan(spearman_rho(x[:3], y[:3])["rho"])


def test_steiger_detects_stronger_inatt_than_hyper():
    rng = np.random.default_rng(1)
    n = 80
    ck = rng.normal(size=n)
    inatt = -ck + 0.15 * rng.normal(size=n)
    hyper = 0.05 * ck + rng.normal(size=n)
    r_in = spearman_rho(ck, inatt)
    r_hy = spearman_rho(ck, hyper)
    r_ih = spearman_rho(inatt, hyper)
    ste = steiger_t(r_in["rho"], r_hy["rho"], r_ih["rho"], n)
    assert r_in["rho"] < -0.5
    assert abs(r_hy["rho"]) < abs(r_in["rho"])
    assert ste["t"] < 0
    assert ste["p"] < 0.05


def test_evaluate_h4_primary_is_adhd_ranked_and_null_when_uncorrelated():
    rng = np.random.default_rng(2)
    n_a, n_t = 40, 40
    rows = []
    for i in range(n_a):
        rows.append({
            "dx_group": "ADHD",
            "adhd_measure": 1 if i < 20 else 2,
            "C_k": rng.normal(),
            "r_eff": rng.normal(),
            "dmn_contamination": rng.normal(),
            "L_FPN": rng.normal(),
            "L_DAN": rng.normal(),
            "inattentive": (25.0 + rng.normal()) if i < 20 else (70.0 + rng.normal()),
            "hyper_impulsive": (20.0 + rng.normal()) if i < 20 else (65.0 + rng.normal()),
            "age": 10.0, "gender": 1, "mean_fd": 0.1,
            "site": "NYU" if i < 20 else "Peking_1",
        })
    for i in range(n_t):
        rows.append({
            "dx_group": "TDC",
            "adhd_measure": 1,
            "C_k": rng.normal(),
            "r_eff": rng.normal(),
            "inattentive": 15.0 + rng.normal(),
            "hyper_impulsive": 12.0 + rng.normal(),
            "age": 10.0, "gender": 1, "mean_fd": 0.1, "site": "NYU",
            "dmn_contamination": rng.normal(), "L_FPN": rng.normal(), "L_DAN": rng.normal(),
        })
    out = evaluate_h4(pd.DataFrame(rows))
    assert out["n_adhd_valid"] == n_a
    assert out["n_pooled_valid"] == n_a + n_t
    ck = next(r for r in out["primary_rows"] if r["metric"] == "C_k")
    assert ck["n"] == n_a
    assert ck["p_inatt"] > 0.05
    assert out["headline"]["supported"] is False


def test_evaluate_h4_recovers_planted_inatt_link():
    rng = np.random.default_rng(3)
    n = 60
    ck = rng.normal(size=n)
    df = pd.DataFrame({
        "dx_group": "ADHD",
        "adhd_measure": np.where(np.arange(n) < 30, 1, 2),
        "C_k": ck,
        "r_eff": rng.normal(size=n),
        "dmn_contamination": rng.normal(size=n),
        "L_FPN": rng.normal(size=n),
        "L_DAN": rng.normal(size=n),
        "inattentive": np.where(np.arange(n) < 30, 25.0, 70.0) - 4.0 * ck,
        "hyper_impulsive": np.where(np.arange(n) < 30, 20.0, 65.0) + 0.1 * rng.normal(size=n),
        "age": 11.0, "gender": 0, "mean_fd": 0.12, "site": "NYU",
    })
    out = evaluate_h4(df)
    ck_row = next(r for r in out["primary_rows"] if r["metric"] == "C_k")
    assert ck_row["rho_inatt"] < -0.4
    assert ck_row["p_inatt"] < 0.05
    assert ck_row["abs_inatt_gt_abs_hyper"]
    assert out["headline"]["supported"] is True


def test_h4_block_uses_real_spearman():
    df = pd.DataFrame({
        "C_k": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        "inatt_rank": [1, 2, 3, 4, 5, 6],
        "hyper_rank": [3, 1, 2, 6, 5, 4],
    })
    row = h4_block(df, "C_k", "inatt_rank", "hyper_rank")
    assert row["rho_inatt"] == pytest.approx(spearmanr(df.C_k, df.inatt_rank).statistic)
    assert row["n"] == 6
