import numpy as np
import pandas as pd
import pytest

from jlucid.stats.effects import cohens_d, fdr_table, fisher_z_corr


def test_cohens_d_identical_groups_zero():
    x = np.array([1.0, 2.0, 3.0])
    assert cohens_d(x, x.copy()) == 0.0


def test_cohens_d_matches_manual():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50)
    y = rng.normal(1, 1, 50)
    d = cohens_d(x, y)
    sd = np.sqrt(((49 * x.var(ddof=1)) + (49 * y.var(ddof=1))) / 98)
    expected = (x.mean() - y.mean()) / sd
    assert d == pytest.approx(expected)
    assert d < 0


def test_cohens_d_nan_on_constant():
    assert np.isnan(cohens_d(np.ones(10), np.ones(10) + 0.5))


def test_fisher_z_null():
    z, p = fisher_z_corr(0.0, 100, 0.0, 100)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_fisher_z_difference_significant():
    z, p = fisher_z_corr(0.5, 1000, 0.3, 1000)
    assert p < 0.05
    assert z > 0


def test_fdr_table():
    df = fdr_table(np.array([0.5, 0.01, 0.03]), ["a", "b", "c"])
    assert list(df.p_raw) == sorted(df.p_raw)
    assert (df.p_fdr <= 1.0).all()
    assert (df.p_fdr.diff().dropna() >= 0).all()


def test_h4_stouffer_uses_per_site_fisher_z():
    """Headline H4 is Stouffer of per-site Fisher-z; fisher_z_corr is called."""
    from jlucid.stats.effects import h4_avg_modal

    rng = np.random.default_rng(0)
    rows = []
    for site, n in (("A", 20), ("B", 18)):
        for dx in ("ADHD", "TDC"):
            for i in range(n):
                a = 1.0 + 0.2 * rng.normal()
                sid = f"{site}_{dx}_{i}"
                rows.append({
                    "sid": sid, "avg": a, "modal": a - 0.02 + 0.01 * rng.normal(),
                    "dx_group": dx, "site": site, "scan_dir_id": sid,
                })
    df = pd.DataFrame(rows)
    qc = df[["scan_dir_id", "dx_group", "site"]].drop_duplicates()
    cov = pd.DataFrame({
        "age": np.full(len(qc), 10.0),
        "sex": np.full(len(qc), 1.0),
        "motion": np.full(len(qc), 0.1),
        "site": qc["site"].astype(str).to_numpy(),
    })
    cov = pd.get_dummies(cov, columns=["site"], drop_first=False)
    cov.index = qc["scan_dir_id"].to_numpy()
    h4 = h4_avg_modal(df[["sid", "avg", "modal"]], qc, cov, min_n=10)
    assert h4["n_sites_tested"] == 2
    assert all(r["tested"] for r in h4["per_site"])
    # identical within-site coupling → Stouffer null
    assert h4["p"] > 0.05
    assert np.isfinite(h4["z"])
    assert np.isfinite(h4["z_residualized"])
    # Stouffer equals sum(site z) / sqrt(n_sites)
    zs = [r["z"] for r in h4["per_site"]]
    assert h4["z"] == pytest.approx(sum(zs) / np.sqrt(len(zs)))
