#!/usr/bin/env python
"""Independent audit of docs/ADHD_O2_JSPACE.md coupling claim.

Reimplemented from the write-up only (no peeking at
scripts/adhd_o2_jspace_report.py / j_occ_coupling.py / jspace_static_control.py
until this file has run).

Claim: o4 occupancy (de-novo LEiDA lens) ~ C_k (partial r=-.335) and r_eff
(+.337), N=307, 124 ADHD/183 TDC, 6 sites, surviving age/sex/log-FD/site and a
static-pattern control.

Extra: jfuture/metrics.csv is a *different* Jacobian lens (J_route, 4x12) with
its own C_k/r_eff; compare.

Outputs JSON to outputs/jspace_audit.json and prints a
human table.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
JL = os.path.join(ROOT, ".agents", "previous")
LAB = f"{ROOT}/data/e123_adhd200_denovo_labels.npz"
CEN = f"{ROOT}/data/e123_adhd200_denovo_C.npy"
CV = f"{ROOT}/data/pilot_out/canonical_v.npz"
JSM = f"{JL}/results/jspace/jspace_metrics.csv"
JFU = f"{JL}/results/jspace/jfuture/metrics.csv"
PHENO = f"{JL}/data/adhd200/adhd200_preprocessed_phenotypics.tsv"
MOTION = f"{JL}/data/adhd200/adhd200_maxmotion.tsv"
OUT = f"{ROOT}/outputs"
os.makedirs(OUT, exist_ok=True)

NPERM = 2000
SEED = 20261026
RNG = np.random.default_rng(SEED)

RESULTS: dict = {}


# --------------------------------------------------------------------------
# stats helpers
# --------------------------------------------------------------------------
def design(df: pd.DataFrame) -> np.ndarray:
    """Intercept + age + sex + log mean FD + 5 site dummies (6 sites)."""
    X = [np.ones(len(df)), df["age"].to_numpy(float),
         df["gender"].to_numpy(float), np.log(df["mean_fd"].to_numpy(float))]
    sites = sorted(df["site"].unique())
    for s in sites[1:]:
        X.append((df["site"].to_numpy() == s).astype(float))
    return np.column_stack(X)


def resid(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    return np.asarray(y, float) - X @ beta


def partial_r(y: np.ndarray, x: np.ndarray, X: np.ndarray) -> dict:
    ry, rx = resid(y, X), resid(x, X)
    r = float(np.corrcoef(ry, rx)[0, 1])
    k = X.shape[1] - 1               # conditioning variables, excl. intercept
    df = len(y) - k - 2
    t = r * np.sqrt(df / max(1e-12, 1.0 - r * r))
    p = float(2 * stats.t.sf(abs(t), df))
    return {"r": r, "t": float(t), "p": p, "df": int(df), "N": int(len(y))}


def mde_r(n: int, k: int, alpha: float = 0.05, power: float = 0.80) -> float:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(np.tanh(z / np.sqrt(max(1, n - k - 2))))


def perm_p(y, x, X, site, nperm=NPERM, rng=None):
    """Two-sided within-site permutation of x, covariate design held fixed."""
    rng = rng or RNG
    ry = resid(y, X)
    obs = float(np.corrcoef(ry, resid(x, X))[0, 1])
    groups = [np.where(site == s)[0] for s in np.unique(site)]
    cnt = 0
    for _ in range(nperm):
        idx = np.arange(len(x))
        for g in groups:
            idx[g] = rng.permutation(g)
        rp = float(np.corrcoef(ry, resid(x[idx], X))[0, 1])
        if abs(rp) >= abs(obs) - 1e-15:
            cnt += 1
    return obs, (1 + cnt) / (nperm + 1)


def perm_p_fwer(y, xdict, X, site, nperm=NPERM, rng=None):
    """Max-|r| FWER across metrics, one within-site row permutation per draw."""
    rng = rng or RNG
    ry = resid(y, X)
    obs = {k: float(np.corrcoef(ry, resid(v, X))[0, 1]) for k, v in xdict.items()}
    obs_max = max(abs(v) for v in obs.values())
    groups = [np.where(site == s)[0] for s in np.unique(site)]
    cnt = 0
    for _ in range(nperm):
        idx = np.arange(len(y))
        for g in groups:
            idx[g] = rng.permutation(g)
        mx = 0.0
        for v in xdict.values():
            rr = float(np.corrcoef(ry, resid(v[idx], X))[0, 1])
            mx = max(mx, abs(rr))
        if mx >= obs_max - 1e-15:
            cnt += 1
    return obs, obs_max, (1 + cnt) / (nperm + 1)


def within_site_z(v: np.ndarray, site: np.ndarray) -> np.ndarray:
    out = np.empty_like(v, float)
    for s in np.unique(site):
        m = site == s
        out[m] = (v[m] - v[m].mean()) / (v[m].std(ddof=1) if v[m].std(ddof=1) > 0 else 1)
    return out


# --------------------------------------------------------------------------
# 1. merge
# --------------------------------------------------------------------------
def build_subject_table() -> pd.DataFrame:
    z = np.load(LAB, allow_pickle=True)
    keys = list(z.keys())
    rows = []
    for k in keys:
        sub, run = k.split("||")
        lab = z[k]
        rows.append((sub, run, len(lab), float(np.mean(lab == 4)),
                     float(np.mean(lab == 0)), float(np.mean(lab == 1)),
                     float(np.mean(lab == 2)), float(np.mean(lab == 3))))
    lab_df = pd.DataFrame(rows, columns=["sub", "run", "n_frames", "o4", "o0", "o1", "o2", "o3"])
    # subject level: mean over runs (equal weight per run), plus pooled sensitivity
    g = lab_df.groupby("sub").agg(
        o4=("o4", "mean"), nruns=("o4", "size"), n_frames=("n_frames", "sum"),
        o4_pooled=("o4", lambda v: np.nan))  # placeholder, filled below
    runs_by_sub = lab_df.groupby("sub")["run"].agg(list)
    pooled = lab_df.eval("o4 * n_frames").groupby(lab_df["sub"]).sum() / \
        lab_df.groupby("sub")["n_frames"].sum()
    g["o4_pooled"] = pooled
    g["runs"] = g.index.map(runs_by_sub)
    g = g.reset_index()

    j = pd.read_csv(JSM)
    j["sid7"] = j["sid"].astype(str).str.zfill(7)
    jf = pd.read_csv(JFU)
    jf["sid7"] = jf["sid"].astype(str).str.zfill(7)

    # match diagnostics: with and without zfill
    subs = set(g["sub"])
    raw_j = set(j["sid"].astype(str))
    z_j = set(j["sid7"])
    raw_jf = set(jf["sid"].astype(str))
    z_jf = set(jf["sid7"])
    RESULTS["merge"] = {
        "npz_runs": len(keys), "npz_subjects": len(subs),
        "jspace_rows": len(j), "jfuture_rows": len(jf),
        "match_jspace_raw": len(subs & raw_j),
        "match_jspace_zfill7": len(subs & z_j),
        "match_jfuture_raw": len(subs & raw_jf),
        "match_jfuture_zfill7": len(subs & z_jf),
        "npz_subjects_with_leading_zero": int(sum(s.startswith("0") for s in subs)),
        "jspace_sid_len": int(j["sid"].astype(str).str.len().max()),
        "jspace_all_7digit": bool((j["sid"].astype(str).str.len() == 7).all()),
        "runs_per_subject": {str(k): int(v) for k, v in g["nruns"].value_counts().items()},
        # the trap in the other direction: a raw-int join would silently drop
        # nothing here because every jspace sid is already 7 digits.
        "jspace_sid_min": int(j["sid"].min()),
    }

    df = j.merge(g, left_on="sid7", right_on="sub", how="inner")
    df = df.merge(jf.drop(columns=["sid"]), on="sid7", how="left",
                  suffixes=("", "_fut"))
    # sanity: site labels agree with maxmotion folder
    mot = pd.read_csv(MOTION, sep="\t")
    mot["sid7"] = mot["ScanDir ID"].astype(str).str.zfill(7)
    mm = mot.set_index("sid7")["maxmotion_mm"]
    df["maxmotion_mm"] = df["sid7"].map(mm)
    df["log_fd"] = np.log(df["mean_fd"])
    RESULTS["merge"]["merged_N"] = int(len(df))
    RESULTS["merge"]["maxmotion_matched"] = int(df["maxmotion_mm"].notna().sum())
    RESULTS["merge"]["dx"] = {k: int(v) for k, v in df["dx_group"].value_counts().items()}
    RESULTS["merge"]["site"] = {k: int(v) for k, v in df["site"].value_counts().items()}
    return df


# --------------------------------------------------------------------------
# static-pattern features from canonical_v
# --------------------------------------------------------------------------
def sign_fix(v: np.ndarray, mode: str = "absmax") -> np.ndarray:
    v = np.asarray(v, float)
    if mode == "absmax":
        pivot = np.argmax(np.abs(v), axis=1)
        s = np.sign(v[np.arange(len(v)), pivot])
        s[s == 0] = 1.0
    elif mode == "sum":
        s = np.sign(v.sum(axis=1))
        s[s == 0] = 1.0
    else:
        raise ValueError(mode)
    return v * s[:, None]


def static_features(df: pd.DataFrame, mode: str = "absmax") -> pd.DataFrame:
    cv = np.load(CV, allow_pickle=True)
    C = np.load(CEN)
    Ccorr = C / np.linalg.norm(C, axis=1, keepdims=True)   # centroids are NOT unit norm
    rows = []
    for _, r in df.iterrows():
        sub = r["sub"]
        vv, pp, ev = [], [], []
        for run in r["runs"]:
            k = f"{sub}||{run}"
            if k not in cv:
                continue
            v = sign_fix(cv[k], mode)
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            vv.append(v)
            # phase-coherence matrix M = mean_t v_t v_t^T (sign invariant)
            pp.append(v.T @ v / len(v))
        if not vv:
            rows.append([sub] + [np.nan] * 11)
            continue
        nfr = sum(len(v) for v in vv)
        vmean = sum(v.sum(axis=0) for v in vv) / nfr
        vmean /= np.linalg.norm(vmean)
        cosm = vmean @ Ccorr.T                                  # 5 cosines
        M = sum(p * len(v) for p, v in zip(pp, vv)) / nfr       # pooled frames
        ev = np.sort(np.linalg.eigvalsh(M))[::-1][:5]
        rows.append([sub] + list(cosm) + list(ev) + [nfr])
    cols = ["sub"] + [f"cos_{j}" for j in range(5)] + \
           [f"pc_eig_{j}" for j in range(5)] + ["pc_frames"]
    sf = pd.DataFrame(rows, columns=cols)
    return sf


# --------------------------------------------------------------------------
# static control regression
# --------------------------------------------------------------------------
def control_block(df, y, xname, extra_cols, label):
    X = np.column_stack([design(df)] + [df[c].to_numpy(float) for c in extra_cols])
    res = partial_r(y, df[xname].to_numpy(float), X)
    mdl = mde_r(len(df), X.shape[1] - 1)
    res["MDE_r"] = mdl
    RESULTS.setdefault("static_control", {})[label] = res
    return res


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    df = build_subject_table()
    y = df["o4"].to_numpy(float)
    print(f"N={len(df)}  " + " ".join(f"{k}:{v}" for k, v in RESULTS["merge"]["dx"].items()))
    print("sites:", RESULTS["merge"]["site"])

    # ---- reference model: write-up spec (age, sex, logFD, site) ----
    X = design(df)
    site = df["site"].to_numpy()
    kcov = X.shape[1] - 1
    metrics = ["C_k", "r_eff", "dmn_contamination"]
    tab = {}
    for m in metrics:
        pr = partial_r(y, df[m].to_numpy(float), X)
        _, pp = perm_p(y, df[m].to_numpy(float), X, site)
        pr["perm_p"] = pp
        pr["MDE_r"] = mde_r(len(df), kcov)
        pr["raw_mean"] = float(df[m].mean())
        pr["raw_sd"] = float(df[m].std())
        tab[m] = pr
    obs, obs_max, fwer = perm_p_fwer(y, {m: df[m].to_numpy(float) for m in metrics}, X, site)
    RESULTS["coupling"] = {"table": tab, "fwer_p": fwer, "fwer_max_abs_r": obs_max,
                           "nperm": NPERM, "df": tab["C_k"]["df"],
                           "MDE_r": mde_r(len(df), kcov)}
    print("\n== primary (o4 ~ J | age+sex+logFD+site) ==")
    for m, v in tab.items():
        print(f"{m:20s} r={v['r']:+.4f} t={v['t']:+.2f} p={v['p']:.2e} "
              f"perm={v['perm_p']:.4f} N={v['N']} MDE={v['MDE_r']:.4f}")
    print(f"FWER(max|r| over 3) = {fwer:.4f}")

    # ---- sensitivity: + DX, maxmotion instead of mean FD, pooled o4 ----
    Xdx = np.column_stack([X, (df.dx_group == "ADHD").astype(float)])
    sens = {}
    for m in metrics:
        sens[f"{m}|+DX"] = partial_r(y, df[m].to_numpy(float), Xdx)
    Xmm = np.column_stack([design(df)[:, :3], np.log(df["maxmotion_mm"].to_numpy(float)),
                           design(df)[:, 4:]])
    for m in metrics:
        sens[f"{m}|maxmotion"] = partial_r(y, df[m].to_numpy(float), Xmm)
    for m in metrics:
        sens[f"{m}|pooled_o4"] = partial_r(df["o4_pooled"].to_numpy(float),
                                           df[m].to_numpy(float), X)
    RESULTS["sensitivity"] = sens
    print("\n== sensitivity ==")
    for k, v in sens.items():
        print(f"{k:24s} r={v['r']:+.4f} p={v['p']:.2e} N={v['N']}")

    # ---- item 6: site-wise ----
    sitewise = {}
    for s in sorted(df["site"].unique()):
        d = df[df.site == s]
        if len(d) < 12:
            continue
        Xs = np.column_stack([np.ones(len(d)), d["age"].to_numpy(float),
                              d["gender"].to_numpy(float), np.log(d["mean_fd"].to_numpy(float))])
        pr = partial_r(d["o4"].to_numpy(float), d["C_k"].to_numpy(float), Xs)
        pr["site_mean_o4"] = float(d.o4.mean())
        pr["site_mean_Ck"] = float(d.C_k.mean())
        pr["MDE_r"] = mde_r(len(d), 3)
        sitewise[s] = pr
    RESULTS["sitewise"] = sitewise
    print("\n== site-wise C_k ==")
    for s, v in sitewise.items():
        print(f"{s:12s} N={v['N']:3d} r={v['r']:+.4f} p={v['p']:.2e} MDE={v['MDE_r']:.4f}")

    # ---- item 7a: LOSO + influence ----
    loso = {}
    for s in sorted(df["site"].unique()):
        d = df[df.site != s].copy()
        res = partial_r(d["o4"].to_numpy(float), d["C_k"].to_numpy(float), design(d))
        res["N"] = len(d)
        loso[s] = res
    RESULTS["loso"] = loso
    print("\n== leave-one-site-out (C_k) ==")
    for s, v in loso.items():
        print(f"drop {s:12s} N={v['N']:3d} r={v['r']:+.4f} p={v['p']:.2e}")

    infl = {}
    for m in ["C_k", "r_eff"]:
        x = df[m].to_numpy(float)
        base = partial_r(y, x, X)["r"]
        rr = np.empty(len(df))
        for i in range(len(df)):
            mask = np.arange(len(df)) != i
            rr[i] = partial_r(y[mask], x[mask], X[mask])["r"]
        delta = np.abs(rr - base)
        order = np.argsort(-delta)
        top = [(df.iloc[i]["sub"], float(delta[i])) for i in order[:5]]
        infl[m] = {"base_r": base, "loso_r_min": float(rr.min()), "loso_r_max": float(rr.max()),
                   "max_abs_delta": float(delta.max()), "top5": top,
                   "r_excl_top1": float(partial_r(y[order[1:]], x[order[1:]], X[order[1:]])["r"]),
                   "r_excl_top3": float(partial_r(y[order[3:]], x[order[3:]], X[order[3:]])["r"]),
                   "r_excl_top5": float(partial_r(y[order[5:]], x[order[5:]], X[order[5:]])["r"])}
    RESULTS["influence"] = infl
    print("\n== influence (C_k) ==")
    print(f"base r={infl['C_k']['base_r']:+.4f} LOSO range [{infl['C_k']['loso_r_min']:+.4f},"
          f" {infl['C_k']['loso_r_max']:+.4f}]  excl top3 r={infl['C_k']['r_excl_top3']:+.4f}")

    # ---- item 7b: scaling invariance ----
    scal = {}
    for m in ["C_k", "r_eff"]:
        x = df[m].to_numpy(float)
        variants = {
            "raw": x,
            "global_z": (x - x.mean()) / x.std(ddof=1),
            "within_site_z": within_site_z(x, site),
            "x100": x * 100.0,
            "rank": stats.rankdata(x),
            "log": np.log(x),
        }
        pr = {k: partial_r(y, v, X)["r"] for k, v in variants.items()}
        scal[m] = pr
    RESULTS["scaling"] = scal
    print("\n== scaling invariance ==")
    for m, v in scal.items():
        print(f"{m}: " + "  ".join(f"{kk}={vv:+.6f}" for kk, vv in v.items()))

    # ---- items 4/5: static controls ----
    sf = static_features(df, mode="absmax")
    sf_sum = static_features(df, mode="sum")
    d2 = df.merge(sf, on="sub", how="left").merge(
        sf_sum.rename(columns={c: c + "_s" for c in sf_sum.columns if c != "sub"}),
        on="sub", how="left")
    have = d2["pc_frames"].notna()
    d2 = d2[have].copy()
    RESULTS["static_control_N"] = int(len(d2))
    print(f"\n== static controls (N={len(d2)}) ==")
    cos_cols = [f"cos_{j}" for j in range(5)]
    eig_cols = [f"pc_eig_{j}" for j in range(5)]
    for m in ["C_k", "r_eff", "dmn_contamination"]:
        r0 = control_block(d2, d2["o4"], m, [], f"{m}|covariates_only")
        r1 = control_block(d2, d2["o4"], m, cos_cols, f"{m}|+5_cosines")
        r2 = control_block(d2, d2["o4"], m, eig_cols, f"{m}|+5_pc_eigs")
        r3 = control_block(d2, d2["o4"], m, cos_cols + eig_cols, f"{m}|+5_cos+5_pc_eigs")
        print(f"{m:20s} raw {r0['r']:+.4f} | +cos {r1['r']:+.4f} (p={r1['p']:.2e}) | "
              f"+eig {r2['r']:+.4f} (p={r2['p']:.2e}) | +cos+eig {r3['r']:+.4f} (p={r3['p']:.2e})")
    # sign convention robustness
    for m in ["C_k", "r_eff"]:
        r4 = control_block(d2, d2["o4"], m, [c + "_s" for c in cos_cols], f"{m}|+5_cosines_sumfix")
        RESULTS["static_control"][f"{m}|sumfix"] = r4["r"]
    # cosines alone predict o4?
    Xc = np.column_stack([design(d2)] + [d2[c].to_numpy(float) for c in cos_cols])
    R2 = 1 - resid(d2["o4"].to_numpy(float), Xc).var() / resid(d2["o4"].to_numpy(float), design(d2)).var()
    RESULTS["static_control"]["o4_R2_incremental_cosines"] = float(R2)
    print(f"o4 incremental R2 from 5 cosines: {R2:.4f}")

    # ---- circularity: the J metrics are functions of the k=3 state occupancy ----
    # jspace.py: J_bar = frames["J"].mean(0); J = A(p) with p = one-hot
    # soft_state_prob(labels); A(p) = sum_s p_s A_s with shared A_s.
    # Hence J_bar = sum_s f_s A_s and C_k/r_eff/dmn are deterministic functions
    # of the 2-dof occupancy vector f. Test with the model's own occupancy.
    circ = {}
    occ_cols = [c for c in ["occ_0", "occ_1", "occ_2"] if c in df.columns]
    ss_cols = [c for c in ["occ_0_stayshots", "occ_1_stayshots", "occ_2_stayshots"]
               if c in df.columns]
    for m in ["C_k", "r_eff", "dmn_contamination"]:
        ent = {}
        for tag, cols in [("occ_all", occ_cols), ("occ_stayshots", ss_cols),
                          ("occ_plus_stayshots", occ_cols + ss_cols)]:
            if not cols:
                continue
            Xo = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
            res = resid(df[m].to_numpy(float), Xo)
            ent[f"R2_{tag}"] = float(1 - res.var() / df[m].var(ddof=1))
        # quadratic in stay-shot occupancy
        if ss_cols:
            Xb = [np.ones(len(df))] + [df[c].to_numpy(float) for c in ss_cols]
            import itertools
            for a, b in itertools.combinations_with_replacement(ss_cols, 2):
                Xb.append(df[a].to_numpy(float) * df[b].to_numpy(float))
            Xq = np.column_stack(Xb)
            res = resid(df[m].to_numpy(float), Xq)
            ent["R2_quadratic_stayshots"] = float(1 - res.var() / df[m].var(ddof=1))
        circ[m] = ent
    # partial o4 ~ J | covariates + model occupancy
    Xocc = np.column_stack([X] + [df[c].to_numpy(float) for c in occ_cols]) if occ_cols else X
    for m in ["C_k", "r_eff", "dmn_contamination", "switch_rate"]:
        pr = partial_r(y, df[m].to_numpy(float), Xocc)
        circ[f"partial_{m}_given_occ"] = pr
    # multiple partial R of o4 on the 2-dof occupancy vs on C_k
    for tag, cols in [("occ_all", occ_cols), ("occ_stayshots", ss_cols)]:
        Xo = np.column_stack([X] + [df[c].to_numpy(float) for c in cols])
        ry = resid(y, X)
        ro = resid(y, Xo)
        circ[f"o4_partial_R_{tag}"] = float(np.sqrt(max(0, 1 - ro.var() / ry.var())))
    RESULTS["circularity"] = circ
    print("\n== circularity: J metrics as functions of state occupancy ==")
    for m, v in circ.items():
        if isinstance(v, dict):
            print(f"{m:20s} " + " ".join(f"{k}={vv:.4f}" for k, vv in v.items()))
        elif isinstance(v, float):
            print(f"{m:36s} = {v:.4f}")
        else:
            print(f"{m:36s} r={v['r']:+.4f} p={v['p']:.2e} df={v['df']}")

    # ---- exclude all Peking sites ----
    pek = df["site"].str.startswith("Peking")
    dnp = df[~pek]
    pr = partial_r(dnp["o4"].to_numpy(float), dnp["C_k"].to_numpy(float), design(dnp))
    pr["N"] = int(len(dnp))
    pek_r = partial_r(df[pek]["o4"].to_numpy(float), df[pek]["C_k"].to_numpy(float),
                      design(df[pek]))
    pek_r["N"] = int(pek.sum())
    RESULTS["exclude_peking"] = {"non_peking": pr, "peking_only": pek_r}
    print(f"\n== non-Peking (N={pr['N']}) C_k r={pr['r']:+.4f} p={pr['p']:.2e}; "
          f"Peking-only (N={pek_r['N']}) r={pek_r['r']:+.4f} p={pek_r['p']:.2e}")

    # ---- jfuture lens comparison ----
    fut = {}
    Xf = design(df)
    for m, col in [("C_k_fut", "C_k_fut"), ("C_1", "C_1"), ("r_eff_fut", "r_eff_fut"),
                   ("switch_rate", "switch_rate"), ("axis_stability", "axis_stability"),
                   ("z_var", "z_var"), ("J_t_var", "J_t_var")]:
        if col not in df.columns or df[col].isna().all():
            continue
        pr = partial_r(y, df[col].to_numpy(float), Xf)
        _, pp = perm_p(y, df[col].to_numpy(float), Xf, site)
        pr["perm_p"] = pp
        pr["sd"] = float(df[col].std(ddof=1))
        pr["mean"] = float(df[col].mean())
        fut[m] = pr
    RESULTS["jfuture"] = fut
    print("\n== jfuture (routing Jacobian) lens ==")
    for m, v in fut.items():
        print(f"{m:16s} sd={v['sd']:.5f} r={v['r']:+.4f} p={v['p']:.2e} perm={v['perm_p']:.4f}")

    # ---- metric-cluster correlations ----
    clus = {}
    pairs = [("C_k", "switch_rate"), ("r_eff", "switch_rate"),
             ("dmn_contamination", "switch_rate"), ("C_k", "r_eff"),
             ("C_k", "C_k_fut"), ("r_eff", "r_eff_fut"),
             ("C_k", "axis_stability"), ("C_k", "J_t_var")]
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            m = df[[a, b]].dropna()
            clus[f"{a}~{b}"] = float(np.corrcoef(m[a], m[b])[0, 1])
    RESULTS["metric_correlations"] = clus
    print("\n== metric correlations ==")
    for k, v in clus.items():
        print(f"{k:40s} r={v:+.4f}")

    # ---- save ----
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o

    with open(f"{OUT}/jspace_audit.json", "w") as f:
        json.dump(clean(RESULTS), f, indent=2)
    print(f"\nwrote {OUT}/audit_jspace_results.json")


if __name__ == "__main__":
    main()
