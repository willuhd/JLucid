#!/usr/bin/env python3
"""Generate the four essay figures into docs/essay/figs/.
Read-only wrt data; writes only figures and optional cache.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, "essay", "figs")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src"))
pass  # read_roi inlined below
import leida_lib as L
from survival_lib import labels_from_phases
def read_roi(path):
    try:
        d = pd.read_csv(path, sep='\t', comment='#', header=None)
    except Exception:
        return None
    d = d.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    X = d.values.astype(float); X = X[np.isfinite(X).all(1)]
    return X if X.shape[0] > 30 and X.shape[1] > 50 else None

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 120, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
COL = {"adhd": "#D55E00", "tdc": "#0072B2", "gray": "#8C8C8C",
       "green": "#009E73", "purple": "#CC79A7", "orange": "#E69F00"}
RNG = np.random.default_rng(20261106)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".png"))


# ----------------------------------------------------------------------
# Figure 1: rank-2 geometry and gauge (HBN resting-state, 180 subjects)
# ----------------------------------------------------------------------
def fig1():
    Z = np.load(os.path.join(ROOT, "data", "e123_hbn_labels.npz"), allow_pickle=True)
    keys = [k for k in Z.files if k.endswith("||rest_run-1")]
    sids = sorted(set(k.split("||")[0] for k in keys))[:180]
    MOV = os.path.join(ROOT, "data", "hbn_movie_fmri")
    MISSING = [2, 5, 11, 25, 34, 68, 82, 95, 126, 191]
    KEEP = [i for i in range(200) if (i + 1) not in MISSING]
    Ch = np.load(os.path.join(ROOT, "pilot", "out", "leida_centroids_190.npy"))

    flipA, flipM = [], []
    pA_f, pA_nf, pM_f, pM_nf = [], [], [], []
    ph0, X0 = None, None
    used = 0
    for sid in sids:
        X = read_roi(os.path.join(MOV, f"{sid}__rest_run-1__roi_stats.csv"))
        if X is None or X.shape[1] != 200:
            continue
        _, ph = L.leida_v1(X[:, KEEP])
        c, s = np.cos(ph), np.sin(ph)
        cc, ss, cs = (c * c).sum(1), (s * s).sum(1), (c * s).sum(1)
        disc = np.sqrt(np.maximum((cc - ss) ** 2 + 4 * cs * cs, 0))
        l1 = 0.5 * (cc + ss + disc)
        e0, e1 = cs.copy(), l1 - cc
        nn = np.hypot(e0, e1); nn[nn == 0] = 1
        e0 /= nn; e1 /= nn
        Xl = e0[:, None] * c + e1[:, None] * s
        Xl /= np.linalg.norm(Xl, axis=1, keepdims=True)
        k = np.argmax(np.abs(Xl), axis=1)
        s_arg = np.where(Xl[np.arange(len(Xl)), k] < 0, -1.0, 1.0)
        s_maj = np.where((Xl > 0).sum(1) > Xl.shape[1] / 2, -1.0, 1.0)
        fa = np.diff(s_arg) != 0
        fm = np.diff(s_maj) != 0
        la = labels_from_phases(ph, Ch, "argmax")
        lm = labels_from_phases(ph, Ch, "majneg")
        swa = np.diff(la) != 0
        swm = np.diff(lm) != 0
        flipA.append(fa.mean()); flipM.append(fm.mean())
        if fa.any() and (~fa).any():
            pA_f.append(swa[fa].mean()); pA_nf.append(swa[~fa].mean())
        if fm.any() and (~fm).any():
            pM_f.append(swm[fm].mean()); pM_nf.append(swm[~fm].mean())
        if ph0 is None:
            ph0, X0 = ph, Xl
        used += 1
    print("fig1 subjects", used)

    # panel a: eigenvalue spectrum for three frames
    frames = [50, 150, 250]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    ax = axs[0]
    for t in frames:
        C = np.cos(ph0[t][:, None] - ph0[t][None, :])
        ev = np.linalg.eigvalsh(C)[::-1]
        ax.semilogy(np.arange(1, len(ev) + 1), np.maximum(ev, 1e-18),
                    lw=0.9, alpha=0.9)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue (log scale)")
    ax.set_title("(a) iPA spectrum: rank 2")
    ax.set_ylim(1e-18, 5e3)
    ax.axhline(1e-12, color="0.7", lw=0.6, ls=":")
    ax.text(0.98, 0.08, r"$\sigma_3/\sigma_1 \approx 5.6\times10^{-16}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)

    # panel b: leading eigenvector across frames + (psi, R2) inset
    ax = axs[1]
    Xc = np.zeros_like(X0)
    Xc[0] = X0[0]
    for t in range(1, len(X0)):
        Xc[t] = X0[t] if X0[t] @ Xc[t - 1] >= 0 else -X0[t]
    im = ax.imshow(Xc.T, aspect="auto", cmap="RdBu_r", vmin=-0.16, vmax=0.16,
                   interpolation="nearest")
    ax.set_xlabel("frame")
    ax.set_ylabel("parcel index")
    ax.set_title(r"(b) $v_1(t)$: an $N$-vector per frame")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    psi = 0.5 * np.angle(np.exp(2j * ph0).mean(1))
    R2 = np.abs(np.exp(2j * ph0).mean(1))
    ins = ax.inset_axes([0.55, 0.52, 0.42, 0.43])
    sc = ins.scatter(psi, R2, c=np.arange(len(psi)), s=1.5, cmap="viridis", alpha=0.7)
    ins.set_xlabel(r"$\psi$", fontsize=6); ins.set_ylabel(r"$R_2$", fontsize=6)
    ins.tick_params(labelsize=5)
    ins.set_title(r"$(\psi,R_2)$ summary", fontsize=6, pad=1)

    # panel c: gauge contingency
    ax = axs[2]
    pf = [np.mean(pA_f), np.mean(pM_f)]
    pnf = [np.mean(pA_nf), np.mean(pM_nf)]
    ef = [np.std(pA_f, ddof=1) / np.sqrt(len(pA_f)),
          np.std(pM_f, ddof=1) / np.sqrt(len(pM_f))]
    enf = [np.std(pA_nf, ddof=1) / np.sqrt(len(pA_nf)),
           np.std(pM_nf, ddof=1) / np.sqrt(len(pM_nf))]
    labels = [f"max-component\n(flip {100*np.mean(flipA):.1f}%)",
              f"majority-negative\n(flip {100*np.mean(flipM):.1f}%)"]
    x = np.arange(2); w = 0.36
    b1 = ax.bar(x - w / 2, pf, w, yerr=ef, capsize=2.5, color=COL["adhd"],
                label="gauge flipped")
    b2 = ax.bar(x + w / 2, pnf, w, yerr=enf, capsize=2.5, color=COL["tdc"],
                label="gauge stable")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("P(label switch | gauge state)")
    ax.set_ylim(0, 1.02)
    ax.set_title("(c) sign gauge manufactures switches")
    ax.legend(frameon=False, loc="upper right")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, min(h + 0.02, 0.98),
                f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)

    save(fig, "fig1_rank2_gauge")


# ----------------------------------------------------------------------
# Figure 2: J-space occupancy degeneracy
# ----------------------------------------------------------------------
def fig2():
    S = pd.read_csv(os.path.join(ROOT, "outputs", "jspace_dispersion_subjects.csv"))
    S["sid"] = S.sid.astype(str).str.zfill(7)
    J = pd.read_csv(os.path.join(ROOT, ".agents", "previous", "results", "jspace",
                                 "jfuture", "metrics.csv"))
    J["sid"] = J.sid.astype(str).str.zfill(7)
    M = S.merge(J[["sid", "occ_0_stayshots", "occ_1_stayshots", "occ_2_stayshots"]],
                on="sid", how="inner")
    O = pd.read_csv(os.path.join(ROOT, "outputs", "o4_subject.csv"),
                    dtype={"sid": str})
    O["sid"] = O.sid.str.zfill(7)
    M = M.merge(O[["sid", "o4"]], on="sid", how="inner")
    M["logfd"] = np.log(M.mean_fd.clip(lower=1e-4))

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)

    # panel a: occupancy simplex colored by C_k
    ax = axs[0]
    o = M[["occ_0", "occ_1", "occ_2"]].values
    sc = ax.scatter(o[:, 0], o[:, 1], c=M.C_k_A.values, s=8, cmap="viridis",
                    vmin=0.78, vmax=0.85, alpha=0.85, linewidths=0)
    ax.plot([0, 1], [1, 0], color="0.4", lw=0.8, ls="--")
    ax.set_xlim(0.12, 0.52); ax.set_ylim(0.10, 0.48)
    ax.set_xlabel(r"$f_0$"); ax.set_ylabel(r"$f_1$")
    ax.set_title(r"(a) $C_k=f(f)$")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"$C_k$", fontsize=6.5); cb.ax.tick_params(labelsize=6)

    # panel b: R2 bars
    ax = axs[1]
    metrics = [r"$C_k$", r"$r_{\mathrm{eff}}$", "DMN"]
    lin = [0.5547, 0.6570, 0.6278]
    quad = [0.7656, 0.7671, 0.7312]
    x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w / 2, lin, w, color=COL["tdc"], label="linear (all-shot)")
    b2 = ax.bar(x + w / 2, quad, w, color=COL["orange"], label="quadratic (stay-shot)")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0); ax.set_ylabel(r"$R^2$ on occupancy")
    ax.set_title("(b) occupancy predicts metrics")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    for b in list(b1) + list(b2):
        h = b.get_height(); ax.text(b.get_x() + b.get_width() / 2, h + 0.015,
                                     f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)

    # panel c: coupling collapse
    ax = axs[2]
    site_d = pd.get_dummies(M.site, drop_first=True).astype(float).reset_index(drop=True)
    cov = pd.concat([M[["age", "gender", "logfd"]].reset_index(drop=True), site_d], axis=1)

    def partial(y, x, covs):
        X = pd.concat([pd.Series(1.0, index=covs.index, name="const"), covs], axis=1)
        ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
        r = float(np.corrcoef(ry, rx)[0, 1])
        n, k = len(y), X.shape[1]
        t = r * np.sqrt((n - k - 2) / max(1e-12, 1 - r * r))
        return r, float(2 * stats.t.sf(abs(t), n - k - 2))

    occ = pd.concat([cov, M[["occ_0", "occ_1", "occ_2"]].reset_index(drop=True)], axis=1)
    vals = {}
    for met, lab in [("C_k_A", r"$C_k$"), ("r_eff_A", r"$r_{\mathrm{eff}}$")]:
        r0, p0 = partial(M[met].values, M.o4.values, cov)
        r1, p1 = partial(M[met].values, M.o4.values, occ)
        vals[lab] = (r0, p0, r1, p1)
    labels = list(vals)
    rawv = [vals[k][0] for k in labels]
    occv = [vals[k][2] for k in labels]
    x = np.arange(2); w = 0.36
    b1 = ax.bar(x - w / 2, rawv, w, color=COL["purple"], label="raw partial")
    b2 = ax.bar(x + w / 2, occv, w, color=COL["gray"], label="after occupancy")
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("partial $r$ with the $k=5$ lens")
    ax.set_ylim(-0.45, 0.46)
    ax.set_title("(c) control removes it")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    for b, lab in zip(list(b1) + list(b2), labels + labels):
        h = b.get_height()
        p = vals[lab][1] if b is b1[labels.index(lab)] else vals[lab][3]
        ps = "<0.001" if p < 0.001 else f"={p:.2f}"
        ax.text(b.get_x() + b.get_width() / 2, h + 0.02,
                f"{h:+.2f} (p{ps})", ha="center", va="bottom", fontsize=6)
    save(fig, "fig2_jspace_occupancy")


# ----------------------------------------------------------------------
# Figure 3: pattern-defined positive effect
# ----------------------------------------------------------------------
def true_motion():
    fs = sorted([f for f in glob.glob(os.path.join(ROOT, "data", "adhd200", "*", "*",
                                                   "*cc200_TCs.1D"))
                 if "sfnwmrda" in os.path.basename(f)])
    runs = []
    for f in fs:
        nt = sum(1 for _ in open(f)) - 1
        if nt < 120:
            continue
        b = os.path.basename(f)
        m = re.match(r"sfnwmrda(\d+)_(session_\d+_rest_\d+)_cc200_TCs.1D", b)
        if not m:
            continue
        runs.append((m.group(1).zfill(7), "rp_" + m.group(1) + "_" + m.group(2) + ".1D"))
    mot = {}
    for f in glob.glob(os.path.join(ROOT, "data", "adhd200", "*", "*_motion.csv")):
        try:
            mm = pd.read_csv(f)
        except Exception:
            continue
        if "File" not in mm.columns or "Max Motion (mm)" not in mm.columns:
            continue
        for _, row in mm.iterrows():
            fn = str(row["File"])
            v = pd.to_numeric(row["Max Motion (mm)"], errors="coerce")
            if fn.startswith("rp_") and np.isfinite(v):
                mot[fn] = float(v)
    D = pd.DataFrame(runs, columns=["sid", "run"])
    D["mm"] = D.run.map(mot)
    return D.groupby("sid").mm.mean().reset_index()


def fig3():
    O = pd.read_csv(os.path.join(ROOT, "outputs", "o4_subject.csv"),
                    dtype={"sid": str})
    O["sid"] = O.sid.str.zfill(7)
    A = O[O.Site.isin([1, 3, 4, 5]) & O.ADHD.notna()].copy()
    A = A.merge(true_motion(), on="sid", how="left")
    site_name = {1: "Peking (pooled)", 3: "KKI", 4: "NeuroIMAGE", 5: "NYU"}

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)

    # panel a: distributions
    ax = axs[0]
    groups = [A[A.ADHD == 1].o4.values, A[A.ADHD == 0].o4.values]
    parts = ax.violinplot(groups, positions=[0, 1], widths=0.75, showextrema=False)
    for pc, c in zip(parts["bodies"], [COL["adhd"], COL["tdc"]]):
        pc.set_facecolor(c); pc.set_alpha(0.25); pc.set_edgecolor(c)
    for i, (vals, c) in enumerate(zip(groups, [COL["adhd"], COL["tdc"]])):
        jit = RNG.normal(0, 0.045, len(vals))
        ax.scatter(i + jit, vals, s=3, color=c, alpha=0.35, linewidths=0)
        ax.plot([i - 0.22, i + 0.22], [np.median(vals)] * 2, color="k", lw=1.2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ADHD", "TDC"])
    ax.set_ylabel(r"$k=5$ state occupancy")
    ax.set_title("(a) pattern-matched occupancy")
    ax.text(0.03, 0.95, r"$d=-0.302$" "\n" r"perm $p=0.001$ (floor)",
            transform=ax.transAxes, fontsize=6.5, va="top")

    # panel b: site-wise d with bootstrap CI
    ax = axs[1]
    rows = []
    for s in [1, 3, 4, 5]:
        g = A[A.Site == s]
        a = g[g.ADHD == 1].o4.values; b = g[g.ADHD == 0].o4.values
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = (a.mean() - b.mean()) / sp
        bs = []
        for _ in range(2000):
            aa = RNG.choice(a, len(a), replace=True); bb = RNG.choice(b, len(b), replace=True)
            spp = np.sqrt(((len(aa) - 1) * aa.var(ddof=1) + (len(bb) - 1) * bb.var(ddof=1)) /
                          (len(aa) + len(bb) - 2))
            bs.append((aa.mean() - bb.mean()) / spp)
        rows.append((site_name[s], d, np.percentile(bs, 2.5), np.percentile(bs, 97.5),
                     len(a), len(b)))
    for i, (nm, d, lo, hi, na, nb) in enumerate(rows):
        c = COL["adhd"] if d < 0 else COL["gray"]
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=c, ms=4,
                    capsize=2, lw=1)
    ax.axvline(0, color="0.3", lw=0.7)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{nm} ({na}/{nb})" for nm, _, _, _, na, nb in rows])
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("Cohen's $d$ (ADHD $-$ TDC)")
    ax.set_title("(b) site-wise effect")

    # panel c: meta vs motion-matched
    ax = axs[2]
    # meta (fixed effect, inverse variance)
    ds, vs = [], []
    for s in [1, 3, 4, 5]:
        g = A[A.Site == s]; a = g[g.ADHD == 1].o4.values; b = g[g.ADHD == 0].o4.values
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        dd = (a.mean() - b.mean()) / sp
        se = np.sqrt((len(a) + len(b)) / (len(a) * len(b)) + dd ** 2 / (2 * (len(a) + len(b))))
        ds.append(dd); vs.append(se ** 2)
    w = 1 / np.array(vs); dmeta = float((np.array(ds) * w).sum() / w.sum())
    se_meta = float(np.sqrt(1 / w.sum()))
    # true-motion matched pairs within site
    pairs = []
    for s, g in A.dropna(subset=["mm"]).groupby("Site"):
        a = g[g.ADHD == 1]; b = g[g.ADHD == 0].copy(); used = set()
        for i, row in a.iterrows():
            cand = b[~b.index.isin(used)].copy()
            if len(cand) == 0:
                break
            j = (np.log(cand.mm) - np.log(row.mm)).abs().idxmin()
            used.add(j); pairs.append((row.o4, cand.loc[j, "o4"]))
    P = pd.DataFrame(pairs, columns=["a", "c"])
    diff = P.a.values - P.c.values
    dmat = diff.mean() / diff.std(ddof=1)
    dbt = (P.a.mean() - P.c.mean()) / np.sqrt((P.a.var(ddof=1) + P.c.var(ddof=1)) / 2)
    pm = stats.ttest_rel(P.a, P.c).pvalue
    bsd = []
    for _ in range(2000):
        ii = RNG.integers(0, len(diff), len(diff))
        dd = diff[ii]; bsd.append(dd.mean() / dd.std(ddof=1))
    lo_m, hi_m = np.percentile(bsd, [2.5, 97.5])
    specs = [("all sites (meta)", dmeta, dmeta - 1.96 * se_meta, dmeta + 1.96 * se_meta,
              0.000155, len(A)),
             ("motion-matched", dmat, lo_m, hi_m, pm, len(P))]
    for i, (nm, d, lo, hi, p, n) in enumerate(specs):
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=COL["adhd"],
                    ms=4, capsize=2, lw=1)
        pstr = "<0.001" if p < 0.001 else f"={p:.3f}"
        if nm == "motion-matched":
            txt = f"$d_z={d:+.3f}$, $d={dbt:+.3f}$\n$p{pstr}$, $n={n}$"
        else:
            txt = f"$d={d:+.3f}$\n$p{pstr}$, $n={n}$"
        ax.text(d, i + 0.24, txt, fontsize=6.2, ha="center", va="bottom")
    ax.axvline(0, color="0.3", lw=0.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["all sites (meta)", "motion-matched"])
    ax.set_ylim(-0.6, 1.85)
    ax.set_xlabel("Cohen's $d$")
    ax.set_title("(c) motion-matched effect")
    save(fig, "fig3_o2_effect")


# ----------------------------------------------------------------------
# Figure 4: identifiability
# ----------------------------------------------------------------------
def fig4():
    R = pd.read_csv(os.path.join(ROOT, "outputs", "o4_refits.csv"))
    k, n = int((R.idx == 4).sum()), len(R)
    rec = k / n
    lo, hi = stats.beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)

    ax = axs[0]
    ax.bar([0], [rec], width=0.55, color=COL["adhd"])
    ax.errorbar([0], [rec], yerr=[[rec - lo], [hi - rec]], fmt="none", color="k",
                capsize=3, lw=1)
    ax.axhline(0.2, color="0.4", ls="--", lw=0.9, label="chance 0.20")
    ax.set_xticks([0]); ax.set_xticklabels(["index-4\nrecovery"])
    ax.set_ylim(0, 0.45); ax.set_ylabel("fraction of refits")
    ax.set_title("(a) recovery vs chance")
    ax.legend(frameon=False, loc="upper right", fontsize=6.5)
    ax.text(0, rec + 0.02, f"{k}/{n} = {rec:.1%}\n[{lo:.2f}, {hi:.2f}]",
            ha="center", va="bottom", fontsize=6.5)

    ax = axs[1]
    bins = np.linspace(-0.42, 0.42, 15)
    ax.hist(R.d_matched, bins=bins, color=COL["tdc"], alpha=0.65,
            label="pattern-matched")
    ax.hist(R.d_literal4, bins=bins, color=COL["gray"], alpha=0.55,
            label="literal index")
    ax.axvline(0, color="0.3", lw=0.8)
    ax.axvline(R.d_matched.mean(), color=COL["tdc"], ls="--", lw=0.9)
    ax.axvline(R.d_literal4.mean(), color="0.4", ls=":", lw=0.9)
    ax.set_xlabel("Cohen's $d$ across refits")
    ax.set_ylabel("number of refits")
    ax.set_title("(b) literal vs pattern-matched")
    ax.legend(frameon=False, loc="upper right", fontsize=6.3)

    ax = axs[2]
    counts = [int((R.idx == i).sum()) for i in range(5)]
    ax.bar(range(5), counts, color=COL["green"], width=0.6)
    ax.axhline(n / 5, color="0.4", ls="--", lw=0.9)
    ax.set_xticks(range(5)); ax.set_xlabel("recovered reference index")
    ax.set_ylabel("number of refits"); ax.set_ylim(0, max(counts) + 2.5)
    ax.set_title("(c) the index is uniform")
    save(fig, "fig4_identifiability")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("all figures done")
