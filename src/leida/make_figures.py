#!/usr/bin/env python3
"""
exp/37 — Figure generation for the ADHD dispersion finding + cohort overview.
Reads only cached artifacts (gate1_oracle/*.npz, raw features) and data/ manifests;
writes PNGs + figure_stats.json to results/figures/.

Figures:
  fig1_cohort.png           — cohort demographics (both datasets)
  fig2_dispersion.png       — D distributions by group/state (the headline)
  fig3_sites.png            — per-site consistency (ADHD-200)
  fig4_within_subject.png   — PennLEAD rest -> n-back within-subject change
  fig5_block_forest.png     — per-feature-block effect sizes (both datasets)
  fig6_null_calibration.png — permutation null histograms with observed statistics
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/37_leida"
FIG = f"{BASE}/results/figures"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "figure.dpi": 160, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_HC, C_AD, C_G3 = "#4C72B0", "#C44E52", "#CCB974"
rng = np.random.RandomState(20260930)

# ----------------------------- load artifacts -----------------------------
z = np.load(f"{OUT}/batch15_adhd200_comb.npz", allow_pickle=True)
comb = z["comb"]; do = z["do"]; s = z["s"]; mo = z["mo"]; keep_idx = z["keep_idx"]
medk = z["medk"]
if medk.dtype.kind == "S":
    medk = np.array([x.decode() for x in medk])
noncoded = (do == 0) | ((do == 1) & np.isin(medk, ["2", "-999", "nan"]))  # non-medicated-coded ADHD
valid_un = noncoded & np.isfinite(comb)

sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
ages_all = np.array(meta_all["ages"], dtype=float)
ph = pd.read_csv(f"{BASE}/data/adhd200/adhd200_preprocessed_phenotypics.tsv", sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
gmap = dict(zip(ph["ScanDir ID"], ph["Gender"].fillna(0).astype(int)))
sex_all = np.array([gmap.get(x, 0) for x in ids_all], dtype=float)
ag = ages_all[keep_idx].copy(); ag[np.isnan(ag)] = np.nanmean(ages_all)
sx = sex_all[keep_idx]

zp = np.load(f"{OUT}/batch15c_penn_dispz.npz", allow_pickle=True)
pids_r = [str(x) for x in zp["pids_rest"]]; pids_n = [str(x) for x in zp["pids_nback"]]
comb_r = zp["comb_rest"]; comb_n = zp["comb_nback"]
dx_r = zp["dx_rest"]; dx_n = zp["dx_nback"]
cohort = pd.read_csv(f"{BASE}/data/pennlead/pheno/cohort.csv")
cohort["participant_id"] = cohort["participant_id"].astype(str)
cm = cohort.set_index("participant_id")
age_n = np.array([float(cm.loc[p, "age"]) if p in cm.index else np.nan for p in pids_n])
sex_n = np.array([str(cm.loc[p, "sex"]) if p in cm.index else "?" for p in pids_n])
grp_n = np.array([str(cm.loc[p, "study_group"]) if p in cm.index else "?" for p in pids_n])

# ADHD-200 per-block normative z (same recipe as batch 15)
zb1 = np.load(f"{OUT}/features.npz"); zb5 = np.load(f"{OUT}/batch5_features.npz"); zb2 = np.load(f"{OUT}/batch2_features.npz")
BLOCKS_A2 = {"H3 (Hancock-VAR variant)": zb5["H3__hvar"],
             "K1 (signed-A controllability)": zb5["K1__signed"],
             "Lnet28 (LEiDA co-leadership)": zb1["g050c5K60__Lnet28"],
             "W1 (phase-difference variance)": zb2["g050c5K60__W1"]}
hc = do == 0

def norm_z(X):
    n, d = X.shape
    Z = np.full_like(X, np.nan)
    for st in np.unique(s):
        mhc = hc & (s == st); mall = s == st
        if mhc.sum() < 30:
            continue
        A = np.column_stack([np.ones(mhc.sum()), ag[mhc], sx[mhc]])
        for j in range(d):
            y = X[mhc, j]; ok = np.isfinite(y)
            if ok.sum() < 30:
                continue
            coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
            r = y[ok] - A[ok] @ coef
            mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
            Aall = np.column_stack([np.ones(mall.sum()), ag[mall], sx[mall]])
            Z[mall, j] = (X[mall, j] - Aall @ coef) / mad
    return np.clip(Z, -3, 3)

def delta_stats(D, lab, ss=None, nperm=5000, seed=1, return_null=False):
    """ADHD - HC mean difference of D, permutation p, var ratio, Cohen's d."""
    mA = lab == 1; mH = lab == 0
    obs = D[mA].mean() - D[mH].mean()
    va, vh = D[mA].var(ddof=1), D[mH].var(ddof=1)
    n1, n0 = int(mA.sum()), int(mH.sum())
    sp = np.sqrt(((n1 - 1) * va + (n0 - 1) * vh) / (n1 + n0 - 2))
    d = obs / sp
    vr = va / vh
    rngp = np.random.RandomState(seed); nl = []
    for _ in range(nperm):
        lp = lab.copy()
        if ss is None:
            lp = rngp.permutation(lab)
        else:
            for st in np.unique(ss):
                mk = ss == st
                lp[mk] = rngp.permutation(lab[mk])
        nl.append(D[lp == 1].mean() - D[lp == 0].mean())
    nl = np.array(nl)
    p = float((np.sum(np.abs(nl) >= abs(obs)) + 1) / (nperm + 1))
    out = dict(delta=float(obs), p=p, vr=float(vr), d=float(d), nA=n1, nH=n0)
    if return_null:
        out["null"] = nl
    return out

def boot_ci(Da, Dh, n=1000, seed=7):
    r = np.random.RandomState(seed); b = []
    for _ in range(n):
        a = Da[r.randint(0, len(Da), len(Da))]
        h = Dh[r.randint(0, len(Dh), len(Dh))]
        b.append(a.mean() - h.mean())
    return np.percentile(b, [2.5, 97.5])

STATS = {}

# key tests (subject level; preregistered statistic)
st_a2 = delta_stats(comb[valid_un], do[valid_un], s[valid_un], return_null=True)
m_r = np.isfinite(dx_r) & np.isfinite(comb_r)
st_pl_r = delta_stats(comb_r[m_r], dx_r[m_r], None, return_null=True)
m_n = np.isfinite(dx_n) & np.isfinite(comb_n)
st_pl_n = delta_stats(comb_n[m_n], dx_n[m_n], None, return_null=True)
common = [p for p in pids_r if p in pids_n]
ir = [pids_r.index(p) for p in common]; inn = [pids_n.index(p) for p in common]
dcomb = comb_n[inn] - comb_r[ir]; dx_c = dx_n[inn]
m_ws = np.isfinite(dcomb) & np.isfinite(dx_c)
st_ws = delta_stats(dcomb[m_ws], dx_c[m_ws], None, return_null=True)
for nm, st in [("adhd200_combined_rest", st_a2), ("pennlead_rest", st_pl_r),
               ("pennlead_nback", st_pl_n), ("pennlead_within_subject", st_ws)]:
    STATS[nm] = {k: v for k, v in st.items() if k != "null"}

# per-block (ADHD-200 rest, non-coded; PennLEAD nback)
blk_a2, blk_pl = {}, {}
for k, X in BLOCKS_A2.items():
    Z = norm_z(X[keep_idx])
    Dk = np.nanmean(Z ** 2, axis=1)
    mk = noncoded & np.isfinite(Dk)
    blk_a2[k] = delta_stats(Dk[mk], do[mk], s[mk], nperm=2000, seed=2)
    STATS[f"a2_block::{k}"] = blk_a2[k]
pl_keys = {"H3 (Hancock-VAR variant)": "mz2_nback_H3v",
           "K1 (signed-A controllability)": "mz2_nback_K1",
           "Lnet28 (LEiDA co-leadership)": "mz2_nback_Lnet_r",
           "W1 (phase-difference variance)": "mz2_nback_W1_r"}
for k, key in pl_keys.items():
    Dk = zp[key]
    mk = np.isfinite(dx_n) & np.isfinite(Dk)
    blk_pl[k] = delta_stats(Dk[mk], dx_n[mk], None, nperm=2000, seed=3)
    STATS[f"pl_block::{k}"] = blk_pl[k]

# ----------------------------- helpers -----------------------------
def style_violin(parts, color, alpha=0.30):
    for pc in parts["bodies"]:
        pc.set_facecolor(color); pc.set_alpha(alpha); pc.set_edgecolor("none")
    if "cmedians" in parts:
        parts["cmedians"].set_color(color); parts["cmedians"].set_linewidth(1.4)

def strip(a, x, D, color, seed=0):
    r = np.random.RandomState(seed)
    a.scatter(x + r.uniform(-0.10, 0.10, len(D)), D, s=7, alpha=0.30,
              color=color, edgecolors="none", zorder=3)

# ============================ FIG 1: cohort ============================
fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.9))

# (a) ADHD-200 age by group
axA = ax[0, 0]
v = axA.violinplot([ag[do == 0], ag[do == 1]], positions=[1, 2], showmedians=True, showextrema=False, widths=0.8)
style_violin(v, "#9db6d0", 0.5)
axA.set_xticks([1, 2]); axA.set_xticklabels([f"HC\n(n={int((do == 0).sum())})", f"ADHD\n(n={int((do == 1).sum())})"])
axA.set_ylabel("Age (years)"); axA.set_title("ADHD-200 QC cohort: age")

# (b) ADHD-200 motion by group
axB = ax[0, 1]
v = axB.violinplot([mo[do == 0], mo[do == 1]], positions=[1, 2], showmedians=True, showextrema=False, widths=0.8)
style_violin(v, "#d0b89d", 0.5)
axB.set_xticks([1, 2]); axB.set_xticklabels(["HC", "ADHD"])
axB.set_ylabel("Max motion (mm)"); axB.set_title("ADHD-200 QC cohort: in-scanner motion")

# (c) ADHD-200 site composition
axC = ax[0, 2]
sites = [1, 3, 4, 5, 6]
nhc = [int(((s == st) & (do == 0)).sum()) for st in sites]
nad = [int(((s == st) & (do == 1)).sum()) for st in sites]
axC.bar(range(len(sites)), nhc, color=C_HC, label="HC")
axC.bar(range(len(sites)), nad, bottom=nhc, color=C_AD, label="ADHD")
axC.set_xticks(range(len(sites))); axC.set_xticklabels([f"site {st}" for st in sites])
axC.set_ylabel("n subjects"); axC.set_title("ADHD-200: site composition")
axC.legend(frameon=False, fontsize=8)

# (d) PennLEAD age by dx
axD = ax[1, 0]
mm = np.isfinite(dx_n) & np.isfinite(age_n)
v = axD.violinplot([age_n[mm & (dx_n == 0)], age_n[mm & (dx_n == 1)]], positions=[1, 2], showmedians=True, showextrema=False, widths=0.8)
style_violin(v, "#9db6d0", 0.5)
axD.set_xticks([1, 2]); axD.set_xticklabels([f"non-ADHD\n(n={int((mm & (dx_n == 0)).sum())})", f"ADHD\n(n={int((mm & (dx_n == 1)).sum())})"])
axD.set_ylabel("Age (years)"); axD.set_title("PennLEAD n-back QC cohort: age")

# (e) study-group composition
axE = ax[1, 1]
gnames = ["ADHD", "PRO/CHR", "TD/NC"]
n1g = [int((mm & (grp_n == g) & (dx_n == 1)).sum()) for g in gnames]
n0g = [int((mm & (grp_n == g) & (dx_n == 0)).sum()) for g in gnames]
axE.bar(range(3), n0g, color=C_HC, label="dx_adhd = 0")
axE.bar(range(3), n1g, bottom=n0g, color=C_AD, label="dx_adhd = 1")
for i, (a, b) in enumerate(zip(n0g, n1g)):
    axE.text(i, a + b + 1, str(a + b), ha="center", fontsize=8)
axE.set_xticks(range(3)); axE.set_xticklabels(gnames)
axE.set_ylabel("n subjects"); axE.set_title("PennLEAD: study group x dx")
axE.legend(frameon=False, fontsize=8)

# (f) sex by dx
axF = ax[1, 2]
cats = ["non-ADHD", "ADHD"]
male = [int((mm & (dx_n == g) & (sex_n == "M")).sum()) for g in (0, 1)]
female = [int((mm & (dx_n == g) & (sex_n == "F")).sum()) for g in (0, 1)]
axF.bar(range(2), male, color="#55a868", label="male")
axF.bar(range(2), female, bottom=male, color="#c1b2d6", label="female")
for i, (m_, f_) in enumerate(zip(male, female)):
    axF.text(i, m_ + f_ + 1, str(m_ + f_), ha="center", fontsize=8)
axF.set_xticks(range(2)); axF.set_xticklabels(cats)
axF.set_ylabel("n subjects"); axF.set_title("PennLEAD: sex composition")
axF.legend(frameon=False, fontsize=8)

fig.suptitle("Figure 1 — Cohort overview (ADHD-200 QC n=722 across 5 sites; PennLEAD n-back QC n=86, single site)", y=1.02, fontsize=10)
fig.savefig(f"{FIG}/fig1_cohort.png")
plt.close(fig)

# ============================ FIG 2: D distributions ============================
fig, ax = plt.subplots(1, 3, figsize=(10.5, 4.0))

def dpanel(a, Dh, Da, title, note):
    v1 = a.violinplot([Dh], positions=[1], showmedians=True, showextrema=False, widths=0.8)
    style_violin(v1, C_HC, 0.35)
    v2 = a.violinplot([Da], positions=[2], showmedians=True, showextrema=False, widths=0.8)
    style_violin(v2, C_AD, 0.35)
    strip(a, 1, Dh, C_HC, seed=11); strip(a, 2, Da, C_AD, seed=22)
    a.set_xticks([1, 2])
    a.set_xticklabels([f"controls\n(n={len(Dh)})", f"ADHD\n(n={len(Da)})"])
    a.set_xlim(0.5, 2.5)
    a.set_title(title)
    a.text(0.03, 0.97, note, transform=a.transAxes, va="top", fontsize=8,
           bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.85))

Dh = comb[valid_un & (do == 0)]; Da = comb[valid_un & (do == 1)]
dpanel(ax[0], Dh, Da, "ADHD-200, rest",
       f"$\\Delta D$ = {st_a2['delta']:+.3f}\nVR = {st_a2['vr']:.2f}, d = {st_a2['d']:.2f}\np = {st_a2['p']:.4f} (within-site perm)")
dpanel(ax[1], comb_r[m_r & (dx_r == 0)], comb_r[m_r & (dx_r == 1)], "PennLEAD, rest",
       f"$\\Delta D$ = {st_pl_r['delta']:+.3f}\nVR = {st_pl_r['vr']:.2f}\np = {st_pl_r['p']:.2f} (n.s.)")
dpanel(ax[2], comb_n[m_n & (dx_n == 0)], comb_n[m_n & (dx_n == 1)], "PennLEAD, n-back task",
       f"$\\Delta D$ = {st_pl_n['delta']:+.3f}\nVR = {st_pl_n['vr']:.2f}, d = {st_pl_n['d']:.2f}\np = {st_pl_n['p']:.4f} (perm)")
for a in ax:
    a.set_ylabel("$D$ = mean $z^2$ (dispersion index)")
fig.suptitle("Figure 2 — Normative-deviation dispersion index $D$ by group and state", y=1.04, fontsize=10)
fig.savefig(f"{FIG}/fig2_dispersion.png")
plt.close(fig)

# ============================ FIG 3: site split ============================
fig, ax = plt.subplots(figsize=(5.6, 4.0))
deltas, los, his, labs = [], [], [], []
for st in sites:
    mkA = valid_un & (do == 1) & (s == st)
    mkH = valid_un & (do == 0) & (s == st)
    dd = comb[mkA].mean() - comb[mkH].mean()
    lo, hi = boot_ci(comb[mkA], comb[mkH], n=2000, seed=int(st))
    deltas.append(dd); los.append(lo); his.append(hi)
    labs.append(f"site {st}\nn={mkA.sum()}/{mkH.sum()}")
x = np.arange(len(sites))
ax.axhline(0, color="#999999", lw=0.8, ls="--")
ax.errorbar(x, deltas, yerr=[np.array(deltas) - np.array(los), np.array(his) - np.array(deltas)],
            fmt="o", color=C_AD, capsize=3, lw=1.2, markersize=6)
ax.axhline(st_a2["delta"], color=C_HC, lw=1.2, ls=":",
           label=f"pooled $\\Delta D$ = {st_a2['delta']:+.3f}")
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
ax.set_ylabel("site-level $\\Delta D$ (ADHD $-$ controls)")
ax.set_title("Figure 3 — ADHD-200: effect consistency across acquisition sites\n(non-coded ADHD vs HC, 95% bootstrap CI)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.savefig(f"{FIG}/fig3_sites.png")
plt.close(fig)

# ============================ FIG 4: within-subject state change ============================
fig, ax = plt.subplots(figsize=(4.6, 4.6))
Dr = comb_r[ir]; Dn = comb_n[inn]; dxc = dx_n[inn]
for g, col in [(0, C_HC), (1, C_AD)]:
    mk = m_ws & (dxc == g)
    for a_, b_ in zip(Dr[mk], Dn[mk]):
        ax.plot([0, 1], [a_, b_], color=col, alpha=0.22, lw=0.7, zorder=2)
chg_hc = dcomb[m_ws & (dxc == 0)].mean(); chg_ad = dcomb[m_ws & (dxc == 1)].mean()
ax.plot([0, 1], [comb_r[ir][m_ws & (dxc == 0)].mean(), comb_n[inn][m_ws & (dxc == 0)].mean()],
        color=C_HC, lw=3, marker="o", label=f"controls: $\\Delta$ = {chg_hc:+.3f}", zorder=4)
ax.plot([0, 1], [comb_r[ir][m_ws & (dxc == 1)].mean(), comb_n[inn][m_ws & (dxc == 1)].mean()],
        color=C_AD, lw=3, marker="o", label=f"ADHD: $\\Delta$ = {chg_ad:+.3f}", zorder=4)
ax.set_xticks([0, 1]); ax.set_xticklabels(["rest", "n-back"])
ax.set_xlim(-0.15, 1.15)
ax.set_ylabel("$D$ (mean $z^2$)")
ax.set_title("Figure 4 — PennLEAD within-subject\nrest $\\to$ n-back change in $D$", fontsize=9.5)
ax.text(0.03, 0.97, f"ADHD $-$ controls change:\n{st_ws['delta']:+.3f}, p = {st_ws['p']:.3f}",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.85))
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.savefig(f"{FIG}/fig4_within_subject.png")
plt.close(fig)

# ============================ FIG 5: block forest ============================
fig, ax = plt.subplots(figsize=(7.4, 4.4))
names = list(BLOCKS_A2.keys()) + ["COMBINED (H3+K1+Lnet28)"]
ypos = np.arange(len(names))[::-1]
for i, k in enumerate(names):
    y = ypos[i]
    if k.startswith("COMBINED"):
        a2d, a2lo, a2hi = st_a2["delta"], *boot_ci(Da, Dh, n=1000, seed=41)
        pld, pllo, plhi = st_pl_n["delta"], *boot_ci(comb_n[m_n & (dx_n == 1)], comb_n[m_n & (dx_n == 0)], n=1000, seed=42)
        pa2, ppl = st_a2["p"], st_pl_n["p"]
    else:
        Dk = np.nanmean(norm_z(BLOCKS_A2[k][keep_idx]) ** 2, axis=1)
        mk = noncoded & np.isfinite(Dk)
        a2d, a2lo, a2hi = blk_a2[k]["delta"], *boot_ci(Dk[mk & (do == 1)], Dk[mk & (do == 0)], n=1000, seed=43)
        pa2 = blk_a2[k]["p"]
        Dk2 = zp[pl_keys[k]]; mk2 = np.isfinite(dx_n) & np.isfinite(Dk2)
        pld, pllo, plhi = blk_pl[k]["delta"], *boot_ci(Dk2[mk2 & (dx_n == 1)], Dk2[mk2 & (dx_n == 0)], n=1000, seed=44)
        ppl = blk_pl[k]["p"]
    ax.errorbar(a2d, y + 0.13, xerr=[[a2d - a2lo], [a2hi - a2d]], fmt="o", color=C_HC,
                capsize=2.5, lw=1.1, markersize=6, label="ADHD-200 (rest)" if i == 0 else None)
    ax.errorbar(pld, y - 0.13, xerr=[[pld - pllo], [plhi - pld]], fmt="s", color=C_AD,
                capsize=2.5, lw=1.1, markersize=6, label="PennLEAD (n-back)" if i == 0 else None)
    ax.text(max(a2hi, plhi) + 0.03, y, f"p={pa2:.3f} | p={ppl:.3f}", fontsize=7.5, va="center", color="#555555")
ax.axvline(0, color="#999999", lw=0.8, ls="--")
ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel("$\\Delta D$ = mean $D_{ADHD}$ $-$ mean $D_{controls}$ (in $z^2$ units)")
ax.set_title("Figure 5 — Per-feature-block dispersion effect (95% bootstrap CI; p = permutation)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.savefig(f"{FIG}/fig5_block_forest.png")
plt.close(fig)

# ============================ FIG 6: null calibration ============================
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
def nullpanel(a, nl, obs, p, title):
    a.hist(nl, bins=40, color="#b8c4d9", edgecolor="white", lw=0.4)
    a.axvline(obs, color=C_AD, lw=1.6)
    a.axvline(-obs if obs > 0 else obs, color=C_AD, lw=0.8, ls=":")
    a.set_title(title, fontsize=9)
    a.text(0.97, 0.95, f"observed $\\Delta D$ = {obs:+.3f}\np = {p:.4f}", transform=a.transAxes,
           ha="right", va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.85))
nullpanel(ax[0], st_a2["null"], st_a2["delta"], st_a2["p"], "ADHD-200, rest (within-site perm)")
nullpanel(ax[1], st_pl_n["null"], st_pl_n["delta"], st_pl_n["p"], "PennLEAD, n-back (perm)")
nullpanel(ax[2], st_ws["null"], st_ws["delta"], st_ws["p"], "PennLEAD within-subject change (perm)")
for a in ax:
    a.set_xlabel("null $\\Delta D$")
ax[0].set_ylabel("permutations")
fig.suptitle("Figure 6 — Permutation null distributions vs observed statistics", y=1.05, fontsize=10)
fig.savefig(f"{FIG}/fig6_null_calibration.png")
plt.close(fig)

# ----------------------------- save + print -----------------------------
with open(f"{FIG}/figure_stats.json", "w") as f:
    json.dump(STATS, f, indent=2)
print(json.dumps(STATS, indent=2))
print("\nFIGURES WRITTEN:", sorted(os.listdir(FIG)))
