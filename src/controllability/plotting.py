"""Plotting for reporting — multiple graphs per run, per ROI, per null, with formulas.
Uses matplotlib + seaborn, Intel-Mac friendly (MPLCONFIGDIR=/tmp).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Formulas for overlay (top-left)
FORMULA_TEXT = r"$A=|FC|/(c+\lambda_{max}(|FC|))-I$" + "\n" + r"$\phi_i=\sum_j v_{ij}^2/(-2\lambda_j)$" + "\n" + r"$\mu_i=\sum_j (1-e^{\lambda_j})v_{ij}^2$"

def add_formula(ax, text=FORMULA_TEXT, fontsize=7):
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=fontsize,
            va='top', ha='left', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

def plot_distribution(avgs, mods, labels, title, out_path, formula=True):
    """Violin + swarm for avg/modal HC vs ADHD."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df = pd.DataFrame({"group": np.where(labels==0, "HC", "ADHD"), "avg": avgs, "modal": mods})
    for i, col in enumerate(["avg", "modal"]):
        ax = axes[i]
        sns.violinplot(data=df, x="group", y=col, palette={"HC":"#4C72B0", "ADHD":"#DD8452"}, ax=ax, inner="quartile")
        sns.stripplot(data=df, x="group", y=col, color="black", size=2, alpha=0.4, ax=ax, jitter=True)
        ax.set_title(f"{col} controllability")
        if formula and i==0:
            add_formula(ax)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_ladder_heatmap(df, out_path, title="Ladder p-values (permutation)"):
    """Heatmap of p_avg / p_modal across configs. Expects df with roi,c,norm,p_avg,p_modal."""
    # Create pivot: roi vs config string
    df2 = df.copy()
    df2["cfg"] = df2["roi"].astype(str) + "|c=" + df2["c"].astype(str) + "|" + df2["norm"].astype(str)
    # Sort
    fig, axes = plt.subplots(2, 1, figsize=( max(8, len(df2)*0.4), 6))
    for ax, col, name in zip(axes, ["p_avg", "p_modal"], ["avg", "modal"]):
        # Bar plot sorted by p
        df_sorted = df2.sort_values(col)
        colors = ["#d73027" if p<0.05 else "#4575b4" for p in df_sorted[col]]
        ax.barh(range(len(df_sorted)), -np.log10(df_sorted[col].values+1e-300), color=colors)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted["cfg"].values, fontsize=6)
        ax.set_xlabel("-log10(p_perm)")
        ax.set_title(f"{name} controllability — red = p<0.05")
        ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=1)
        add_formula(ax, fontsize=6)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_effect_sizes(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ["d_avg", "d_modal"]):
        df_sorted = df.sort_values(col)
        ax.barh(range(len(df_sorted)), df_sorted[col].values, color=["#e45756" if abs(x)>0.5 else "#54a24b" for x in df_sorted[col].values])
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels((df_sorted["roi"].astype(str)+" c"+df_sorted["c"].astype(str)).values, fontsize=6)
        ax.set_xlabel("Cohen's d")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(col)
        add_formula(ax, fontsize=6)
    fig.suptitle("Effect sizes across ladder (Cohen's d)", fontsize=11)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_avg_modal_scatter(avgs, mods, labels, out_path, title="Avg vs Modal (per subject)"):
    fig, ax = plt.subplots(figsize=(5,5))
    hc = labels==0; adhd=labels==1
    ax.scatter(avgs[hc], mods[hc], c="#4C72B0", label="HC", alpha=0.7, s=30)
    ax.scatter(avgs[adhd], mods[adhd], c="#DD8452", label="ADHD", alpha=0.7, s=30, marker="^")
    # correlation
    r = np.corrcoef(avgs, mods)[0,1]
    ax.set_xlabel("avg controllability (whole-brain mean)")
    ax.set_ylabel("modal controllability")
    ax.legend()
    ax.set_title(f"{title}  r={r:.2f}")
    add_formula(ax)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return r

def plot_site_violin(values, sites, labels, out_path, title="Site distribution"):
    df = pd.DataFrame({"site": sites.astype(str), "value": values, "group": np.where(labels==0, "HC", "ADHD")})
    fig, ax = plt.subplots(figsize=(8,4))
    sns.violinplot(data=df, x="site", y="value", hue="group", palette={"HC":"#4C72B0", "ADHD":"#DD8452"}, ax=ax, split=False)
    ax.set_title(title)
    add_formula(ax, fontsize=6)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_fc_example(fc, out_path, title="FC example (one subject)"):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    add_formula(ax, fontsize=6)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
