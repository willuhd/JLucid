"""Central configuration for the JLucid stage-1 pipeline.

Every analysis choice documented in .agents/step-1.md (decisions D1-D12) is a
constant here, with a comment citing the source so the report can be
regenerated from code.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"          # original downloads (user-managed)
PER_SITE_DIRS = DATASETS_DIR / "per-site"      # curled *_motion.csv files
DATA_DIR = REPO_ROOT / "data"                  # gitignored working area
ATHENA_MAIN_DIR = DATA_DIR / "athena" / "main" # extracted filtfix (training-sample) archive
ATHENA_TEST_DIR = DATA_DIR / "athena" / "test" # extracted TestRelease archive (unused for analysis)
PROCESSED_DIR = DATA_DIR / "processed"
DOWNLOAD_DIR = DATA_DIR / "download"
EXTERNAL_DIR = DATA_DIR / "external"           # nilearn atlas cache
FIGURES_DIR = PROCESSED_DIR / "figures"
LOGS_DIR = PROCESSED_DIR / "logs"
SRC_DATA_DIR = Path(__file__).resolve().parent / "data"

# matplotlib needs a writable config dir inside the sandbox
os.environ.setdefault("MPLCONFIGDIR", str(DATA_DIR / ".cache" / "mplconfig"))

for _d in (PROCESSED_DIR, DOWNLOAD_DIR, FIGURES_DIR, LOGS_DIR, DATA_DIR / ".cache"):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data format (D1, D11)
# ---------------------------------------------------------------------------
N_ROI = 90                     # AAL-90 cerebral ROIs (Paper 7 parcellation)
EXPECTED_TC_COLS = 116         # the Athena archive ships AAL-116; we keep the first 90
TC_PREFIX = "sfnwmrda"         # Athena-filtered variant (0.009-0.08 Hz, 6 motion params
                               # + WM/CSF regressed, 6mm smoothing) - verified spectrally
REST_SESSION = "rest_1"        # primary resting-state run used for all analyses
MIN_FRAMES = 120               # D5: subjects with fewer usable frames are dropped

SITE_CODE = {
    "KKI": "3", "NeuroIMAGE": "4", "NYU": "5", "OHSU": "6",
    "Peking_1": "1", "Peking_2": "1", "Peking_3": "1",
    "Pittsburgh": "7", "WashU": "8",
}
SITE_TR = {  # from the Athena per-site preproc.sh scripts (verified 2026-08-09)
    "KKI": 2.5, "NeuroIMAGE": 1.96, "NYU": 2.0, "OHSU": 2.5,
    "Peking_1": 2.0, "Peking_2": 2.0, "Peking_3": 2.0,
    "Pittsburgh": 1.5, "WashU": 2.5,
}
SITE_NAME = {
    "1": "Peking", "2": "Bradley", "3": "KKI", "4": "NeuroIMAGE",
    "5": "NYU", "6": "OHSU", "7": "Pittsburgh", "8": "WashU",
}

# Phenotypic Key: DX 0=TDC, 1=ADHD-Combined, 2=ADHD-Hyper/Impulsive,
# 3=ADHD-Inattentive, pending=unreleased diagnosis (D5)
DX_GROUP = {"0": "TDC", "1": "ADHD", "2": "ADHD", "3": "ADHD", "pending": "pending"}

# ---------------------------------------------------------------------------
# QC gates (D4, D5)
# ---------------------------------------------------------------------------
# Primary motion gate: the Athena release ships only per-scan summary CSVs
# (max displacement/rotation); per-frame rp_*.1D live in the multi-GB preproc
# tarballs, deferred. Section 14 fallback: use max displacement.
MOTION_MAX_MM_PRIMARY = 3.0
MOTION_MAX_MM_SENSITIVITY = 2.0
QC_ATHENA_KEEP = "1"           # D5: keep QC_Athena == 1

# Frame-level scrubbing (implemented + tested; used when rp_*.1D become
# available later - currently all-False in the masks)
FD_THRESHOLD_MM = 0.5          # Power et al. 2012
SCRUB_BEFORE = 1
SCRUB_AFTER = 2

# ---------------------------------------------------------------------------
# LEiDA (D6-D9, D12)
# ---------------------------------------------------------------------------
SEED = 42
K_RANGE = range(3, 21)         # D7: k = 3..20 (Paper 7)
ANCHOR_K = 5                   # plan expectation: 4-8 states, anchor k=5
N_INIT = 20                    # D9: k-means repetitions per k
SILHOUETTE_SUBSAMPLE = 10_000
STABILITY_RUNS = 10
STABILITY_SUBSAMPLE = 10_000
N_PERMUTATIONS = 5000          # D12 (stats scaffold for step 2)


# ---------------------------------------------------------------------------
# Step 2 - state-dependent controllability + transition energy (D2-*)
# ---------------------------------------------------------------------------
C_STAB = 1.0                   # D2-3: stabilization constant c (Paper 6 eq. 4)
RHO = 1.0                      # D2-8: control penalty (Paper 6 eq. 12)
T_HORIZON = 1.0                # D2-8: control horizon (Paper 6: T = 1)
DT = 0.001                     # D2-8: time step (Paper 6: 1000 steps)
N_STEPS = int(round(T_HORIZON / DT))
MIN_STATE_FRAMES = 20          # D2-4: min frames per subject-state for FC_s
STEP2_K_VALUES = [3, 5]        # D2-2: k=3 primary, k=5 sensitivity
STEP2_PRIMARY_K = 3
SYSTEM_GLOBAL = "global"

def system_name(s: int | str) -> str:
    """System key: 'global' or 'state{s}'."""
    return SYSTEM_GLOBAL if s == SYSTEM_GLOBAL else f"state{s}"

ENERGY_TRANSITIONS = ["T1", "T2", "T3", "T4"]   # D2-9 primary transitions
# Headline energy system per transition (F1). "dmn_like" is resolved from
# centroids with the same DMN-membership mean-|V| rule as script 11.
# Per-state T1/T2/T3 stay in transition_energy.csv as sensitivities.
ENERGY_PRIMARY_SYSTEM = {
    "T1": "global",     # Paper 6 static baseline A
    "T2": "dmn_like",   # A_s of the DMN-like state (matches T2 x0)
    "T3": "dmn_like",
    "T4": "global",     # only system computed for T4
}
# H1 TPM cells on the data-driven k=3 map:
# 0=VIS-DAN, 1=DMN-LIMBIC, 2=SMN-VAN. Do not invent an FPN/DAN state.
H1_TPM_CELLS = [(1, 0), (0, 1), (1, 2), (2, 1)]
H4_MIN_N_PER_GROUP = 10  # per-site Fisher-z requires n>=10 in both groups
ACTIVATION_NETWORKS = ["VIS", "SMN", "DAN", "VAN", "LIMBIC", "FPN", "DMN"]
ACTIVATION_SENSITIVITY_NETWORKS = ACTIVATION_NETWORKS + ["Subcortical"]
TARGET_AMPLITUDES = [0.5, 2.0]  # D2-9 sensitivity: amplitude scaling for T2

# ---------------------------------------------------------------------------
# Step-2 derived artifacts
# ---------------------------------------------------------------------------
STATE_FC_FILE = PROCESSED_DIR / "state_fc.h5"
STATE_FC_COVERAGE_CSV = PROCESSED_DIR / "state_fc_coverage.csv"
CONTROLLABILITY_CSV = PROCESSED_DIR / "controllability_metrics.csv"
CONTROLLABILITY_NODE_PARQUET = PROCESSED_DIR / "controllability_node.parquet"
TRANSITION_ENERGY_CSV = PROCESSED_DIR / "transition_energy.csv"
TRANSITION_ENERGY_NODE_PARQUET = PROCESSED_DIR / "transition_energy_node.parquet"
GROUP_STATS_CSV = PROCESSED_DIR / "group_stats_results.csv"
GROUP_STATS_COMPARISONS_CSV = PROCESSED_DIR / "group_stats_comparisons.csv"
STEP2_VALIDATION_JSON = PROCESSED_DIR / "step2_validation_results.json"
STEP2_VALIDATION_MD = PROCESSED_DIR / "step2_validation_report.md"
PAPER6_REPLICATION_JSON = PROCESSED_DIR / "paper6_replication.json"
STEP2_REPORT_HTML = PROCESSED_DIR / "step2_report.html"
STEP2_REPORT_MD = PROCESSED_DIR / "step2_report.md"

# ---------------------------------------------------------------------------
# Derived artifacts
# ---------------------------------------------------------------------------
SUBJECT_TABLE = PROCESSED_DIR / "subject_table.parquet"
TC_INDEX = PROCESSED_DIR / "timecourse_index.parquet"
TC_FILE = PROCESSED_DIR / "timecourses.h5"
QC_SUMMARY = PROCESSED_DIR / "qc_summary.parquet"
QC_SUMMARY_MOTION = PROCESSED_DIR / "qc_summary_motion.parquet"
KEEP_COL = "keep_motion_final"  # motion-controlled cohort (set when QC_SUMMARY_MOTION exists)
KEEP_SENS_COL = "keep_motion_final_sens"
K_SELECTION_CSV = PROCESSED_DIR / "k_selection.csv"
K_SELECTION_PNG = FIGURES_DIR / "k_selection.png"
STATE_METRICS_CSV = PROCESSED_DIR / "state_metrics.csv"
STATE_LABELS_PARQUET = PROCESSED_DIR / "state_labels.parquet"
STATE_LOADINGS_CSV = PROCESSED_DIR / "state_network_loadings.csv"
VALIDATION_REPORT = PROCESSED_DIR / "validation_report.md"
REPORT_HTML = PROCESSED_DIR / "step1_report.html"
REPORT_MD = PROCESSED_DIR / "step1_report.md"
MANIFEST = DOWNLOAD_DIR / "manifest.json"

YEO_NETWORKS = ["VIS", "SMN", "DAN", "VAN", "LIMBIC", "FPN", "DMN"]


# ---------------------------------------------------------------------------
# Step 3 - latent Neural ODE + Jacobian lens (D3-*)
# ---------------------------------------------------------------------------
STEP3_DIR = PROCESSED_DIR / "step3"
STEP3_DATASET = STEP3_DIR / "dataset.parquet"      # metadata + folds (long, per subject)
STEP3_DATA_H5 = STEP3_DIR / "train_data.h5"        # /v1 /labels /p /mask per subject (usable-aligned)
STEP3_CHECKPOINT = STEP3_DIR / "checkpoint.pt"
STEP3_JACOBIANS_H5 = STEP3_DIR / "jacobians.h5"
STEP3_AXES_NPZ = STEP3_DIR / "jspace_axes.npz"
STEP3_METRICS_CSV = STEP3_DIR / "jspace_metrics.csv"
STEP3_CONTROL_CSV = STEP3_DIR / "jspace_control.csv"
STEP3_ENERGY_CSV = STEP3_DIR / "latent_energy.csv"
STEP3_GROUP_STATS_CSV = STEP3_DIR / "jspace_group_stats.csv"
MODEL_COMPARISON_JSON = STEP3_DIR / "model_comparison.json"
ABLATION_JSON = STEP3_DIR / "ablation_results.json"
STEP3_VALIDATION_JSON = STEP3_DIR / "validation_results.json"
STEP3_VALIDATION_MD = STEP3_DIR / "validation_report.md"
STEP3_REPORT_MD = STEP3_DIR / "step3_report.md"
STEP3_REPORT_HTML = STEP3_DIR / "step3_report.html"

# Phase 6 model architecture (plan.md; D3-3..D3-10)
STEP3_K_COND = 3                 # k=3 primary conditioning states
STEP3_LATENT_DIMS = [8, 12, 16]  # D3-4 sensitivity grid
LATENT_DIM = 12                  # primary latent dimension
WINDOW_TR = 7                    # encoder window (sweep 5-15)
WINDOW_SWEEP = [5, 7, 9, 15]
HIDDEN_SIZE = 64                 # dynamics/decoder MLP width
HIDDEN_LAYERS = 2
ACTIVATION = "tanh"
ODESOLVER = "euler"              # short-shot protocol: fixed-step Euler in TR units
ODESOLVER_FALLBACK = "euler"
ODE_DT = 1.0                     # integration step in TR units
BETA_KL = 1e-3                   # D3-9 KL weight (avoid posterior collapse)
GAMMA_STATE = 0.05               # future-state CE (keep small; no future p leak)
ETA_SMOOTH = 1e-3                # smoothness weight on the integrated path
L_CONSIST_WEIGHT = 0.0           # full-scan consist; unused in short-shot protocol
L_NOW_WEIGHT = 0.5               # encoder recon of V1(t) (anchors z0)
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 80
EARLY_STOP_PATIENCE = 12
VAL_FRAC = 0.10                  # fixed stratified val split (Phase C)
N_FOLDS = 5                      # subject-level folds for step-4 classifiers
DELTA_PRIMARY = 2                # future horizon (TR); sensitivity 1,3
DELTA_SENSITIVITY = [1, 3]
# Short-shot training (replaces 200-step single shooting)
HORIZON_TR = 3                   # train rollout length; matches V1 memory (~6 TR)
SHOT_BATCH = 256                 # windows per step; this is what makes a GPU useful
PREDICT_RESIDUAL = True          # V1_hat(t+h) = unit(V1(t) + Dec(z_h))
CONDITION_MODE = "p0"            # p0 = hold p(t) during the shot; never feed p(t+h)
UNIT_SPHERE = True
EVAL_DELTAS = [1, 2, 3, 5]
STEP3_CHECKPOINT_FULL = STEP3_DIR / "checkpoint_full.pt"  # JLucid 1 (invalid full-scan)
STEP3_FORECAST_JSON = STEP3_DIR / "forecast_comparison.json"
# Artifact dirs (disk layout unchanged; code for 2.1/2.2/3.1–3.8 retired)
ODE21_DIR = STEP3_DIR / "ode21"
ODE21_FLIP_HEAD = ODE21_DIR / "flip_head.pt"
ODE21_SWEEP_JSON = ODE21_DIR / "gate_sweep.json"
FLIP_COS_THRESH = -0.5  # persist cosine below this is a flip-type jump
# JLucid 2.2 — first "J3" persist/jump gate (prettier 2.1; not the thesis NODE)
ODE22_DIR = STEP3_DIR / "ode22"
ODE22_CHECKPOINT = ODE22_DIR / "checkpoint.pt"
ODE22_EVAL_JSON = ODE22_DIR / "forecast.json"
GATE_BCE_WEIGHT = 0.5
STAY_PULL_WEIGHT = 0.25
# JLucid 2.3 — stay-only velocity field + precision invert (thesis-capable 2-family)
ODE23_DIR = STEP3_DIR / "ode23"
ODE23_CHECKPOINT = ODE23_DIR / "checkpoint.pt"
ODE23_EVAL_JSON = ODE23_DIR / "forecast.json"
ODE23_JSPACE_JSON = ODE23_DIR / "jspace.json"
ODE23_STAY_FLOOR = 0.905  # 2.1 win-bar stay; 2.3 must clear this on val
# JLucid 3 — stay-only field + discrete routing (thesis model)
ODE3_DIR = STEP3_DIR / "ode3"
ODE3_CHECKPOINT = ODE3_DIR / "checkpoint.pt"
ODE3_EVAL_JSON = ODE3_DIR / "forecast.json"
ODE3_FEATURES_H5 = ODE3_DIR / "features.h5"
# JLucid 3.x isolation forks (do not merge into ode/ unless a thesis-metric win)
ODE31_DIR = STEP3_DIR / "ode31"
ODE31_CHECKPOINT = ODE31_DIR / "checkpoint.pt"
ODE31_EVAL_JSON = ODE31_DIR / "forecast.json"
ODE31_CELLS_JSON = ODE31_DIR / "cells.json"
ODE32_DIR = STEP3_DIR / "ode32"
ODE32_CHECKPOINT = ODE32_DIR / "checkpoint.pt"
ODE32_EVAL_JSON = ODE32_DIR / "forecast.json"
ODE32_TABLE_JSON = ODE32_DIR / "variant_table.json"
ODE33_DIR = STEP3_DIR / "ode33"
ODE33_CHECKPOINT = ODE33_DIR / "checkpoint.pt"
ODE33_EVAL_JSON = ODE33_DIR / "forecast.json"
ODE33_PERSIST_JSON = ODE33_DIR / "persist_dwell.json"
ODE34_DIR = STEP3_DIR / "ode34"
ODE34_CHECKPOINT = ODE34_DIR / "checkpoint.pt"
ODE34_PEEK_CHECKPOINT = ODE34_DIR / "checkpoint_fold0.pt"
ODE34_EVAL_JSON = ODE34_DIR / "forecast.json"
ODE34_A_NPY = ODE34_DIR / "A_states.npy"
ODE35_DIR = STEP3_DIR / "ode35"
ODE35_CHECKPOINT = ODE35_DIR / "checkpoint.pt"
ODE35_EVAL_JSON = ODE35_DIR / "forecast.json"
ODE36_DIR = STEP3_DIR / "ode36"
ODE36_CHECKPOINT = ODE36_DIR / "checkpoint.pt"
ODE36_EVAL_JSON = ODE36_DIR / "forecast.json"
ODE36_TABLE_JSON = ODE36_DIR / "variant_table.json"
ODE36_PARETO_JSON = ODE36_DIR / "pareto.json"
ODE37_DIR = STEP3_DIR / "ode37"
ODE37_CHECKPOINT = ODE37_DIR / "checkpoint.pt"
ODE37_PEEK_CHECKPOINT = ODE37_DIR / "checkpoint_fold0.pt"
ODE37_EVAL_JSON = ODE37_DIR / "forecast.json"
ODE37_CELLS_JSON = ODE37_DIR / "cells.json"
ODE37_JSPACE_JSON = ODE37_DIR / "jspace.json"
ODE37_A_NPY = ODE37_DIR / "A_states.npy"
ODE38_DIR = STEP3_DIR / "ode38"
ODE38_EVAL_JSON = ODE38_DIR / "forecast.json"
# JLucid 4 — stay field + identifiable J + packed dest/when
ODE4_DIR = STEP3_DIR / "ode4"
ODE4_CHECKPOINT = ODE4_DIR / "checkpoint.pt"
ODE4_EVAL_JSON = ODE4_DIR / "forecast.json"
ODE4_JSPACE_JSON = ODE4_DIR / "jspace.json"
ODE4_A_NPY = ODE4_DIR / "A_states.npy"

# J-space downstream (JLucid 4 A(p); §8–9 of .agents/step-3.md)
JSPACE_K = 3                     # top-k for C_k, A_J, and matched ablations
JSPACE_HURWITZ_MARGIN = 1e-3     # symmetrize + shift; not Paper-6 FC stabilize()
JSPACE_DMN_EPS = 1e-8            # L_DMN / (L_FPN/DAN + eps)

# Routing / future-output Jacobian (plan Phase 7; not A(p))
JFUTURE_DIR = STEP3_DIR / "jfuture"
JFUTURE_METRICS_CSV = JFUTURE_DIR / "metrics.csv"
JFUTURE_GROUP_STATS_CSV = JFUTURE_DIR / "group_stats.csv"
JFUTURE_H4_JSON = JFUTURE_DIR / "h4.json"
JFUTURE_ABLATION_JSON = JFUTURE_DIR / "ablation.json"
JFUTURE_JACOBIANS_H5 = JFUTURE_DIR / "jacobians.h5"
JFUTURE_REPORT_MD = JFUTURE_DIR / "report.md"
JFUTURE_VALIDATION_JSON = JFUTURE_DIR / "validation.json"
JFUTURE_MAX_STAY_OUT = 32        # sampled stay frames for 90×12 J_out
OCC_INDEP_R_MAX = 0.50           # |ρ(C_k, occupancy)| above this = still collapsed

STEP3_INPUT_DIM = 90             # V1(t) dimension
STEP3_OUTPUT_DIM = STEP3_K_COND + len(YEO_NETWORKS) + 1  # 3 state probs + 7 Yeo + Subcortical
