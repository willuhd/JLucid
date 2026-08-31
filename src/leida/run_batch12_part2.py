#!/usr/bin/env python3
"""exp/37 Batch 12 part 2 — Gate 1 (oracle) + Gate 2 (honest battery) for all surviving
batch-12 variants. PREREG: SIEVE_TABLE.md batch-12 section.
Variants: CGATE(70) EDIFF(3) DYNCTRL(15) EMAP(18) [Gate-0 survivors only]
          + Dp(15, theorist cache DDF) + Cp(14, CSF) + Ap(5, occF x age interaction).
Estimators: primary pooled r_sc (new-quantity convention per batch-5 note), 5 seeds averaged
  for the nested-CV-free LOSO (no seed dependence in LOSO itself; seed only affects alpha
  inner folds). Per prereg the primary statistic is the 5-seed mean.
Targets: Inatt, Hyper (T-cohort n=336). IQ positive control (n=795).
Familywise null: 200 perms within site, max-stat over ALL cells (surviving variants x 2
  symptom targets + IQ canary cells counted in the SAME max-stat family per prereg).
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

BASE = Path("/Volumes/thinkplus/Code/JLucid")
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "results/37_leida"
ALPHAS = [0.1, 1, 10, 100, 1000, 3e3, 1e4, 3e4, 1e5]
import time as _t; T0 = _t.time()
def log(m): print(f"[{_t.time()-T0:7.1f}s] {m}", flush=True)

gate0 = json.load(open(OUT / "batch12_gate0.json"))
Z = np.load(OUT / "batch12_feats.npz")
Z5 = np.load(BASE / "results/pilot_geometry/pilot7_d5_raw.npz")
DDF = Z5["DDF"]; CSF = Z5["CSF"]; occF = Z5["occF"]; agesF = Z5["ages"]

from controllability.datasets_cc200 import load_cc200
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites_all = np.array(meta_all["sites"])
ph = pd.read_csv(BASE / "data/adhd200/adhd200_preprocessed_phenotypics.tsv", sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
for c in ["Inattentive", "Hyper/Impulsive"]:
    ph[c + "_n"] = pd.to_numeric(ph[c], errors="coerce")
    ph.loc[ph[c + "_n"] == -999, c + "_n"] = np.nan
ph["f4"] = pd.to_numeric(ph["Full4 IQ"], errors="coerce")
ph.loc[ph["f4"] == -999, "f4"] = np.nan
phm = ph.set_index("ScanDir ID")

keep = [i for i, sid in enumerate(ids_all)
        if sites_all[i] in (3, 5, 6) and sid in phm.index
        and str(phm.loc[sid, "ADHD Measure"]) in ("2", "3")
        and not pd.isna(phm.loc[sid, "Inattentive_n"])]
keep = np.array(keep)
y_in = np.array([float(phm.loc[ids_all[i], "Inattentive_n"]) for i in keep])
y_hy = np.array([float(phm.loc[ids_all[i], "Hyper/Impulsive_n"]) for i in keep])
sites_tc = sites_all[keep]
ages_tc = np.array(Z5["ages"])[keep]
keep_iq = np.array([i for i, sid in enumerate(ids_all) if sid in phm.index and not pd.isna(phm.loc[sid, "f4"])])
y_iq = np.array([float(phm.loc[ids_all[i], "f4"]) for i in keep_iq])
sites_iq = sites_all[keep_iq]
log(f"T-cohort n={len(keep)}; IQ n={len(keep_iq)}")

# assemble variant matrices (all 872 rows, then subset)
VAR = {}
alive4 = [nm for nm in gate0 if gate0[nm]["frac_ge_030"] >= 0.5]
log(f"Gate-0 survivors among t4: {alive4}")
for nm in alive4:
    VAR[nm] = Z[f"{nm}_full"]
VAR["Dp"] = DDF
VAR["Cp"] = CSF
# A': occF x age interaction (cohort-centered per t5 spec)
occ = Z5["occF"]
age_all = np.array(Z5["ages"], float)
occ_c = occ - occ.mean(0)
age_z = (age_all - age_all.mean()) / age_all.std()
VAR["Ap"] = occ_c * age_z[:, None]

def loso(X, yv, sv, seed, fixed_alpha=None):
    pred = np.full(len(yv), np.nan); sel = []
    rng = np.random.RandomState(seed)
    for s in np.unique(sv):
        te = sv == s; tr = ~te
        Xtr, ytr = X[tr], yv[tr]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-12
        ymu, ysd = ytr.mean(), ytr.std() + 1e-12
        if fixed_alpha is None:
            inner = np.array_split(rng.permutation(len(ytr)), 3)
            best_a, best_s = ALPHAS[0], -1
            for a in ALPHAS:
                sc = []
                for iv in inner:
                    tr2 = np.setdiff1d(np.arange(len(ytr)), iv)
                    mu2 = Xtr[tr2].mean(0); sd2 = Xtr[tr2].std(0) + 1e-12
                    wz = (Xtr[tr2]-mu2)/sd2
                    w = np.linalg.solve(wz.T@wz + a*np.eye(wz.shape[1]), wz.T@ytr[tr2])
                    pr = ((Xtr[iv]-mu2)/sd2)@w
                    sc.append(np.corrcoef(pr, ytr[iv])[0,1] if np.std(pr) > 1e-12 else 0)
                m_ = np.mean(sc)
                if m_ > best_s: best_s, best_a = m_, a
            sel.append(best_a)
        else:
            sel.append(fixed_alpha)
        Xz = (Xtr-mu)/sd
        w = np.linalg.solve(Xz.T@Xz + sel[-1]*np.eye(Xz.shape[1]), Xz.T@(ytr-ymu))
        pred[te] = ((X[te]-mu)/sd)@w * ysd + ymu
    return pred, sel

def scr(pred, obs, sv):
    pc = pred.copy(); oc = obs.copy()
    for s in np.unique(sv):
        m = sv == s
        if m.sum() >= 2: pc[m] -= pc[m].mean(); oc[m] -= oc[m].mean()
    return float(np.corrcoef(pc, oc)[0,1]) if np.std(pc) > 1e-12 else 0.0

# ---------------- Gate 1: oracle (in-sample) ----------------
log("Gate 1 (oracle) ...")
def oracle(X, yv):
    best = (-9, None)
    for a in ALPHAS:
        mu = X.mean(0); sd = X.std(0) + 1e-12
        Xz = (X-mu)/sd
        w = np.linalg.solve(Xz.T@Xz + a*np.eye(Xz.shape[1]), Xz.T@yv)
        r = np.corrcoef(Xz@w, yv)[0,1]
        if r > best[0]: best = (r, a)
    return best

gate1 = {}
for vn, Xall in VAR.items():
    X = Xall[keep]
    r_i, a_i = oracle(X, y_in)
    r_h, a_h = oracle(X, y_hy)
    Xq = Xall[keep_iq]
    r_q, a_q = oracle(Xq, y_iq)
    gate1[vn] = {"inatt": r_i, "hyper": r_h, "iq": r_q, "alpha_inatt": a_i}
    print(f"  {vn:9s} oracle Inatt={r_i:+.3f} Hyper={r_h:+.3f} IQ={r_q:+.3f}")
DEAD1 = [v for v in gate1 if max(gate1[v]["inatt"], gate1[v]["hyper"]) < 0.15]
log(f"Gate 1 DEAD: {DEAD1}")

# ---------------- Gate 2: honest battery ----------------
log("Gate 2 (LOSO battery, 5 seeds, then 200-perm familywise null) ...")
cells2 = [v for v in VAR if v not in DEAD1]
targets = {"Inatt": y_in, "Hyper": y_hy}
G2 = {}
for vn in cells2:
    X = VAR[vn][keep]
    for tn, yv in targets.items():
        rs = []
        for seed in range(5):
            pred, sel = loso(X, yv, sites_tc, seed)
            rs.append(scr(pred, yv, sites_tc))
        G2[(vn, tn)] = {"r_sc_mean": float(np.mean(rs)), "r_sc_seeds": [round(v,4) for v in rs]}
        print(f"  {vn} x {tn}: r_sc={np.mean(rs):+.4f} (seeds {['%.2f'%v for v in rs]})")

# familywise null over all G2 cells + IQ canary cells
def iq_canary_stat():
    out = {}
    for vn in cells2:
        Xq = VAR[vn][keep_iq]
        rs = []
        for seed in range(5):
            pred, sel = loso(Xq, y_iq, sites_iq, seed)
            rs.append(scr(pred, y_iq, sites_iq))
        out[vn] = float(np.mean(rs))
    return out
IQc = iq_canary_stat()
log(f"IQ canary r_sc: {[(k, round(v,3)) for k, v in IQc.items()]}")

log("fw null 200 perms ...")
rng = np.random.RandomState(20260907)
null_max = []
for pi in range(200):
    yp_in = y_in.copy(); yp_hy = y_hy.copy()
    for s in np.unique(sites_tc):
        m = sites_tc == s
        yp_in[m] = yp_in[m][rng.permutation(m.sum())]
        yp_hy[m] = yp_hy[m][rng.permutation(m.sum())]
    yq = y_iq.copy()
    for s in np.unique(sites_iq):
        m = sites_iq == s
        yq[m] = yq[m][rng.permutation(m.sum())]
    mx = 0.0
    for vn in cells2:
        X = VAR[vn][keep]
        for yv, yp in ((y_in, yp_in), (y_hy, yp_hy)):
            pred, _ = loso(X, yp, sites_tc, 0)
            mx = max(mx, abs(scr(pred, yp, sites_tc)))
        pred, _ = loso(VAR[vn][keep_iq], yq, sites_iq, 0)
        mx = max(mx, abs(scr(pred, yq, sites_iq)))
    null_max.append(mx)
    if (pi+1) % 50 == 0: log(f"  perm {pi+1}")
null_max = np.array(null_max)

results = {"gate0": gate0, "gate1": gate1, "gate1_dead": DEAD1, "gate2": {},
           "iq_canary": IQc, "null_q95": float(np.quantile(null_max, 0.95))}
for (vn, tn), cell in G2.items():
    obs = abs(cell["r_sc_mean"])
    p = float((np.sum(null_max >= obs) + 1) / 201)
    cell["fw_p"] = p
    results["gate2"][f"{vn}|{tn}"] = cell
    print(f"  {vn} x {tn}: obs={cell['r_sc_mean']:+.4f} fw_p={p:.4f}")
alive = [(k, v) for k, v in results["gate2"].items() if v["fw_p"] < 0.05 and abs(v["r_sc_mean"]) > 0.15]
results["ALIVE"] = [k for k, _ in alive]
log(f"ALIVE cells: {results['ALIVE']}")
json.dump(results, open(OUT / "batch12_results.json", "w"), indent=1, default=str)
log("saved batch12_results.json")
