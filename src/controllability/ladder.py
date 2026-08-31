"""Advisor ladder — ordered exhaustive search per advisor advice.
Sequence:
 1. whole-brain Schaefer100 c=1 none
 2. ROI subsets (frontal/parietal/dmn/combined) on same atlas
 3. Atlas variants (aal90, schaefer100, schaefer200, schaefer300)
 4. FC norms (none, fisher, global_zscore)
 5. c sweep (0.5,1,2,5) + use_abs toggle
 6. dataset switch flagged
Advisors verbatim: '不在考虑全脑的ROI，只考虑额叶、顶叶或者默认模式网络… 把它们当成一个网络'
                 '可以都试试 90 100 300' '有没有做归一化，或则z-score' '尝试调整A的计算中的一些超参数（参考或者改进公式4）'
                 '如果上述方法都不成功，再考虑换数据集'
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from .config import PipelineConfig
from .fc import compute_fc, normalize_fc
from .controllability import build_A, avg_modal_controllability
from .parcellations import get_roi_indices, get_aal_roi_indices_aal90, get_network_labels
from .stats import permutation_test, welch_ttest, bh_fdr
from .preprocessing import preprocess_timeseries_placeholder

def run_one_config(timeseries_list, labels, atlas: str, roi: str, c: float, fc_norm: str, use_abs: bool, n_perm: int, seed: int):
    """Run one ladder rung: compute controllability + stats."""
    # ROI indices — handle synthetic N mismatch (e.g., synthetic 100 vs schaefer300 300)
    # Use effective N from data; if atlas expects more ROIs than data provides, truncate labels/indices.
    N_data = timeseries_list[0].shape[1] if len(timeseries_list)>0 else 100
    try:
        raw_labels = get_network_labels(atlas)
    except Exception:
        raw_labels = ["DMN"]*N_data
    # If data N != atlas N, adapt: slice or tile network labels to data size for smoke tests
    if len(raw_labels) != N_data:
        if len(raw_labels) > N_data:
            # Truncate (schaefer300 vs 100-ROI synthetic) — keeps first N_data parcels' networks
            raw_labels = raw_labels[:N_data]
        else:
            # Pad by repeating pattern (e.g., aal90 90 vs data 100)
            reps = int(np.ceil(N_data / len(raw_labels)))
            raw_labels = (raw_labels * reps)[:N_data]
    # Now derive ROI indices from adapted labels
    if atlas == "aal90" and roi != "whole":
        # For adapted case, we need to map via AAL frontal sets adapted to N_data size
        # Fallback to network-based frontal/parietal if size mismatch
        if N_data == 90:
            idx = get_aal_roi_indices_aal90(roi)
        else:
            idx = get_roi_indices(atlas, roi)  # network proxy, now using adapted labels length
            # But get_roi_indices uses fresh get_network_labels which we adapted; re-derive directly:
            nets_arr = np.array(raw_labels)
            if roi == "frontal": idx = np.where(np.isin(nets_arr, ["FPN","SAL"]))[0]
            elif roi == "parietal": idx = np.where(np.isin(nets_arr, ["DAN","SOM"]))[0]
            elif roi == "dmn": idx = np.where(nets_arr == "DMN")[0]
            elif roi == "combined": idx = np.where(np.isin(nets_arr, ["FPN","DAN","DMN"]))[0]
            elif roi == "whole": idx = np.arange(len(nets_arr))
            else: idx = np.arange(len(nets_arr))
    else:
        # Use adapted labels to compute indices directly to avoid mismatch
        nets_arr = np.array(raw_labels)
        if roi == "whole": idx = np.arange(len(nets_arr))
        elif roi == "frontal": idx = np.where(np.isin(nets_arr, ["FPN","SAL"]))[0]
        elif roi == "parietal": idx = np.where(np.isin(nets_arr, ["DAN","SOM"]))[0]
        elif roi == "dmn": idx = np.where(nets_arr == "DMN")[0]
        elif roi == "combined": idx = np.where(np.isin(nets_arr, ["FPN","DAN","DMN"]))[0]
        else: idx = np.arange(len(nets_arr))
    # Ensure idx is within data bounds
    idx = idx[idx < N_data]
    if len(idx) == 0:
        idx = np.arange(N_data)
    # per-subject FC → A → controllability
    avg_individual = []
    modal_individual = []
    avg_networks = {}  # net -> list
    modal_networks = {}
    avg_nodes_all = []
    modal_nodes_all = []
    network_labels = raw_labels
    # Determine network names present in the ROI subset
    nets_in_roi = sorted(set(np.array(network_labels)[idx].tolist()))
    for ts in timeseries_list:
        # ts is (T,N_total); slice to roi subset
        ts_roi = ts[:, idx]
        # minimal preprocess: drop first 10, detrend, bandpass placeholder
        # For synthetic, this is near no-op but validates pipeline
        # ts_roi = preprocess_timeseries_placeholder(ts_roi, drop_first=0, detrend=False)
        fc = compute_fc(ts_roi, zscore_timeseries=False)
        fc = normalize_fc(fc, method=fc_norm)
        A = build_A(fc, c=c, use_abs=use_abs)
        avg_node, modal_node, vals, vecs = avg_modal_controllability(A)
        avg_nodes_all.append(avg_node)
        modal_nodes_all.append(modal_node)
        avg_individual.append(float(np.mean(avg_node)))
        modal_individual.append(float(np.mean(modal_node)))
        # network-level: group avg_node by net within ROI subset
        nets_roi = np.array(network_labels)[idx]
        for net in nets_in_roi:
            mask = nets_roi == net
            if mask.sum() == 0: continue
            if net not in avg_networks:
                avg_networks[net] = []
                modal_networks[net] = []
            avg_networks[net].append(float(np.mean(avg_node[mask])))
            modal_networks[net].append(float(np.mean(modal_node[mask])))
    avg_individual = np.array(avg_individual)
    modal_individual = np.array(modal_individual)
    labels = np.asarray(labels)
    hc = labels == 0
    adhd = labels == 1
    # stats: individual level
    res = {}
    # Welch + perm
    res["individual"] = {
        "avg": {**welch_ttest(avg_individual[hc], avg_individual[adhd]), **permutation_test(avg_individual[hc], avg_individual[adhd], n_perm=n_perm, seed=seed)},
        "modal": {**welch_ttest(modal_individual[hc], modal_individual[adhd]), **permutation_test(modal_individual[hc], modal_individual[adhd], n_perm=n_perm, seed=seed+1)},
    }
    # network level with FDR
    if avg_networks:
        nets = list(avg_networks.keys())
        p_avgs = []
        p_mods = []
        for net in nets:
            pa = permutation_test(np.array(avg_networks[net])[hc], np.array(avg_networks[net])[adhd], n_perm=n_perm, seed=seed)
            pm = permutation_test(np.array(modal_networks[net])[hc], np.array(modal_networks[net])[adhd], n_perm=n_perm, seed=seed+2)
            p_avgs.append(pa["p_perm"]); p_mods.append(pm["p_perm"])
        from .stats import bh_fdr
        pcorr_a, rej_a = bh_fdr(np.array(p_avgs), q=0.05)
        pcorr_m, rej_m = bh_fdr(np.array(p_mods), q=0.05)
        res["network"] = {}
        for i, net in enumerate(nets):
            res["network"][net] = {
                "p_perm_avg": float(p_avgs[i]), "p_fdr_avg": float(pcorr_a[i]), "sig_avg": bool(rej_a[i]),
                "p_perm_modal": float(p_mods[i]), "p_fdr_modal": float(pcorr_m[i]), "sig_modal": bool(rej_m[i]),
                "mean_hc_avg": float(np.mean(np.array(avg_networks[net])[hc])),
                "mean_adhd_avg": float(np.mean(np.array(avg_networks[net])[adhd])),
                "mean_hc_modal": float(np.mean(np.array(modal_networks[net])[hc])),
                "mean_adhd_modal": float(np.mean(np.array(modal_networks[net])[adhd])),
            }
    # return concise summary for CSV
    summary = {
        "atlas": atlas, "roi": roi, "c": c, "fc_norm": fc_norm, "use_abs": use_abs,
        "n_rois": len(idx),
        "p_perm_avg_ind": res["individual"]["avg"]["p_perm"],
        "p_perm_modal_ind": res["individual"]["modal"]["p_perm"],
        "t_avg": res["individual"]["avg"]["t"],
        "t_modal": res["individual"]["modal"]["t"],
        "cohens_d_avg": res["individual"]["avg"]["cohens_d"],
        "cohens_d_modal": res["individual"]["modal"]["cohens_d"],
        "sig_any": any(v["sig_avg"] or v["sig_modal"] for v in res.get("network", {}).values()) or res["individual"]["avg"]["p_perm"]<0.05 or res["individual"]["modal"]["p_perm"]<0.05,
    }
    return summary, res, {"avg_individual": avg_individual, "modal_individual": modal_individual}

def run_ladder(timeseries_list, labels, out_csv: str | None = None, verbose: bool = True, n_perm: int = 2000):
    """Exhaustive advisor ladder — returns DataFrame of all rungs.
    Args:
        n_perm: permutations per config (paper 6 = 5000; CI/smoke uses 200-500 for speed)
    """
    cfgs = []
    # Step 1: whole-brain Schaefer100 c=1 none (paper exact)
    cfgs.append(("schaefer100","whole",1.0,"none",True))
    # Step 2: ROI subsets same atlas/c
    for roi in ["frontal","parietal","dmn","combined"]:
        cfgs.append(("schaefer100", roi, 1.0, "none", True))
    # Step 3: Atlas variants whole-brain
    for atlas in ["aal90","schaefer200","schaefer300"]:
        cfgs.append((atlas,"whole",1.0,"none",True))
    # Atlas + ROI combos (focused: frontal/dm n on each atlas)
    for atlas in ["schaefer300","aal90"]:
        for roi in ["frontal","dmn","combined"]:
            cfgs.append((atlas,roi,1.0,"none",True))
    # Step 4: FC norms
    for norm in ["fisher","global_zscore"]:
        cfgs.append(("schaefer100","whole",1.0,norm,True))
        cfgs.append(("schaefer100","combined",1.0,norm,True))
    # Step 5: c sweep + use_abs toggle
    for c in [0.5,2.0,5.0]:
        cfgs.append(("schaefer100","whole",c,"none",True))
        cfgs.append(("schaefer100","combined",c,"none",True))
    # robustness: without abs
    cfgs.append(("schaefer100","whole",1.0,"none",False))
    cfgs.append(("schaefer100","combined",1.0,"none",False))
    rows = []
    for atlas, roi, c, norm, use_abs in cfgs:
        summ, _, _ = run_one_config(timeseries_list, labels, atlas=atlas, roi=roi, c=c, fc_norm=norm, use_abs=use_abs, n_perm=n_perm, seed=42)
        rows.append(summ)
        if verbose:
            flag = "★" if summ["sig_any"] else " "
            print(f"{flag} {atlas:12} roi={roi:8} c={c:<4} norm={norm:13} abs={str(use_abs):5} | p_avg={summ['p_perm_avg_ind']:.3f} p_mod={summ['p_perm_modal_ind']:.3f} d_avg={summ['cohens_d_avg']:.2f} d_mod={summ['cohens_d_modal']:.2f}")
    df = pd.DataFrame(rows)
    if out_csv:
        Path = __import__("pathlib").Path
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[ladder] saved {out_csv} ({len(df)} configs)")
    return df
