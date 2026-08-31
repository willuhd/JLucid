"""CLI — single entry point for synthetic / adhd200 / ucla ladder.
Intel-Mac friendly: streams subjects, torch float64 CPU, n_jobs=1 default.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .datasets import generate_synthetic_timeseries, load_adhd200
from .ladder import run_one_config, run_ladder
from .config import PipelineConfig

def main():
    p = argparse.ArgumentParser(description="ADHD Controllability — Paper 6 replication (torch 2.13 CPU)")
    p.add_argument("--mode", choices=["synthetic","adhd200","ucla"], default="synthetic", help="data source")
    p.add_argument("--n-subjects", type=int, default=40, help="for synthetic; for real caps download")
    p.add_argument("--atlas", type=str, default="schaefer100", choices=["aal90","schaefer100","schaefer200","schaefer300","schaefer400"])
    p.add_argument("--roi", type=str, default="whole", choices=["whole","frontal","parietal","dmn","combined"])
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--fc-norm", type=str, default="none", choices=["none","fisher","global_zscore","row_zscore"], dest="fc_norm")
    p.add_argument("--use-abs", action="store_true", default=True, help="use |FC| (default true); --no-use-abs to disable")
    p.add_argument("--no-use-abs", dest="use_abs", action="store_false")
    p.add_argument("--n-perm", type=int, default=5000, dest="n_perm")
    p.add_argument("--out", type=str, default="results/run")
    p.add_argument("--ladder", action="store_true", help="run full advisor ladder (exhaustive)")
    p.add_argument("--group-diff", type=float, default=0.0, help="synthetic group difference magnitude (0=null, 0.3=moderate, 0.6=strong)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load data ----
    if args.mode == "synthetic":
        # map atlas to n_rois
        n_rois_map = {"aal90":90,"schaefer100":100,"schaefer200":200,"schaefer300":300,"schaefer400":400}
        n_rois = n_rois_map[args.atlas]
        ts_list, labels, meta = generate_synthetic_timeseries(n_subjects=args.n_subjects, n_rois=n_rois, T=150, seed=args.seed, group_diff=args.group_diff)
        print(f"[cli] synthetic: {args.n_subjects} subjects, {n_rois} ROIs, group_diff={args.group_diff}, atlas={args.atlas}")
        if args.group_diff==0:
            print("[cli] NOTE: group_diff=0 → expect null (honest pipeline validation). Use --group-diff 0.4 to test detection power.")
    elif args.mode == "adhd200":
        # Synthetic fallback unless user has downloaded; we try fetch but default to synthetic in CI
        result = load_adhd200(data_dir="data/adhd200", n_subjects=args.n_subjects if args.n_subjects!=40 else 40, download=False, seed=args.seed)
        # load_adhd200 with download=False returns synthetic
        if isinstance(result, tuple) and len(result)==3 and isinstance(result[0], list):
            ts_list, labels, meta = result
        else:
            # real bunch handling — for now fallback
            ts_list, labels, meta = generate_synthetic_timeseries(n_subjects=args.n_subjects, n_rois=100, seed=args.seed)
    elif args.mode == "ucla":
        from .datasets import load_ucla
        ts_list, labels, meta = load_ucla(n_subjects=args.n_subjects, seed=args.seed)
    else:
        raise ValueError(args.mode)

    if args.ladder:
        df = run_ladder(ts_list, labels, out_csv=str(out / "ladder_results.csv"), verbose=True, n_perm=args.n_perm)
        # highlight best hit
        hits = df[df["sig_any"]]
        summary = {
            "n_configs": len(df),
            "n_hits": len(hits),
            "best_p_avg": float(df["p_perm_avg_ind"].min()),
            "best_p_modal": float(df["p_perm_modal_ind"].min()),
        }
        print(f"\n[ladder] Summary: {summary}")
        if len(hits):
            print("[ladder] HITS (FDR-significant configs):")
            print(hits.to_string(index=False))
        else:
            print("[ladder] No config reached p<0.05 — advisor says either signal hidden deeper or absent. Consider: larger N, different preprocessing z-score, or dataset switch (UCLA ↔ ADHD-200).")
        # save summary
        import json as js
        (out / "ladder_summary.json").write_text(js.dumps(summary, indent=2))
    else:
        summ, res, raw = run_one_config(ts_list, labels, atlas=args.atlas, roi=args.roi, c=args.c, fc_norm=args.fc_norm, use_abs=args.use_abs, n_perm=args.n_perm, seed=args.seed)
        print(f"\n[one] {args.atlas} roi={args.roi} c={args.c} norm={args.fc_norm} abs={args.use_abs}")
        print(f"  individual avg  p_perm={summ['p_perm_avg_ind']:.4f} t={summ['t_avg']:.2f} d={summ['cohens_d_avg']:.2f} n_rois={summ['n_rois']}")
        print(f"  individual modal p_perm={summ['p_perm_modal_ind']:.4f} t={summ['t_modal']:.2f} d={summ['cohens_d_modal']:.2f}")
        # save
        import json as js
        (out / "one_config_summary.json").write_text(js.dumps(summ, indent=2))
        (out / "one_config_full.json").write_text(js.dumps(res, indent=2, default=str))
        pd.DataFrame([summ]).to_csv(out / "one_config.csv", index=False)
        print(f"[cli] wrote {out}/one_config*.json/csv")

if __name__ == "__main__":
    main()
