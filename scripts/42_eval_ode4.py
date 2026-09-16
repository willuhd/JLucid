#!/usr/bin/env python3
"""Evaluate JLucid 4: stay vs persist/half-step, J, packed discrete."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import h5py
import numpy as np
import torch

from jlucid.config import (
    ODE3_FEATURES_H5,
    ODE4_A_NPY,
    ODE4_CHECKPOINT,
    ODE4_DIR,
    ODE4_EVAL_JSON,
    ODE4_JSPACE_JSON,
    STEP3_CHECKPOINT,
    STEP3_DATA_H5,
    WINDOW_TR,
)
from jlucid.ode.jlucid4.discrete import evaluate_discrete
from jlucid.ode.jlucid4.eval import evaluate, format_table
from jlucid.ode.jlucid4.jacobians import jacobian_smoke, jspace_peek, spectral_report
from jlucid.ode.jlucid4.models import ThesisModel
from jlucid.ode.jlucid4.shots import load_bundle
from jlucid.utils.logging import get_logger


def _val_ids(ckpt: dict) -> list[str]:
    ids = list(ckpt.get("result", {}).get("val_ids") or [])
    if ids:
        return [str(s) for s in ids]
    if STEP3_CHECKPOINT.exists():
        j2 = torch.load(STEP3_CHECKPOINT, map_location="cpu", weights_only=False)
        ids = list(j2.get("result", {}).get("val_ids") or [])
        if ids:
            return [str(s) for s in ids]
    raise SystemExit("no val_ids in checkpoint")


def _load_model(path: Path, device) -> tuple[ThesisModel, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    model = ThesisModel(
        latent=int(cfg.get("latent", 12)),
        window=int(cfg.get("window", WINDOW_TR)),
        gamma_mode=str(cfg.get("gamma_mode", "learned")),
        gamma_value=float(cfg.get("gamma_value", 0.5)),
        decode_mode=str(cfg.get("decode_mode", "recursive")),
        small_decode_init=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.switch_tau = float(ckpt.get("switch_tau", model.switch_tau))
    model._discrete_packed = bool(ckpt.get("discrete_packed", False))
    if "switch_mean" in ckpt["state_dict"]:
        model._discrete_packed = True
    if ckpt.get("centroids") is not None:
        model.centroids = np.asarray(ckpt["centroids"])
    model.eval()
    return model, ckpt


def _rows(ev: dict) -> list[dict]:
    rows = []
    for h, r in ev.get("horizons", {}).items():
        rows.append({
            "delta": int(h),
            "n_shots": r["n_shots"],
            "n_subjects": r["n_subjects"],
            "model_subject": r["model_subject"],
            "persist_subject": r["persist_subject"],
            "half_subject": r.get("half_subject"),
            "gain_subject": r["gain_subject"],
            "gain_vs_half_subject": r.get("gain_vs_half_subject"),
            "gain_subject_ci95": r.get("gain_subject_ci95"),
            "gain_vs_half_ci95": r.get("gain_vs_half_ci95"),
            "gain_excludes_0": r.get("gain_excludes_0"),
            "frac_subjects_beat_persist": r.get("frac_subjects_beat_persist"),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ODE4_CHECKPOINT)
    ap.add_argument("--out", type=Path, default=ODE4_EVAL_JSON)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--horizons", type=str, default="1,2,3")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    log = get_logger("ode4_eval",
                     ROOT / "data" / "processed" / "logs" / "42_eval_ode4.log")
    device = torch.device(args.device)
    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    ODE4_DIR.mkdir(parents=True, exist_ok=True)
    if not args.ckpt.exists():
        raise SystemExit(f"missing {args.ckpt}; run scripts/41_train_ode4.py")

    model, ckpt = _load_model(args.ckpt, device)
    val_ids = _val_ids(ckpt)
    report: dict = {
        "protocol": {
            "stay_field": "z_{t+1}=z+A(p)z+b(p)",
            "V_base": "unit(V+γ(V-Vprev))",
            "Vhat": "unit(V_base+Dec(z))",
            "A(p)": "sum_s p_s A_s",
            "eps": 0.0,
            "decode_mode": ckpt.get("config", {}).get("decode_mode", "recursive"),
            "headline": "stay vs persist and vs 0-param half-step; packed dest+when",
            "not_trained": ["mixed V1 cosine", "invert"],
        },
        "val_n": len(val_ids),
        "horizons": list(horizons),
        "ckpt": str(args.ckpt),
        "config": ckpt.get("config"),
    }

    with h5py.File(STEP3_DATA_H5, "r") as h5, h5py.File(ODE3_FEATURES_H5, "r") as feat:
        bundle = load_bundle(h5, val_ids, feat_h5=feat)
        ev = evaluate(model, bundle, val_ids, device, horizons=horizons,
                      n_boot=args.n_boot)
        report["stay"] = ev
        report["table"] = _rows(ev)
        print(format_table(ev, "JLucid 4"))

        js = jacobian_smoke(model, bundle, val_ids, device)
        spec = spectral_report(model)
        report["jacobian_smoke"] = js
        report["spectral"] = spec
        print(f"  J smoke ok={js['ok']} match={js['matches_closed_form']} "
              f"max|J|={js['max_abs']:.3f} cond(I+A)={spec['max_cond_IplusA']:.2f} "
              f"sane={spec['sane_linearization']}")

        peek = jspace_peek(model, bundle, val_ids, device)
        ODE4_JSPACE_JSON.write_text(json.dumps(peek, indent=2))
        report["jspace"] = str(ODE4_JSPACE_JSON)

        dest = evaluate_discrete(
            model, bundle, val_ids, device, window=WINDOW_TR,
            centroids=getattr(model, "centroids", None))
        report["discrete"] = dest
        if dest.get("ok"):
            sw = dest["switch"]
            print(f"  discrete: switch F1={sw['switch_f1']:.3f}  "
                  f"dest-on-true={dest['dest_on_true_switches']:.3f}  "
                  f"acc={sw['state_acc_all']:.3f}  τ={dest['tau']:.3f}")
        else:
            print(f"  discrete skipped: {dest.get('reason')}")

    np.save(ODE4_A_NPY, model.stay.A.detach().cpu().numpy())
    args.out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
