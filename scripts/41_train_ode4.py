#!/usr/bin/env python3
"""Train JLucid 4 stay field. Dest/when are packed frozen, never trained."""
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
    LATENT_DIM,
    MAX_EPOCHS,
    ODE3_CHECKPOINT,
    ODE3_FEATURES_H5,
    ODE36_CHECKPOINT,
    ODE4_A_NPY,
    ODE4_CHECKPOINT,
    ODE4_DIR,
    SEED,
    SHOT_BATCH,
    STEP3_CHECKPOINT,
    STEP3_DATA_H5,
    WINDOW_TR,
)
from jlucid.ode.data import load_step3_dataset
from jlucid.ode.jlucid4.models import ThesisModel
from jlucid.ode.jlucid4.pack import load_stay_into, pack_discrete
from jlucid.ode.jlucid4.train import seed_all, train_stay
from jlucid.utils.logging import get_logger


def _reuse_val_ids() -> list | None:
    if STEP3_CHECKPOINT.exists():
        ckpt = torch.load(STEP3_CHECKPOINT, map_location="cpu", weights_only=False)
        ids = list(ckpt.get("result", {}).get("val_ids") or [])
        if ids:
            return [str(s) for s in ids]
    return None


def write_history_artifacts(history: list, out_dir: Path) -> dict:
    """Keep the full epoch table and curves. Never discard the run record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    if not history:
        return {"history": str(hist_path), "figures": []}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(r["epoch"]) for r in history]
    tr_stay = [r["train"]["stay"] for r in history]
    va_stay = [r["val"]["stay"] for r in history]
    tr_safe = [r["train"]["safe"] for r in history]
    va_safe = [r["val"]["safe"] for r in history]
    n_steps = max(
        int(k.split("_")[1])
        for r in history for k in r["val"] if k.startswith("gain_")
    )

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.5), sharex=True)
    axes[0].plot(epochs, tr_stay, color="0.45", label="train stay")
    axes[0].plot(epochs, va_stay, color="C0", label="val stay")
    axes[0].set_ylabel("stay loss (1 − cos)")
    axes[0].legend(frameon=False)
    axes[0].set_title("JLucid 4 training")

    axes[1].plot(epochs, tr_safe, color="0.45", label="train stay-safe")
    axes[1].plot(epochs, va_safe, color="C1", label="val stay-safe")
    axes[1].set_ylabel("stay-safe hinge")
    axes[1].legend(frameon=False)

    for h in range(1, n_steps + 1):
        axes[2].plot(epochs, [r["val"][f"gain_{h}"] for r in history],
                     label=f"val Δ{h} vs persist")
    axes[2].axhline(0.0, color="0.6", lw=0.8)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("val gain (cos)")
    axes[2].legend(frameon=False, ncol=3)

    fig.tight_layout()
    figures = []
    for suffix in (".png", ".pdf"):
        path = out_dir / f"loss_vs_epoch{suffix}"
        fig.savefig(path, dpi=150)
        figures.append(str(path))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for h in range(1, n_steps + 1):
        ax.plot(epochs, [r["val"][f"cos_{h}"] for r in history],
                label=f"model Δ{h}")
        ax.plot(epochs, [r["val"][f"persist_{h}"] for r in history],
                ls="--", color="0.5",
                label=f"persist Δ{h}" if h == 1 else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val cosine (shot-mean)")
    ax.set_title("JLucid 4 in-dwell cosine vs persist")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        path = out_dir / f"cosine_vs_epoch{suffix}"
        fig.savefig(path, dpi=150)
        figures.append(str(path))
    plt.close(fig)
    return {"history": str(hist_path), "figures": figures}


def _save(model: ThesisModel, path: Path, *, config: dict, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": config,
        "result": {k: v for k, v in result.items() if k != "history"},
        "history": result.get("history"),
        "A": model.stay.A.detach().cpu(),
        "b": model.stay.b.detach().cpu(),
        "switch_tau": float(model.switch_tau),
        "discrete_packed": bool(model._discrete_packed),
        "centroids": None if model.centroids is None else np.asarray(model.centroids),
    }, path)
    np.save(path.with_name(path.stem + "_A.npy"),
            model.stay.A.detach().cpu().numpy())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--latent", type=int, default=LATENT_DIM)
    ap.add_argument("--window", type=int, default=WINDOW_TR)
    ap.add_argument("--batch", type=int, default=SHOT_BATCH)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weights", type=str, default="1,2,2")
    ap.add_argument("--w-safe", type=float, default=1.0)
    ap.add_argument("--stay-init", type=Path, default=None,
                    help="optional stay-field checkpoint to continue from")
    ap.add_argument("--dest-ckpt", type=Path, default=ODE3_CHECKPOINT)
    ap.add_argument("--when-ckpt", type=Path, default=ODE36_CHECKPOINT)
    ap.add_argument("--pack-only", action="store_true",
                    help="write a packed checkpoint and exit (no train)")
    ap.add_argument("--no-j2-encoder", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=Path, default=ODE4_CHECKPOINT)
    args = ap.parse_args()

    log = get_logger("ode4_train",
                     ROOT / "data" / "processed" / "logs" / "41_train_ode4.log")
    if not ODE3_FEATURES_H5.exists():
        raise SystemExit(f"missing {ODE3_FEATURES_H5}; run scripts/21_ode3_features.py")
    seed_all(SEED)
    meta, _ = load_step3_dataset()
    device = torch.device(args.device)
    val_ids = _reuse_val_ids()
    ODE4_DIR.mkdir(parents=True, exist_ok=True)
    weights = [float(x) for x in args.weights.split(",") if x.strip()]

    model = ThesisModel(
        latent=args.latent, window=args.window,
        gamma_mode="learned", gamma_value=0.5,
        decode_mode="recursive", small_decode_init=not bool(args.stay_init),
    ).to(device)

    if args.stay_init and args.stay_init.exists():
        load_stay_into(model, args.stay_init)
        log.info("loaded stay from %s", args.stay_init)
    elif not args.no_j2_encoder and STEP3_CHECKPOINT.exists():
        j2 = torch.load(STEP3_CHECKPOINT, map_location="cpu", weights_only=False)
        model.encoder.load_j2_encoder(j2["state_dict"])
        log.info("warm-started stay encoder from JLucid 2")

    if args.dest_ckpt.exists() and args.when_ckpt.exists():
        pack_discrete(model, args.dest_ckpt, args.when_ckpt)
        log.info("packed dest/when")
    else:
        log.info("dest/when not packed (missing %s or %s)",
                 args.dest_ckpt, args.when_ckpt)

    cfg = {
        "latent": args.latent, "window": args.window, "k": 3,
        "eps": 0.0, "gamma_mode": "learned", "gamma_value": 0.5,
        "decode_mode": "recursive", "horizon": args.horizon,
        "weights": weights, "w_safe": args.w_safe,
        "discrete_packed": bool(model._discrete_packed),
        "seed": SEED,
        "stay_init": None if args.stay_init is None else str(args.stay_init),
        "j2_encoder": bool(not args.no_j2_encoder and STEP3_CHECKPOINT.exists()
                           and not (args.stay_init and args.stay_init.exists())),
        "dest_ckpt": str(args.dest_ckpt) if args.dest_ckpt.exists() else None,
        "when_ckpt": str(args.when_ckpt) if args.when_ckpt.exists() else None,
        "recipe": "jlucid4",
        "packed_only": bool(args.pack_only),
    }

    if args.pack_only:
        result = {"val_ids": val_ids, "n_steps": args.horizon, "packed_only": True}
        _save(model, args.out, config=cfg, result=result)
        np.save(ODE4_A_NPY, model.stay.A.detach().cpu().numpy())
        print(json.dumps({"out": str(args.out),
                          "discrete_packed": model._discrete_packed}, indent=2))
        return 0

    with h5py.File(STEP3_DATA_H5, "r") as h5, h5py.File(ODE3_FEATURES_H5, "r") as feat:
        result = train_stay(
            model, h5, feat, meta, device, epochs=args.epochs,
            lr=args.lr, batch_size=args.batch, window=args.window,
            n_steps=args.horizon, val_ids=val_ids, weights=weights,
            w_safe=args.w_safe, use_dwell=True, log=log.info,
        )
    _save(model, args.out, config=cfg, result=result)
    np.save(ODE4_A_NPY, model.stay.A.detach().cpu().numpy())
    artifacts = write_history_artifacts(result.get("history") or [], ODE4_DIR)
    summary = {
        "out": str(args.out),
        "best_score": result["best_score"],
        "n_train": result["n_train"],
        "n_val": result["n_val"],
        "n_train_subj": result.get("n_train_subj"),
        "n_val_subj": result.get("n_val_subj"),
        "n_epochs": len(result.get("history") or []),
        "discrete_packed": model._discrete_packed,
        "packed_only": False,
        "artifacts": artifacts,
        "config": cfg,
    }
    (ODE4_DIR / "train_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
