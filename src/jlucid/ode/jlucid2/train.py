"""Training loops for the latent Neural ODE.

``train`` is the original per-subject full-scan loop (kept to load/score
the first run). ``train_shots`` is the short-horizon batched protocol.
"""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import torch

from jlucid.config import (
    EARLY_STOP_PATIENCE,
    HORIZON_TR,
    LR,
    MAX_EPOCHS,
    SEED,
    SHOT_BATCH,
    VAL_FRAC,
    WEIGHT_DECAY,
    WINDOW_TR,
)
from jlucid.ode.jlucid2.models import LatentODEModel, model_loss, shot_loss
from jlucid.ode.jlucid2.shots import collate_shots, load_bundle, shot_index


def seed_all(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subject_tensors(h5, sid: str, stats: dict, device: torch.device):
    v1 = h5[f"v1/{sid}"][:].astype(np.float32)
    labels = h5[f"labels/{sid}"][:].astype(np.int64)
    mask = h5[f"mask/{sid}"][:].astype(bool)
    p = h5[f"p/{sid}"][:].astype(np.float32)
    wmask = h5[f"wmask/{sid}"][:].astype(bool)
    v1 = (v1 - stats["v1_mean"]) / np.maximum(stats["v1_std"], 1e-8)
    return (
        torch.from_numpy(v1).to(device),
        torch.from_numpy(labels).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(p).to(device),
        torch.from_numpy(wmask).to(device),
    )


def subject_loss(model, h5, sid, stats, device):
    v1, labels, mask, p, wmask = load_subject_tensors(h5, sid, stats, device)
    out = model(v1, p)
    return model_loss(model, out, v1, labels, mask, wmask)


def train_step(model, optimizer, h5, train_sids, stats, device,
               accum: int = 8) -> dict:
    model.train()
    optimizer.zero_grad()
    agg = {"total": 0.0, "recon": 0.0, "kl": 0.0, "state_ce": 0.0,
           "smooth": 0.0, "consist": 0.0}
    n = 0
    for i, sid in enumerate(train_sids):
        ls = subject_loss(model, h5, sid, stats, device)
        (ls["total"] / accum).backward()
        for k in agg:
            agg[k] += float(ls[k].detach())
        n += 1
        if (i + 1) % accum == 0 or (i + 1) == len(train_sids):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()
    return {k: v / max(n, 1) for k, v in agg.items()}


@torch.no_grad()
def evaluate(model, h5, sids, stats, device) -> dict:
    model.eval()
    agg = {"total": 0.0, "recon": 0.0, "kl": 0.0, "state_ce": 0.0,
           "smooth": 0.0, "consist": 0.0}
    n = 0
    for sid in sids:
        ls = subject_loss(model, h5, sid, stats, device)
        for k in agg:
            agg[k] += float(ls[k].item())
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def train(model, h5, meta, stats, device, epochs: int = MAX_EPOCHS,
          lr: float = LR, wd: float = WEIGHT_DECAY,
          accum: int = 8, patience: int = EARLY_STOP_PATIENCE,
          val_frac: float = VAL_FRAC, seed: int = SEED,
          log=print, ckpt_every: int = 25, ckpt_path: Path | None = None) -> dict:
    seed_all(seed)
    meta = meta.copy()
    n_val = max(1, int(round(len(meta) * val_frac)))
    # stratified val: keep dx balance by sampling evenly per group
    val_ids = []
    for g in meta["dx_group"].unique():
        sub = meta[meta["dx_group"] == g]
        n_g = max(1, int(round(len(sub) * val_frac)))
        val_ids.extend(np.random.choice(sub["raw_id"].to_numpy(), size=n_g,
                                        replace=False).tolist())
    val_ids = sorted(set(val_ids))
    train_ids = [s for s in meta["raw_id"].tolist() if s not in set(val_ids)]
    log(f"train {len(train_ids)} / val {len(val_ids)} subjects")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val = float("inf")
    best_state = None
    history = []
    patience_left = patience
    last_saved = -1
    for epoch in range(epochs):
        t0 = time.time()
        tr = train_step(model, optimizer, h5, train_ids, stats, device, accum=accum)
        va = evaluate(model, h5, val_ids, stats, device)
        history.append({"epoch": epoch, "train": tr, "val": va})
        log(f"epoch {epoch:03d} train_recon={tr['recon']:.4f} "
            f"val_recon={va['recon']:.4f} state_ce={va['state_ce']:.3f} "
            f"({time.time() - t0:.1f}s)")
        if va["recon"] < best_val - 1e-6:
            best_val = va["recon"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(f"early stop at epoch {epoch}")
                break
        if ckpt_path is not None and epoch - last_saved >= ckpt_every:
            last_saved = epoch
            tmp = {"state_dict": model.state_dict(),
                   "config": {k: getattr(model, k) for k in
                              ("latent", "k", "window", "solver", "rtol", "atol")},
                   "result": {"best_val_recon": float(best_val),
                              "n_train": len(train_ids), "n_val": len(val_ids),
                              "epoch": epoch, "val_ids": val_ids},
                   "meta": {"raw_frames": 0, "usable_frames": 0, "n_subjects": len(meta)}}
            torch.save(tmp, ckpt_path)
            log(f"periodic checkpoint saved at epoch {epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val_recon": float(best_val),
        "n_train": len(train_ids), "n_val": len(val_ids),
        "val_ids": val_ids, "history": history,
    }


def _model_config(model: LatentODEModel) -> dict:
    keys = ("latent", "k", "window", "solver", "rtol", "atol",
            "predict_residual", "condition", "unit_sphere")
    cfg = {}
    for k in keys:
        if hasattr(model, k):
            cfg[k] = getattr(model, k)
    return cfg


def save_checkpoint(model, result: dict, path: Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": _model_config(model),
        "result": {k: v for k, v in result.items() if k != "history"},
        "meta": meta,
    }, path)


def split_ids(meta, val_frac: float = VAL_FRAC, seed: int = SEED,
              val_ids: list | None = None) -> tuple[list, list]:
    """Reuse a provided val split (e.g. the original 30) when given."""
    rng = np.random.default_rng(seed)
    if val_ids is None:
        val_ids = []
        for g in meta["dx_group"].unique():
            sub = meta[meta["dx_group"] == g]
            n_g = max(1, int(round(len(sub) * val_frac)))
            val_ids.extend(rng.choice(sub["raw_id"].to_numpy(), size=n_g,
                                      replace=False).tolist())
        val_ids = sorted(set(str(s) for s in val_ids))
    else:
        val_ids = [str(s) for s in val_ids]
    train_ids = [str(s) for s in meta["raw_id"].tolist() if str(s) not in set(val_ids)]
    return train_ids, val_ids


def _epoch_shots(model, bundle, index, optimizer, device, *,
                 window: int, horizon: int, batch_size: int,
                 train: bool) -> dict:
    if train:
        model.train()
        rng = np.random.default_rng()
        order = rng.permutation(len(index))
    else:
        model.eval()
        order = np.arange(len(index))
    agg = {"total": 0.0, "pred": 0.0, "now": 0.0, "kl": 0.0,
           "state_ce": 0.0, "smooth": 0.0, "cos1": 0.0}
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for start in range(0, len(order), batch_size):
            batch_idx = [index[int(i)] for i in order[start:start + batch_size]]
            batch = collate_shots(bundle, batch_idx, window, horizon, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model.forward_shots(
                batch["windows"], batch["p0"], n_steps=horizon,
                sample=train, v1_now=batch["v1_now"],
            )
            ls = shot_loss(out, batch["v1_future"], batch["labels_future"],
                           batch["v1_now"])
            if train:
                ls["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = batch["windows"].shape[0]
            for k in agg:
                agg[k] += float(ls[k].detach()) * bs
            n += bs
    return {k: v / max(n, 1) for k, v in agg.items()}


def train_shots(model, h5, meta, stats, device, *,
                epochs: int = MAX_EPOCHS, lr: float = LR, wd: float = WEIGHT_DECAY,
                batch_size: int = SHOT_BATCH, horizon: int = HORIZON_TR,
                window: int | None = None, patience: int = EARLY_STOP_PATIENCE,
                val_frac: float = VAL_FRAC, seed: int = SEED,
                val_ids: list | None = None, native: bool = True,
                log=print, ckpt_every: int = 10,
                ckpt_path: Path | None = None) -> dict:
    """Batched short-shot training. Early-stops on val (1 - cos Δ=1)."""
    seed_all(seed)
    window = int(window or model.window)
    train_ids, val_ids = split_ids(meta, val_frac=val_frac, seed=seed,
                                   val_ids=val_ids)
    bundle = load_bundle(h5, train_ids + val_ids, native=native, stats=stats)
    tr_idx = shot_index({s: bundle[s] for s in train_ids}, window, horizon)
    va_idx = shot_index({s: bundle[s] for s in val_ids}, window, horizon)
    log(f"shots: train {len(train_ids)} subj / {len(tr_idx)} shots; "
        f"val {len(val_ids)} subj / {len(va_idx)} shots; "
        f"batch={batch_size} horizon={horizon} window={window}")
    if not tr_idx or not va_idx:
        raise RuntimeError("no valid shots (window/horizon vs masks)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val = float("inf")
    best_state = None
    history = []
    patience_left = patience
    last_saved = -1
    for epoch in range(epochs):
        t0 = time.time()
        tr = _epoch_shots(model, bundle, tr_idx, optimizer, device,
                          window=window, horizon=horizon, batch_size=batch_size,
                          train=True)
        va = _epoch_shots(model, bundle, va_idx, optimizer, device,
                          window=window, horizon=horizon, batch_size=batch_size,
                          train=False)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train": tr, "val": va, "sec": dt})
        log(f"epoch {epoch:03d} train_pred={tr['pred']:.4f} val_pred={va['pred']:.4f} "
            f"val_cos1={va['cos1']:.4f} state_ce={va['state_ce']:.3f} ({dt:.2f}s)")
        if va["pred"] < best_val - 1e-6:
            best_val = va["pred"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(f"early stop at epoch {epoch}")
                break
        if ckpt_path is not None and epoch - last_saved >= ckpt_every:
            last_saved = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "config": _model_config(model),
                "result": {"best_val_pred": float(best_val), "epoch": epoch,
                           "n_train": len(train_ids), "n_val": len(val_ids),
                           "val_ids": val_ids},
                "meta": {"horizon": horizon, "window": window},
            }, ckpt_path)
            log(f"periodic checkpoint saved at epoch {epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val_pred": float(best_val),
        "n_train": len(train_ids), "n_val": len(val_ids),
        "n_train_shots": len(tr_idx), "n_val_shots": len(va_idx),
        "val_ids": val_ids, "history": history,
        "horizon": horizon, "window": window, "batch_size": batch_size,
    }
