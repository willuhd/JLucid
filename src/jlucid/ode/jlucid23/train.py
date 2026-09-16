"""Train the 2.3 stay field + flip head on 1-step shots."""

from __future__ import annotations

import time

import numpy as np
import torch

from jlucid.config import (
    EARLY_STOP_PATIENCE,
    LR,
    MAX_EPOCHS,
    SEED,
    SHOT_BATCH,
    WEIGHT_DECAY,
    WINDOW_TR,
)
from jlucid.ode.jlucid2.train import seed_all, split_ids
from jlucid.ode.jlucid23.models import StaySplitModel, split_loss
from jlucid.ode.jlucid23.shots import collate_shots, load_bundle, shot_index


def _epoch(model, bundle, index, optimizer, device, *,
           window: int, batch_size: int, train: bool,
           w_bce: float, w_pull: float) -> dict:
    if train:
        model.train()
        order = np.random.default_rng().permutation(len(index))
    else:
        model.eval()
        order = np.arange(len(index))
    keys = ("total", "stay_fit", "pull", "bce", "vel_fit", "cos1",
            "g_flip", "g_stay")
    agg = {k: 0.0 for k in keys}
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for start in range(0, len(order), batch_size):
            batch_idx = [index[int(i)] for i in order[start:start + batch_size]]
            b = collate_shots(bundle, batch_idx, window, 1, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model(b["windows"], b["v1_now"], b["p0"])
            ls = split_loss(out, b["v1_future"][:, 0], b["v1_now"],
                            b["labels_now"], b["labels_future"][:, 0],
                            w_bce=w_bce, w_pull=w_pull)
            if train:
                ls["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = b["windows"].shape[0]
            for k in keys:
                agg[k] += float(ls[k].detach()) * bs
            n += bs
    return {k: v / max(n, 1) for k, v in agg.items()}


def train_split(model: StaySplitModel, h5, meta, device, *,
                epochs: int = MAX_EPOCHS, lr: float = LR, wd: float = WEIGHT_DECAY,
                batch_size: int = SHOT_BATCH, window: int | None = None,
                patience: int = EARLY_STOP_PATIENCE, seed: int = SEED,
                val_ids: list | None = None, w_bce: float = 1.0,
                w_pull: float = 0.35, freeze_encoder: bool = True,
                log=print) -> dict:
    seed_all(seed)
    window = int(window or model.window)
    train_ids, val_ids = split_ids(meta, seed=seed, val_ids=val_ids)
    bundle = load_bundle(h5, train_ids + val_ids, native=True, stats=None)
    tr_idx = shot_index({s: bundle[s] for s in train_ids}, window, 1)
    va_idx = shot_index({s: bundle[s] for s in val_ids}, window, 1)
    log(f"2.3 shots: train {len(train_ids)}/{len(tr_idx)}  val {len(val_ids)}/{len(va_idx)}")
    if not tr_idx or not va_idx:
        raise RuntimeError("no valid 1-step shots")

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    best = float("inf")
    best_state = None
    history = []
    left = patience
    for epoch in range(epochs):
        t0 = time.time()
        tr = _epoch(model, bundle, tr_idx, opt, device, window=window,
                    batch_size=batch_size, train=True, w_bce=w_bce, w_pull=w_pull)
        va = _epoch(model, bundle, va_idx, opt, device, window=window,
                    batch_size=batch_size, train=False, w_bce=w_bce, w_pull=w_pull)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train": tr, "val": va, "sec": dt})
        # early-stop on stay residual fit (the thesis field), not mixed cosine
        score = va["stay_fit"]
        log(f"epoch {epoch:03d} stay_fit={va['stay_fit']:.4f} vel_fit={va['vel_fit']:.4f} "
            f"bce={va['bce']:.3f} g_flip={va['g_flip']:.3f} g_stay={va['g_stay']:.3f} "
            f"({dt:.1f}s)")
        if score < best - 1e-6:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            left = patience
        else:
            left -= 1
            if left <= 0:
                log(f"early stop at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val_stay_fit": float(best),
        "n_train": len(train_ids), "n_val": len(val_ids),
        "n_train_shots": len(tr_idx), "n_val_shots": len(va_idx),
        "val_ids": val_ids, "history": history,
        "freeze_encoder": freeze_encoder, "w_bce": w_bce, "w_pull": w_pull,
    }
