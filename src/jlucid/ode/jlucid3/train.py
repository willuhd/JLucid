"""JLucid 3 training: stay-field on dwell shots, router on all shots."""

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
from jlucid.ode.jlucid3.models import ThesisModel, router_loss, stay_loss
from jlucid.ode.jlucid3.shots import collate_shots, load_bundle, shot_index, stay_index


def _epoch(model, bundle, stay_idx, all_idx, optimizer, device, *,
           window: int, batch_size: int, n_steps: int, train: bool,
           w_route: float) -> dict:
    if train:
        model.train()
        rng = np.random.default_rng()
        stay_ord = rng.permutation(len(stay_idx))
        all_ord = rng.permutation(len(all_idx))
    else:
        model.eval()
        stay_ord = np.arange(len(stay_idx))
        all_ord = np.arange(len(all_idx))
    agg = {"stay": 0.0, "bce": 0.0, "ce": 0.0, "cos_stay": 0.0}
    n_s = n_r = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for start in range(0, len(stay_ord), batch_size):
            bi = [stay_idx[int(i)] for i in stay_ord[start:start + batch_size]]
            b = collate_shots(bundle, bi, window, n_steps, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model(b["windows"], b["p0"], b["v1_now"],
                        b["v2_now"], b["gap_now"], n_steps=n_steps)
            # average 1-cos over the stay horizon
            pred = out["v_stay"]  # (B, H, 90)
            y = b["v1_future"]
            ls = stay_loss(pred.reshape(-1, 90), y.reshape(-1, 90))
            if train:
                ls.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = pred.shape[0]
            agg["stay"] += float(ls.detach()) * bs
            agg["cos_stay"] += float((1.0 - ls.detach())) * bs
            n_s += bs
        for start in range(0, len(all_ord), batch_size):
            bi = [all_idx[int(i)] for i in all_ord[start:start + batch_size]]
            b = collate_shots(bundle, bi, window, 1, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model(b["windows"], b["p0"], b["v1_now"],
                        b["v2_now"], b["gap_now"], n_steps=1)
            rl = router_loss(out, b["v1_now"], b["v1_future"][:, 0],
                             b["labels_future"][:, 0])
            total = w_route * (rl["bce"] + rl["ce"])
            if train:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = b["windows"].shape[0]
            agg["bce"] += float(rl["bce"].detach()) * bs
            agg["ce"] += float(rl["ce"].detach()) * bs
            n_r += bs
    return {
        "stay": agg["stay"] / max(n_s, 1),
        "cos_stay": agg["cos_stay"] / max(n_s, 1),
        "bce": agg["bce"] / max(n_r, 1),
        "ce": agg["ce"] / max(n_r, 1),
        "n_stay": n_s, "n_route": n_r,
    }


def train_thesis(model: ThesisModel, h5, feat_h5, meta, device, *,
                 epochs: int = MAX_EPOCHS, lr: float = LR, wd: float = WEIGHT_DECAY,
                 batch_size: int = SHOT_BATCH, window: int | None = None,
                 n_steps: int = 1, patience: int = EARLY_STOP_PATIENCE,
                 seed: int = SEED, val_ids: list | None = None,
                 w_route: float = 1.0, log=print) -> dict:
    seed_all(seed)
    window = int(window or model.window)
    train_ids, val_ids = split_ids(meta, seed=seed, val_ids=val_ids)
    bundle = load_bundle(h5, train_ids + val_ids, feat_h5=feat_h5)
    tr_stay = stay_index({s: bundle[s] for s in train_ids}, window, n_steps)
    va_stay = stay_index({s: bundle[s] for s in val_ids}, window, n_steps)
    tr_all = shot_index({s: bundle[s] for s in train_ids}, window, 1)
    va_all = shot_index({s: bundle[s] for s in val_ids}, window, 1)
    log(f"JLucid 3: train stay {len(tr_stay)} / all {len(tr_all)}; "
        f"val stay {len(va_stay)} / all {len(va_all)}; H={n_steps}")
    if not tr_stay or not va_stay:
        raise RuntimeError("no stay shots")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = float("inf")
    best_state = None
    left = patience
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        tr = _epoch(model, bundle, tr_stay, tr_all, opt, device, window=window,
                    batch_size=batch_size, n_steps=n_steps, train=True,
                    w_route=w_route)
        va = _epoch(model, bundle, va_stay, va_all, opt, device, window=window,
                    batch_size=batch_size, n_steps=n_steps, train=False,
                    w_route=w_route)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train": tr, "val": va, "sec": dt})
        log(f"epoch {epoch:03d} stay={va['stay']:.4f} cos_stay={va['cos_stay']:.4f} "
            f"bce={va['bce']:.3f} ce={va['ce']:.3f} ({dt:.2f}s)")
        score = va["stay"] + 0.25 * va["bce"]
        if score < best - 1e-6:
            best = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            left = patience
        else:
            left -= 1
            if left <= 0:
                log(f"early stop at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_score": float(best),
        "n_train_stay": len(tr_stay), "n_val_stay": len(va_stay),
        "n_train_all": len(tr_all), "n_val_all": len(va_all),
        "val_ids": val_ids, "history": history, "n_steps": n_steps,
    }
