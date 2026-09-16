"""JLucid 4 training: stay field only, on whole-horizon dwells."""

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
    VAL_FRAC,
    WEIGHT_DECAY,
)
from jlucid.ode.jlucid4.models import (
    ThesisModel,
    horizon_weights,
    stay_loss_weighted,
    stay_safe_penalty,
)
from jlucid.ode.jlucid4.shots import collate_shots, dwell_index, load_bundle, stay_index


def seed_all(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_ids(meta, val_frac: float = VAL_FRAC, seed: int = SEED,
              val_ids: list | None = None) -> tuple[list, list]:
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


def _epoch(model, bundle, idx, optimizer, device, *,
           window: int, batch_size: int, n_steps: int, train: bool,
           weights: torch.Tensor, w_safe: float) -> dict:
    if train:
        model.train()
        order = np.random.default_rng().permutation(len(idx))
    else:
        model.eval()
        order = np.arange(len(idx))
    agg = {"stay": 0.0, "safe": 0.0}
    cos_h = np.zeros(n_steps, dtype=np.float64)
    persist_h = np.zeros(n_steps, dtype=np.float64)
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for start in range(0, len(order), batch_size):
            bi = [idx[int(i)] for i in order[start:start + batch_size]]
            b = collate_shots(bundle, bi, window, n_steps, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model(b["windows"], b["p0"], b["v1_now"],
                        b["v2_now"], b["gap_now"], n_steps=n_steps)
            pred = out["v_stay"]
            y = b["v1_future"]
            ls = stay_loss_weighted(pred, y, weights)
            safe = stay_safe_penalty(pred, y, b["v1_now"])
            total = ls + w_safe * safe
            if train:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.stay_params(), 5.0)
                optimizer.step()
            bs = pred.shape[0]
            agg["stay"] += float(ls.detach()) * bs
            agg["safe"] += float(safe.detach()) * bs
            with torch.no_grad():
                cos = (pred * y).sum(dim=-1).mean(dim=0).cpu().numpy()
                pcos = (b["v1_now"].unsqueeze(1) * y).sum(dim=-1).mean(dim=0).cpu().numpy()
            cos_h += cos * bs
            persist_h += pcos * bs
            n += bs
    out = {"stay": agg["stay"] / max(n, 1), "safe": agg["safe"] / max(n, 1), "n": n}
    for h in range(n_steps):
        out[f"cos_{h + 1}"] = float(cos_h[h] / max(n, 1))
        out[f"persist_{h + 1}"] = float(persist_h[h] / max(n, 1))
        out[f"gain_{h + 1}"] = out[f"cos_{h + 1}"] - out[f"persist_{h + 1}"]
    return out


def train_stay(model: ThesisModel, h5, feat_h5, meta, device, *,
               epochs: int = MAX_EPOCHS, lr: float = LR, wd: float = WEIGHT_DECAY,
               batch_size: int = SHOT_BATCH, window: int | None = None,
               n_steps: int = 3, patience: int = EARLY_STOP_PATIENCE,
               seed: int = SEED, val_ids: list | None = None,
               train_ids: list | None = None,
               weights: list[float] | None = None,
               w_safe: float = 1.0, use_dwell: bool = True,
               log=print) -> dict:
    seed_all(seed)
    window = int(window or model.window)
    if train_ids is not None and val_ids is not None:
        train_ids = [str(s) for s in train_ids]
        val_ids = [str(s) for s in val_ids]
    else:
        train_ids, val_ids = split_ids(meta, seed=seed, val_ids=val_ids)
    bundle = load_bundle(h5, train_ids + val_ids, feat_h5=feat_h5)
    indexer = dwell_index if use_dwell else stay_index
    tr_idx = indexer({s: bundle[s] for s in train_ids}, window, n_steps)
    va_idx = indexer({s: bundle[s] for s in val_ids}, window, n_steps)
    w = horizon_weights(n_steps, weights)
    log(f"JLucid 4: train {len(tr_idx)} / val {len(va_idx)}; "
        f"H={n_steps} decode={model.decode_mode} gamma={model.gamma_mode} "
        f"dwell={use_dwell} weights={w.tolist()} w_safe={w_safe}")
    if not tr_idx or not va_idx:
        raise RuntimeError("no stay/dwell shots at this horizon")
    model.freeze_discrete()
    opt = torch.optim.Adam(model.stay_params(), lr=lr, weight_decay=wd)
    best = float("inf")
    best_state = None
    left = patience
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        tr = _epoch(model, bundle, tr_idx, opt, device, window=window,
                    batch_size=batch_size, n_steps=n_steps, train=True,
                    weights=w, w_safe=w_safe)
        va = _epoch(model, bundle, va_idx, opt, device, window=window,
                    batch_size=batch_size, n_steps=n_steps, train=False,
                    weights=w, w_safe=w_safe)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train": tr, "val": va, "sec": dt})
        gains = " ".join(
            f"Δ{h}={va[f'cos_{h}']:.4f}({va[f'gain_{h}']:+.4f})"
            for h in range(1, n_steps + 1))
        log(f"epoch {epoch:03d} stay={va['stay']:.4f} {gains} ({dt:.2f}s)")
        d1_ok = va.get("gain_1", 0.0) >= -0.002
        score = va["stay"] if d1_ok else va["stay"] + 1.0
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
        "n_train": len(tr_idx), "n_val": len(va_idx),
        "n_train_subj": len(train_ids), "n_val_subj": len(val_ids),
        "train_ids": train_ids, "val_ids": val_ids,
        "history": history, "n_steps": n_steps,
        "use_dwell": bool(use_dwell),
        "decode_mode": model.decode_mode,
        "gamma_mode": model.gamma_mode,
        "weights": w.tolist(),
        "eps": 0.0,
    }
