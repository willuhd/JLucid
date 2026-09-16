"""Honest local-rollout evaluation for short-shot (and legacy) Neural ODEs.

From information available at time t (encoder window + optional p(t)),
predict V1 and state at t+Δ. Never score a full-scan reconstruction from
t=0 as a forecast. Baselines: persistence, current-state centroid,
encoder-only (no ODE), persist-label, majority class.
"""

from __future__ import annotations

import numpy as np
import torch

from jlucid.config import EVAL_DELTAS, HORIZON_TR, SHOT_BATCH, WINDOW_TR
from jlucid.ode.jlucid2.shots import load_bundle, shot_index


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    return (a * b).sum(axis=-1) / np.maximum(an * bn, 1e-12)


def bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float("nan"),) * 3
    means = np.array([rng.choice(v, size=len(v), replace=True).mean()
                      for _ in range(n)])
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def state_centroids(bundle: dict, sids: list[str], k: int = 3) -> np.ndarray:
    """Unit mean V1 per LEiDA state over the given subjects (usable frames)."""
    acc = [[] for _ in range(k)]
    for sid in sids:
        rec = bundle[sid]
        m = rec["mask"] & (rec["labels"] >= 0)
        for s in range(k):
            sel = m & (rec["labels"] == s)
            if sel.any():
                acc[s].append(rec["v1"][sel])
    cents = []
    for s in range(k):
        if not acc[s]:
            cents.append(np.zeros(90, dtype=np.float32))
            continue
        c = np.concatenate(acc[s], axis=0).mean(axis=0)
        n = np.linalg.norm(c)
        cents.append((c / max(n, 1e-12)).astype(np.float32))
    return np.stack(cents)


@torch.no_grad()
def _rollout_subject(model, rec, *, window: int, horizon: int,
                     hold_p: bool, native_pred: bool,
                     stats: dict | None, device, batch_size: int) -> dict | None:
    """Local rollouts for one subject. Returns native-space arrays or None."""
    v1_nat = rec["v1_native"]
    v1_in = rec["v1"]
    labels = rec["labels"]
    mask = rec["mask"]
    p = rec["p"]
    t_len = len(mask)
    starts = []
    for t in range(window - 1, t_len - horizon):
        if bool(mask[t - window + 1 : t + horizon + 1].all()):
            starts.append(t)
    if not starts:
        return None
    pred = np.zeros((len(starts), horizon + 1, v1_nat.shape[1]), dtype=np.float32)
    logits = np.zeros((len(starts), horizon + 1, model.k), dtype=np.float32)
    mean = std = None
    if not native_pred:
        mean = np.asarray(stats["v1_mean"], dtype=np.float32)
        std = np.maximum(np.asarray(stats["v1_std"], dtype=np.float32), 1e-8)
    for s0 in range(0, len(starts), batch_size):
        sl = starts[s0:s0 + batch_size]
        windows = np.stack([v1_in[t - window + 1 : t + 1] for t in sl])
        p0 = np.stack([p[t] for t in sl])
        v1_now_in = np.stack([v1_in[t] for t in sl])
        tw = torch.from_numpy(windows).to(device)
        tp = torch.from_numpy(p0).to(device)
        tn = torch.from_numpy(v1_now_in).to(device)
        if hold_p or model.condition == "none":
            out = model.forward_shots(tw, tp, n_steps=horizon, sample=False,
                                      v1_now=tn)
            vhat = out["v1_hat"].detach().cpu().numpy()          # (H+1, B, 90)
            lg = out["state_logits"].detach().cpu().numpy()
        else:
            # leaky: feed true p[t], p[t+1], ... (how the original run trained)
            mu, _ = model.encoder(tw)
            p_steps = np.stack([np.stack([p[t + i] for t in sl])
                                for i in range(horizon)])        # (H, B, k)
            z = model.integrate_steps(
                mu, torch.from_numpy(p_steps).to(device), horizon, hold_p=False)
            vhat_t, lg_t, _ = model.decode_path(z, tn)
            vhat = vhat_t.detach().cpu().numpy()
            lg = lg_t.detach().cpu().numpy()
        vhat = np.transpose(vhat, (1, 0, 2))
        lg = np.transpose(lg, (1, 0, 2))
        if not native_pred:
            vhat = vhat * std + mean
        pred[s0:s0 + len(sl)] = vhat
        logits[s0:s0 + len(sl)] = lg
    return {
        "t": np.asarray(starts, dtype=int),
        "pred": pred, "logits": logits,
        "v1": v1_nat, "labels": labels, "mask": mask,
    }


def _subject_metrics(roll: dict, cents: np.ndarray, deltas: list[int],
                     k: int) -> dict:
    t0 = roll["t"]
    pred = roll["pred"]
    logits = roll["logits"]
    v1 = roll["v1"]
    lab = roll["labels"]
    out = {}
    for d in deltas:
        if d >= pred.shape[1]:
            continue
        tt = t0 + d
        y = v1[tt]
        yhat = pred[:, d]
        persist = v1[t0]
        cent = cents[np.clip(lab[t0], 0, k - 1)]
        enc0 = pred[:, 0]
        out[d] = {
            "cos_model": float(cosine_rows(y, yhat).mean()),
            "r2_model": r2(y, yhat),
            "cos_persist": float(cosine_rows(y, persist).mean()),
            "r2_persist": r2(y, persist),
            "cos_centroid": float(cosine_rows(y, cent).mean()),
            "r2_centroid": r2(y, cent),
            "cos_enc": float(cosine_rows(y, enc0).mean()),
            "r2_enc": r2(y, enc0),
            "acc_model": float((logits[:, d].argmax(axis=1) == lab[tt]).mean()),
            "acc_persist": float((lab[t0] == lab[tt]).mean()),
            "n": int(len(tt)),
        }
        # majority on this subject's targets (for completeness)
        vals, cnts = np.unique(lab[tt], return_counts=True)
        out[d]["acc_majority"] = float(cnts.max() / len(tt))
    return out


def evaluate_forecast(model, h5, sids, stats, device, *,
                      train_sids: list[str] | None = None,
                      window: int | None = None, horizon: int = 5,
                      deltas: list[int] | None = None,
                      hold_p: bool = True, native: bool = True,
                      batch_size: int = SHOT_BATCH) -> dict:
    """Per-subject then pooled forecast metrics + bootstrap CIs."""
    window = int(window or model.window)
    deltas = list(deltas or EVAL_DELTAS)
    sids = [str(s) for s in sids]
    train_sids = [str(s) for s in (train_sids or sids)]
    all_ids = sorted(set(sids + train_sids))
    bundle_in = load_bundle(h5, all_ids, native=native, stats=stats)
    bundle_nat = load_bundle(h5, all_ids, native=True, stats=None)
    for sid in sids:
        bundle_in[sid]["v1_native"] = bundle_nat[sid]["v1"]
    cents = state_centroids(bundle_nat, train_sids, k=model.k)
    per = []
    for sid in sids:
        roll = _rollout_subject(
            model, bundle_in[sid], window=window, horizon=horizon,
            hold_p=hold_p, native_pred=native, stats=stats,
            device=device, batch_size=batch_size,
        )
        if roll is None:
            continue
        mets = _subject_metrics(roll, cents, deltas, model.k)
        mets["sid"] = sid
        per.append(mets)

    summary = {"n_subjects": len(per), "hold_p": hold_p, "native": native,
               "window": window, "horizon": horizon, "per_delta": {}}
    keys = ["cos_model", "r2_model", "cos_persist", "r2_persist",
            "cos_centroid", "r2_centroid", "cos_enc", "r2_enc",
            "acc_model", "acc_persist", "acc_majority"]
    for d in deltas:
        entry = {}
        for key in keys:
            vals = np.array([row[d][key] for row in per if d in row], dtype=float)
            mean, lo, hi = bootstrap_ci(vals)
            entry[key] = {"mean": mean, "ci95": [lo, hi], "n": int(len(vals))}
        summary["per_delta"][str(d)] = entry
    return summary


def load_model_from_ckpt(ckpt: dict, device) -> torch.nn.Module:
    from jlucid.ode.jlucid2.models import LatentODEModel
    cfg = dict(ckpt.get("config") or {})
    # original checkpoint has no protocol flags
    cfg.setdefault("predict_residual", False)
    cfg.setdefault("condition", "leaky")
    cfg.setdefault("unit_sphere", False)
    # drop keys the ctor does not take
    allowed = {"latent", "k", "window", "in_dim", "hidden", "layers",
               "solver", "rtol", "atol", "predict_residual", "condition",
               "unit_sphere"}
    cfg = {k: v for k, v in cfg.items() if k in allowed}
    model = LatentODEModel(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
