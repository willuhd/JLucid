"""Discrete dest / when: features, metrics, eval. No 3.x imports."""

from __future__ import annotations

import numpy as np
import torch

from jlucid.config import FLIP_COS_THRESH, LATENT_DIM, SHOT_BATCH, STEP3_K_COND
from jlucid.ode.jlucid4.shots import shot_index, stack_switch_shots

SCALAR_NAMES = [
    "last_cos", "dest_margin", "persist_p", "flip_logit", "dwell_log",
    "dest_logit_0", "dest_logit_1", "dest_logit_2",
    "origin_0", "origin_1", "origin_2",
    "centroid_margin", "v1v2", "dgap", "gap",
]


def feature_names(latent: int = LATENT_DIM) -> list[str]:
    return list(SCALAR_NAMES) + [f"z0_{i}" for i in range(int(latent))]


def cosine_rows(a, b) -> np.ndarray:
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(an * bn, 1e-12)


def mask_origin(logits, origin, fill: float = -1e9):
    if torch.is_tensor(logits):
        out = logits.clone()
        o = origin.long().clamp(0, out.shape[-1] - 1)
        out[torch.arange(out.shape[0], device=out.device), o] = fill
        return out
    out = np.array(logits, copy=True, dtype=np.float64)
    o = np.asarray(origin, dtype=int).clip(0, out.shape[-1] - 1)
    out[np.arange(out.shape[0]), o] = fill
    return out


def dest_margin(logits: np.ndarray, origin, fill: float = -1e9) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    o = np.asarray(origin, dtype=int)
    b = x.shape[0]
    origin_logit = x[np.arange(b), o]
    masked = x.copy()
    masked[np.arange(b), o] = fill
    return (masked.max(axis=-1) - origin_logit).astype(np.float64)


def persist_prob(logits: np.ndarray, origin) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    o = np.asarray(origin, dtype=int)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    sm = e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)
    return sm[np.arange(sm.shape[0]), o]


def compose_route(origin, dest_masked, switch_pred) -> np.ndarray:
    origin = np.asarray(origin, dtype=int)
    dest_masked = np.asarray(dest_masked, dtype=int)
    call = np.asarray(switch_pred, dtype=bool)
    return np.where(call, dest_masked, origin)


def average_precision(y_true, scores) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float(prec[y].sum() / n_pos)


def roc_auc(y_true, scores) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def pr_at_threshold(y_true, scores, tau: float) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=bool)
    call = np.asarray(scores) >= tau
    prec = float(y[call].mean()) if call.any() else float("nan")
    rec = float(call[y].mean()) if y.any() else float("nan")
    return prec, rec


def topk_precision_recall(y_true, scores, frac: float = 0.05):
    s = np.asarray(scores, dtype=float)
    tau = float(np.quantile(s, 1.0 - frac))
    prec, rec = pr_at_threshold(y_true, s, tau)
    return prec, rec, tau


def binary_f1(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    tp = float((y & p).sum())
    fp = float((~y & p).sum())
    fn = float((y & ~p).sum())
    if tp == 0 and (fp == 0 or fn == 0):
        return 0.0
    prec = tp / max(tp + fp, 1e-12)
    rec = tp / max(tp + fn, 1e-12)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def offdiag_macro_f1(origin, dest_true, dest_pred, k: int = 3) -> float:
    o = np.asarray(origin, dtype=int)
    yt = np.asarray(dest_true, dtype=int)
    yp = np.asarray(dest_pred, dtype=int)
    sw = o != yt
    if not sw.any():
        return float("nan")
    f1s = []
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            true_c = sw & (o == i) & (yt == j)
            pred_c = sw & (o == i) & (yp == j)
            f1s.append(binary_f1(true_c, pred_c))
    return float(np.mean(f1s))


def confusion_k(y_true, y_pred, k: int = 3) -> list[list[int]]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    c = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            c[i, j] = int(((yt == i) & (yp == j)).sum())
    return c.tolist()


def precision_recall_rate(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    tp = float((y & p).sum())
    fp = float((~y & p).sum())
    fn = float((y & ~p).sum())
    prec = tp / max(tp + fp, 1e-12)
    rec = tp / max(tp + fn, 1e-12)
    f1 = 0.0 if (prec + rec) == 0 else float(2.0 * prec * rec / (prec + rec))
    return {
        "precision": float(prec), "recall": float(rec), "f1": f1,
        "pred_rate": float(p.mean()) if len(p) else float("nan"),
        "n_pred": int(p.sum()), "n_true": int(y.sum()), "n": int(len(y)),
    }


def switch_metrics(origin, dest_true, dest_for_switch, switch_pred, *,
                   flip_true=None, flip_score=None, switch_score=None,
                   k: int = 3) -> dict:
    origin = np.asarray(origin, dtype=int)
    dest_true = np.asarray(dest_true, dtype=int)
    dest_for_switch = np.asarray(dest_for_switch, dtype=int)
    call = np.asarray(switch_pred, dtype=bool)
    sw = origin != dest_true
    pred_label = np.where(call, dest_for_switch, origin)
    pr = precision_recall_rate(sw, call)
    dest_true_acc = (
        float((dest_for_switch[sw] == dest_true[sw]).mean())
        if sw.any() else float("nan")
    )
    dest_pred_acc = (
        float((dest_for_switch[call] == dest_true[call]).mean())
        if call.any() else None
    )
    out = {
        "n_all": int(len(origin)),
        "n_switch": int(sw.sum()),
        "switch_rate": float(sw.mean()) if len(sw) else float("nan"),
        "switch_precision": pr["precision"],
        "switch_recall": pr["recall"],
        "switch_f1": pr["f1"],
        "switch_pred_rate": pr["pred_rate"],
        "n_pred_switch": pr["n_pred"],
        "dest_acc_on_true_switches": dest_true_acc,
        "dest_acc_on_pred_switches": dest_pred_acc,
        "switch_f1_6cell": offdiag_macro_f1(origin, dest_true, pred_label, k=k),
        "state_acc_all": float((pred_label == dest_true).mean()),
        "state_persist_label": float((origin == dest_true).mean()),
        "switch_confusion": (
            confusion_k(dest_true[sw], pred_label[sw], k=k) if sw.any() else None
        ),
    }
    if switch_score is not None:
        out["switch_ap"] = average_precision(sw, switch_score)
        out["switch_auc"] = roc_auc(sw, switch_score)
    if flip_true is not None and flip_score is not None:
        flip_true = np.asarray(flip_true, dtype=bool)
        flip_score = np.asarray(flip_score, dtype=float)
        out["n_flip"] = int(flip_true.sum())
        out["flip_rate"] = float(flip_true.mean()) if len(flip_true) else float("nan")
        out["flip_ap"] = average_precision(flip_true, flip_score)
        out["flip_auc"] = roc_auc(flip_true, flip_score)
        p05, r05 = pr_at_threshold(flip_true, flip_score, 0.5)
        p5, r5, t5 = topk_precision_recall(flip_true, flip_score, 0.05)
        out["flip_prec_0.5"] = p05
        out["flip_rec_0.5"] = r05
        out["flip_prec_top5"] = p5
        out["flip_rec_top5"] = r5
        out["flip_tau_top5"] = t5
    return out


def standardize_apply(X: np.ndarray, mean, std) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


@torch.no_grad()
def collect_dest(model, packed: dict, device, batch: int = SHOT_BATCH) -> dict:
    model.eval()
    n = packed["v1_win"].shape[0]
    flip, dest, z0 = [], [], []
    for s0 in range(0, n, batch):
        sl = slice(s0, s0 + batch)
        windows = torch.from_numpy(np.ascontiguousarray(packed["v1_win"][sl])).to(device)
        p0 = torch.from_numpy(np.ascontiguousarray(packed["p0"][sl])).to(device)
        now = torch.from_numpy(np.ascontiguousarray(packed["v1_now"][sl])).to(device)
        v2 = torch.from_numpy(np.ascontiguousarray(packed["v2_now"][sl])).to(device)
        gap = torch.from_numpy(np.ascontiguousarray(packed["gap_now"][sl])).to(device)
        out = model.dest_route(windows, p0, now, v2, gap)
        flip.append(out["flip_logit"].detach().cpu().numpy())
        dest.append(out["state_logits"].detach().cpu().numpy())
        z0.append(out["dest_z0"].detach().cpu().numpy())
    flip_logit = np.concatenate(flip, axis=0)
    dest_logits = np.concatenate(dest, axis=0)
    z0_arr = np.concatenate(z0, axis=0)
    origin = np.asarray(packed["labels_now"], dtype=np.int64)
    dest_m = mask_origin(torch.from_numpy(dest_logits),
                         torch.from_numpy(origin)).argmax(-1).numpy()
    flip_score = 1.0 / (1.0 + np.exp(-np.clip(flip_logit, -40.0, 40.0)))
    return {
        "flip_logit": flip_logit.astype(np.float32),
        "flip_score": flip_score.astype(np.float32),
        "dest_logits": dest_logits.astype(np.float32),
        "z0": z0_arr.astype(np.float32),
        "dest_masked": dest_m.astype(np.int64),
        "dest_unmasked": dest_logits.argmax(-1).astype(np.int64),
    }


def switch_arrays(packed: dict, dest: dict, centroids: np.ndarray | None = None,
                  flip_thresh: float = FLIP_COS_THRESH) -> dict:
    origin = np.asarray(packed["labels_now"], dtype=np.int64)
    nxt = np.asarray(packed["labels_next"], dtype=np.int64)
    persist_cos = cosine_rows(packed["v1_now"], packed["v1_next"])
    flip = persist_cos < float(flip_thresh)
    last_cos = (packed["v1_win"][:, -1] * packed["v1_win"][:, -2]).sum(-1)
    v1v2 = (packed["v1_now"] * packed["v2_now"]).sum(-1)
    dgap = packed["gap_win"][:, -1] - packed["gap_win"][:, -2]
    dest_logits = np.asarray(dest["dest_logits"])
    margin = dest_margin(dest_logits, origin)
    p_stay = persist_prob(dest_logits, origin)
    b = origin.shape[0]
    if centroids is None:
        cmargin = np.zeros(b, dtype=np.float64)
    else:
        ccos = packed["v1_now"] @ np.asarray(centroids, dtype=np.float32).T
        origin_c = ccos[np.arange(b), origin]
        tmp = ccos.copy()
        tmp[np.arange(b), origin] = -1e9
        cmargin = origin_c - tmp.max(axis=-1)
    return {
        "origin": origin, "dest": nxt, "switch": origin != nxt, "flip": flip,
        "persist_cos": persist_cos.astype(np.float32),
        "last_cos": last_cos.astype(np.float32),
        "v1v2": v1v2.astype(np.float32),
        "dgap": dgap.astype(np.float32),
        "gap": np.asarray(packed["gap_now"], dtype=np.float32),
        "dwell": np.asarray(packed["dwell"], dtype=np.float32),
        "dest_margin": margin.astype(np.float32),
        "persist_p": p_stay.astype(np.float32),
        "centroid_margin": np.asarray(cmargin, dtype=np.float32),
        "dest_logits": dest_logits.astype(np.float32),
        "flip_logit": np.asarray(dest["flip_logit"], dtype=np.float32),
        "flip_score": np.asarray(dest["flip_score"], dtype=np.float32),
        "z0": np.asarray(dest["z0"], dtype=np.float32),
        "dest_masked": np.asarray(dest["dest_masked"], dtype=np.int64),
        "dest_unmasked": np.asarray(dest["dest_unmasked"], dtype=np.int64),
    }


def switch_feature_matrix(arr: dict, latent: int = LATENT_DIM,
                          k: int = STEP3_K_COND) -> np.ndarray:
    n = arr["origin"].shape[0]
    oh = np.zeros((n, k), dtype=np.float32)
    o = np.asarray(arr["origin"], dtype=int).clip(0, k - 1)
    oh[np.arange(n), o] = 1.0
    dest_logits = np.asarray(arr["dest_logits"], dtype=np.float32)
    z0 = np.asarray(arr["z0"], dtype=np.float32)
    parts = [
        arr["last_cos"].reshape(n, 1),
        arr["dest_margin"].reshape(n, 1),
        arr["persist_p"].reshape(n, 1),
        arr["flip_logit"].reshape(n, 1),
        np.log1p(np.clip(arr["dwell"], 0.0, None)).reshape(n, 1),
        dest_logits[:, :k],
        oh,
        arr["centroid_margin"].reshape(n, 1),
        arr["v1v2"].reshape(n, 1),
        arr["dgap"].reshape(n, 1),
        arr["gap"].reshape(n, 1),
        z0,
    ]
    X = np.concatenate([np.asarray(p, dtype=np.float32) for p in parts], axis=1)
    expect = len(feature_names(latent))
    if X.shape[1] != expect:
        raise ValueError(f"feature dim {X.shape[1]} != {expect}")
    return X


@torch.no_grad()
def score_when(model, X: np.ndarray, device) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    logit = model.when(t).detach().cpu().numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))


@torch.no_grad()
def evaluate_discrete(model, bundle, sids, device, *, window,
                      centroids: np.ndarray | None = None) -> dict:
    if not getattr(model, "_discrete_packed", False):
        return {"ok": False, "reason": "dest/when not packed into checkpoint"}
    sub = {s: bundle[s] for s in sids}
    idx = shot_index(sub, window, 1)
    packed = stack_switch_shots(sub, idx, window)
    dest = collect_dest(model, packed, device)
    arr = switch_arrays(packed, dest, centroids=centroids)
    X = switch_feature_matrix(arr, latent=model.latent)
    mean = model.switch_mean.detach().cpu().numpy()
    std = model.switch_std.detach().cpu().numpy()
    score = score_when(model, standardize_apply(X, mean, std), device)
    tau = float(model.switch_tau)
    call = score >= tau
    ev = switch_metrics(
        arr["origin"], arr["dest"], arr["dest_masked"], call,
        flip_true=arr["flip"], flip_score=dest["flip_score"],
        switch_score=score,
    )
    return {
        "ok": True,
        "source": "packed dest + when (frozen)",
        "tau": tau,
        "n_val": int(len(idx)),
        "switch": ev,
        "dest_on_true_switches": ev["dest_acc_on_true_switches"],
        "mean_persist_p": float(np.mean(arr["persist_p"])),
    }
