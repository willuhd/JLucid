"""1-step eval with stay / flip / small_switch split and a constrained τ sweep."""

from __future__ import annotations

import numpy as np
import torch

from jlucid.config import FLIP_COS_THRESH, ODE23_STAY_FLOOR, SHOT_BATCH
from jlucid.ode.jlucid23.models import StaySplitModel, flip_target, stay_mask
from jlucid.ode.jlucid23.shots import collate_shots, load_bundle, shot_index


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    return (a * b).sum(axis=-1) / np.maximum(an * bn, 1e-12)


def regime(v_now, v_next, lab0, lab1, thresh=FLIP_COS_THRESH) -> np.ndarray:
    switched = lab0 != lab1
    is_flip = cosine_rows(v_now, v_next) < thresh
    out = np.full(len(lab0), "stay", dtype=object)
    out[switched & ~is_flip] = "small_switch"
    out[is_flip] = "flip"
    return out


def summarize(yhat, y, persist, lab0, lab1, sids) -> dict:
    cm = cosine_rows(yhat, y)
    cp = cosine_rows(persist, y)
    reg = regime(persist, y, lab0, lab1)
    rows = {}
    for name, mask in (
        ("all", np.ones(len(cm), bool)),
        ("stay", reg == "stay"),
        ("flip", reg == "flip"),
        ("small_switch", reg == "small_switch"),
    ):
        if not mask.any():
            rows[name] = {"n": 0, "cos": None, "persist": None, "gain": None}
            continue
        rows[name] = {
            "n": int(mask.sum()),
            "cos": float(cm[mask].mean()),
            "persist": float(cp[mask].mean()),
            "gain": float(cm[mask].mean() - cp[mask].mean()),
        }
    subj = []
    for s in np.unique(sids):
        m = sids == s
        subj.append((cm[m].mean(), cp[m].mean()))
    sm, sp = np.mean(subj, axis=0)
    rows["subject_mean"] = {
        "n": int(len(subj)), "cos": float(sm), "persist": float(sp),
        "gain": float(sm - sp),
    }
    return rows


@torch.no_grad()
def collect(model: StaySplitModel, bundle, sids, device, *,
            window=None, batch=SHOT_BATCH) -> dict:
    window = int(window or model.window)
    idx = shot_index({s: bundle[s] for s in sids}, window, 1)
    buckets = {k: [] for k in
               ("vnow", "y", "lab0", "lab1", "g", "vhat_soft", "stay", "vel",
                "last_cos", "sid")}
    model.eval()
    for s0 in range(0, len(idx), batch):
        bi = idx[s0:s0 + batch]
        b = collate_shots(bundle, bi, window, 1, device)
        out = model(b["windows"], b["v1_now"], b["p0"])
        buckets["vnow"].append(b["v1_now"].cpu().numpy())
        buckets["y"].append(b["v1_future"][:, 0].cpu().numpy())
        buckets["lab0"].append(b["labels_now"].cpu().numpy())
        buckets["lab1"].append(b["labels_future"][:, 0].cpu().numpy())
        buckets["g"].append(out["g"].cpu().numpy())
        buckets["vhat_soft"].append(out["v1_hat"].cpu().numpy())
        buckets["stay"].append(out["stay"].cpu().numpy())
        buckets["vel"].append(out["vel"].cpu().numpy())
        if b["windows"].shape[1] >= 2:
            lc = (b["windows"][:, -1] * b["windows"][:, -2]).sum(dim=-1)
        else:
            lc = torch.ones(b["v1_now"].shape[0], device=device)
        buckets["last_cos"].append(lc.cpu().numpy())
        buckets["sid"].extend(s for s, _ in bi)
    pack = {k: np.concatenate(v) if k != "sid" else np.asarray(v)
            for k, v in buckets.items()}
    pack["n_shots"] = len(idx)
    return pack


def hard_compose(stay, vnow, g, last_cos, tau, last_cos_max):
    called = (g >= tau) & (last_cos < last_cos_max)
    out = np.where(called[:, None], -vnow, stay)
    out = out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)
    return out, called


def pick_operating_point(pack: dict, stay_floor: float = ODE23_STAY_FLOOR) -> dict:
    """Train-set τ, last_cos_max that max overall subject to stay ≥ floor.

    Preference order: satisfy the floor, then maximize overall, then stay.
    Falls back to the max-overall point if the floor is unreachable.
    """
    persist = pack["vnow"]
    y = pack["y"]
    best_ok = None
    best_any = None
    # Two scores: flip-head (high = flip) and last-step cosine (low = flip).
    # last_cos alone is the 0-param causal detector 2.1 never treated as
    # a primary invert score; it is the motion that precedes an antipode.
    taus = np.unique(np.concatenate([
        [-1.0],  # last_cos is the only gate
        np.quantile(pack["g"], np.linspace(0.50, 0.995, 32)),
    ]))
    lc_caps = (1.0, 0.4, 0.1, -0.1, -0.3, -0.5, -0.7)
    for lc_max in lc_caps:
        for tau in taus:
            yhat, called = hard_compose(
                pack["stay"], persist, pack["g"], pack["last_cos"],
                float(tau), float(lc_max))
            rows = summarize(yhat, y, persist, pack["lab0"], pack["lab1"], pack["sid"])
            stay = rows["stay"]["cos"]
            overall = rows["all"]["cos"]
            rec = {
                "tau": float(tau), "last_cos_max": float(lc_max),
                "call_rate": float(called.mean()),
                "stay": stay, "overall": overall,
                "flip": rows["flip"]["cos"],
                "small_switch": rows["small_switch"]["cos"],
                "subject_mean": rows["subject_mean"]["cos"],
                "metrics": rows,
            }
            key = (overall, stay)
            if best_any is None or key > (best_any["overall"], best_any["stay"]):
                best_any = rec
            if stay is not None and stay >= stay_floor:
                if best_ok is None or key > (best_ok["overall"], best_ok["stay"]):
                    best_ok = rec
    chosen = best_ok or best_any
    chosen["cleared_floor"] = bool(best_ok is not None)
    chosen["stay_floor"] = stay_floor
    return chosen


def evaluate_pack(pack: dict, tau: float, last_cos_max: float) -> dict:
    persist = pack["vnow"]
    yhat, called = hard_compose(
        pack["stay"], persist, pack["g"], pack["last_cos"], tau, last_cos_max)
    rows = summarize(yhat, pack["y"], persist,
                     pack["lab0"], pack["lab1"], pack["sid"])
    vel_rows = summarize(pack["vel"], pack["y"], persist,
                         pack["lab0"], pack["lab1"], pack["sid"])
    stay_rows = summarize(pack["stay"], pack["y"], persist,
                          pack["lab0"], pack["lab1"], pack["sid"])
    soft_rows = summarize(pack["vhat_soft"], pack["y"], persist,
                          pack["lab0"], pack["lab1"], pack["sid"])
    flip = flip_target(torch.from_numpy(pack["vnow"]),
                       torch.from_numpy(pack["y"])).numpy()
    return {
        "n_shots": pack["n_shots"],
        "tau": tau, "last_cos_max": last_cos_max,
        "call_rate": float(called.mean()),
        "flip_recall": float(called[flip].mean()) if flip.any() else None,
        "flip_precision": float(flip[called].mean()) if called.any() else None,
        "hard": rows,
        "stay_field": stay_rows,
        "velocity": vel_rows,
        "soft": soft_rows,
        "persist": summarize(persist, pack["y"], persist,
                             pack["lab0"], pack["lab1"], pack["sid"]),
    }
