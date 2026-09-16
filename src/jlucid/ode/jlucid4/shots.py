"""JLucid 4 shots: windows, stay/dwell indexes, switch stacks."""

from __future__ import annotations

import h5py
import numpy as np
import torch

from jlucid.config import FLIP_COS_THRESH, STEP3_K_COND


def load_bundle(h5: h5py.File, sids: list[str], *,
                feat_h5: h5py.File | None = None) -> dict:
    bundle = {}
    for sid in sids:
        rec = {
            "v1": h5[f"v1/{sid}"][:].astype(np.float32),
            "labels": h5[f"labels/{sid}"][:].astype(np.int64),
            "mask": h5[f"mask/{sid}"][:].astype(bool),
            "p": h5[f"p/{sid}"][:].astype(np.float32),
        }
        t_len = len(rec["v1"])
        if feat_h5 is not None and f"v2/{sid}" in feat_h5:
            rec["v2"] = feat_h5[f"v2/{sid}"][:].astype(np.float32)
            rec["gap"] = feat_h5[f"gap/{sid}"][:].astype(np.float32)
        else:
            rec["v2"] = np.zeros_like(rec["v1"])
            rec["gap"] = np.zeros(t_len, dtype=np.float32)
        rec["dwell"] = _dwell_series(rec["labels"])
        bundle[str(sid)] = rec
    return bundle


def _dwell_series(labels: np.ndarray) -> np.ndarray:
    d = np.empty(len(labels), dtype=np.float32)
    run = 1.0
    d[0] = 1.0
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1] and labels[i] >= 0:
            run += 1.0
        else:
            run = 1.0
        d[i] = run
    return d


def shot_index(bundle: dict, window: int, horizon: int = 1) -> list[tuple[str, int]]:
    idx: list[tuple[str, int]] = []
    for sid, rec in bundle.items():
        m = rec["mask"]
        t_len = len(m)
        for t in range(window - 1, t_len - horizon):
            if bool(m[t - window + 1: t + horizon + 1].all()):
                idx.append((sid, int(t)))
    return idx


def stay_index(bundle: dict, window: int, horizon: int = 1,
               flip_thresh: float = FLIP_COS_THRESH) -> list[tuple[str, int]]:
    idx = []
    for sid, rec in bundle.items():
        m, lab, v1 = rec["mask"], rec["labels"], rec["v1"]
        t_len = len(m)
        for t in range(window - 1, t_len - horizon):
            if not bool(m[t - window + 1: t + horizon + 1].all()):
                continue
            if lab[t] < 0 or lab[t + 1] < 0:
                continue
            if lab[t] != lab[t + 1]:
                continue
            if float(v1[t] @ v1[t + 1]) < flip_thresh:
                continue
            idx.append((sid, int(t)))
    return idx


def dwell_index(bundle: dict, window: int, horizon: int,
                flip_thresh: float = FLIP_COS_THRESH) -> list[tuple[str, int]]:
    """Whole ``[t, t+H]`` in one label, no flip-sized consecutive jump."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    idx: list[tuple[str, int]] = []
    for sid, rec in bundle.items():
        m, lab, v1 = rec["mask"], rec["labels"], rec["v1"]
        t_len = len(m)
        if t_len <= window - 1 + horizon:
            continue
        dots = np.einsum("td,td->t", v1[:-1], v1[1:])
        for t in range(window - 1, t_len - horizon):
            if not bool(m[t - window + 1: t + horizon + 1].all()):
                continue
            labs = lab[t: t + horizon + 1]
            if int(labs[0]) < 0 or not bool(np.all(labs == labs[0])):
                continue
            if bool(np.any(dots[t: t + horizon] < flip_thresh)):
                continue
            idx.append((sid, int(t)))
    return idx


def collate_shots(bundle: dict, batch: list[tuple[str, int]],
                  window: int, horizon: int, device: torch.device) -> dict:
    windows, p0, v1_now, v1_fut = [], [], [], []
    v2_now, gap_now, lab_now, lab_fut = [], [], [], []
    for sid, t in batch:
        rec = bundle[sid]
        windows.append(rec["v1"][t - window + 1: t + 1])
        p0.append(rec["p"][t])
        v1_now.append(rec["v1"][t])
        v1_fut.append(rec["v1"][t + 1: t + 1 + horizon])
        v2_now.append(rec["v2"][t])
        gap_now.append(rec["gap"][t])
        lab_now.append(rec["labels"][t])
        lab_fut.append(rec["labels"][t + 1: t + 1 + horizon])
    return {
        "windows": torch.from_numpy(np.stack(windows)).to(device),
        "p0": torch.from_numpy(np.stack(p0)).to(device),
        "v1_now": torch.from_numpy(np.stack(v1_now)).to(device),
        "v1_future": torch.from_numpy(np.stack(v1_fut)).to(device),
        "v2_now": torch.from_numpy(np.stack(v2_now)).to(device),
        "gap_now": torch.from_numpy(np.asarray(gap_now, dtype=np.float32)).to(device),
        "labels_now": torch.from_numpy(np.asarray(lab_now, dtype=np.int64)).to(device),
        "labels_future": torch.from_numpy(np.stack(lab_fut)).to(device),
    }


def stack_switch_shots(bundle: dict, index: list[tuple[str, int]],
                       window: int) -> dict[str, np.ndarray]:
    n = len(index)
    if n == 0:
        raise ValueError("empty shot index")
    dim = next(iter(bundle.values()))["v1"].shape[-1]
    v1_w = np.empty((n, window, dim), dtype=np.float32)
    gap_w = np.empty((n, window), dtype=np.float32)
    v1_now = np.empty((n, dim), dtype=np.float32)
    v1_next = np.empty((n, dim), dtype=np.float32)
    v2_now = np.empty((n, dim), dtype=np.float32)
    gap_now = np.empty(n, dtype=np.float32)
    p0 = np.empty((n, bundle[index[0][0]]["p"].shape[-1]), dtype=np.float32)
    lab0 = np.empty(n, dtype=np.int64)
    lab1 = np.empty(n, dtype=np.int64)
    dwell = np.empty(n, dtype=np.float32)
    for i, (sid, t) in enumerate(index):
        rec = bundle[sid]
        sl = slice(t - window + 1, t + 1)
        v1_w[i] = rec["v1"][sl]
        gap_w[i] = rec["gap"][sl]
        v1_now[i] = rec["v1"][t]
        v1_next[i] = rec["v1"][t + 1]
        v2_now[i] = rec["v2"][t]
        gap_now[i] = rec["gap"][t]
        p0[i] = rec["p"][t]
        lab0[i] = rec["labels"][t]
        lab1[i] = rec["labels"][t + 1]
        dwell[i] = rec["dwell"][t]
    return {
        "v1_win": v1_w, "gap_win": gap_w,
        "v1_now": v1_now, "v1_next": v1_next, "v2_now": v2_now,
        "gap_now": gap_now, "p0": p0, "labels_now": lab0, "labels_next": lab1,
        "dwell": dwell,
    }


def state_centroids(bundle: dict, sids: list[str] | None = None,
                    k: int = STEP3_K_COND) -> np.ndarray:
    sids = list(sids) if sids is not None else list(bundle.keys())
    acc: list[list[np.ndarray]] = [[] for _ in range(k)]
    for sid in sids:
        rec = bundle[sid]
        m = rec["mask"] & (rec["labels"] >= 0)
        for s in range(k):
            sel = m & (rec["labels"] == s)
            if sel.any():
                acc[s].append(rec["v1"][sel])
    cents = []
    dim = next(iter(bundle.values()))["v1"].shape[-1]
    for s in range(k):
        if not acc[s]:
            cents.append(np.zeros(dim, dtype=np.float32))
            continue
        c = np.concatenate(acc[s], axis=0).mean(axis=0)
        nrm = float(np.linalg.norm(c))
        cents.append((c / max(nrm, 1e-12)).astype(np.float32))
    return np.stack(cents)
