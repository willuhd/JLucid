"""JLucid 3 shots: V1 windows plus optional V2/gap; stay-only index."""

from __future__ import annotations

import h5py
import numpy as np
import torch

from jlucid.config import FLIP_COS_THRESH


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
        if feat_h5 is not None and f"v2/{sid}" in feat_h5:
            rec["v2"] = feat_h5[f"v2/{sid}"][:].astype(np.float32)
            rec["gap"] = feat_h5[f"gap/{sid}"][:].astype(np.float32)
        else:
            rec["v2"] = np.zeros_like(rec["v1"])
            rec["gap"] = np.zeros(len(rec["v1"]), dtype=np.float32)
        bundle[str(sid)] = rec
    return bundle


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
    """Shots whose next frame stays in-cell and is not a flip jump."""
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
