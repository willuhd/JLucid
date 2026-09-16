"""Short-shot indexing and batching for the Neural ODE.

A shot is a usable window ending at t plus H future frames. Censored
frames break shots (we never integrate through a mask gap).
"""

from __future__ import annotations

import h5py
import numpy as np
import torch


def load_bundle(h5: h5py.File, sids: list[str], *,
                native: bool = True, stats: dict | None = None) -> dict:
    """Preload per-subject arrays. ~30 MB for the full 307-subject cohort."""
    bundle = {}
    mean = std = None
    if not native:
        if stats is None:
            raise ValueError("z-scored bundle needs stats")
        mean = np.asarray(stats["v1_mean"], dtype=np.float32)
        std = np.maximum(np.asarray(stats["v1_std"], dtype=np.float32), 1e-8)
    for sid in sids:
        v1 = h5[f"v1/{sid}"][:].astype(np.float32)
        if not native:
            v1 = (v1 - mean) / std
        bundle[str(sid)] = {
            "v1": v1,
            "labels": h5[f"labels/{sid}"][:].astype(np.int64),
            "mask": h5[f"mask/{sid}"][:].astype(bool),
            "p": h5[f"p/{sid}"][:].astype(np.float32),
        }
    return bundle


def shot_index(bundle: dict, window: int, horizon: int) -> list[tuple[str, int]]:
    """(sid, t) where t is 'now' (last encoder frame) and [t-w+1, t+H] is usable."""
    idx: list[tuple[str, int]] = []
    for sid, rec in bundle.items():
        m = rec["mask"]
        t_len = len(m)
        for t in range(window - 1, t_len - horizon):
            if bool(m[t - window + 1 : t + horizon + 1].all()):
                idx.append((sid, int(t)))
    return idx


def collate_shots(bundle: dict, batch: list[tuple[str, int]],
                  window: int, horizon: int, device: torch.device) -> dict:
    windows, p0, v1_now, v1_fut, lab_now, lab_fut = [], [], [], [], [], []
    for sid, t in batch:
        rec = bundle[sid]
        windows.append(rec["v1"][t - window + 1 : t + 1])
        p0.append(rec["p"][t])
        v1_now.append(rec["v1"][t])
        v1_fut.append(rec["v1"][t + 1 : t + 1 + horizon])
        lab_now.append(rec["labels"][t])
        lab_fut.append(rec["labels"][t + 1 : t + 1 + horizon])
    return {
        "windows": torch.from_numpy(np.stack(windows)).to(device),
        "p0": torch.from_numpy(np.stack(p0)).to(device),
        "v1_now": torch.from_numpy(np.stack(v1_now)).to(device),
        "v1_future": torch.from_numpy(np.stack(v1_fut)).to(device),
        "labels_now": torch.from_numpy(np.asarray(lab_now, dtype=np.int64)).to(device),
        "labels_future": torch.from_numpy(np.stack(lab_fut)).to(device),
    }
