"""EiDA extras for JLucid 3: V2(t) and |λ1|-|λ2| from stored Hilbert phase."""

from __future__ import annotations

import logging

import h5py
import numpy as np

from jlucid.config import ODE3_FEATURES_H5, STEP3_DATA_H5, TC_FILE
from jlucid.leida.phases import dpc_matrix, leading_two

log = logging.getLogger("ode3_features")


def v2_gap_series(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame V2, gap |λ1|-|λ2|, and λ1. Sign-align V2 to the previous frame."""
    t, n = phase.shape
    v2 = np.empty((t, n), dtype=np.float32)
    gap = np.empty(t, dtype=np.float32)
    lam1 = np.empty(t, dtype=np.float32)
    prev = None
    for i in range(t):
        _, vec2, l1, l2 = leading_two(dpc_matrix(phase[i]))
        if prev is not None and float(vec2 @ prev) < 0:
            vec2 = -vec2
        v2[i] = vec2
        gap[i] = abs(l1) - abs(l2)
        lam1[i] = l1
        prev = vec2
    return v2, gap, lam1


def build_features(out_path=ODE3_FEATURES_H5, data_h5=STEP3_DATA_H5,
                   tc_path=TC_FILE) -> int:
    """Write /v2 /gap /lam1 per subject for every raw_id in train_data.h5."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with h5py.File(data_h5, "r") as data, h5py.File(tc_path, "r") as tc, \
            h5py.File(out_path, "w") as out:
        sids = list(data["v1"].keys())
        for sid in sids:
            if f"phase/{sid}" not in tc:
                raise KeyError(f"no phase for {sid}")
            phase = tc[f"phase/{sid}"][:].astype(np.float64)
            v1 = data[f"v1/{sid}"]
            if len(phase) != len(v1):
                raise ValueError(
                    f"length mismatch {sid}: phase {len(phase)} v1 {len(v1)}")
            v2, gap, lam1 = v2_gap_series(phase)
            out.create_dataset(f"v2/{sid}", data=v2, compression="gzip")
            out.create_dataset(f"gap/{sid}", data=gap, compression="gzip")
            out.create_dataset(f"lam1/{sid}", data=lam1, compression="gzip")
            n += 1
            if n % 50 == 0:
                log.info("features %d/%d", n, len(sids))
        out.attrs["n_subjects"] = n
    log.info("wrote %s for %d subjects", out_path, n)
    return n
