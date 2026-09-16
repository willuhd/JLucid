import numpy as np
import torch

from jlucid.ode.jlucid2.shots import collate_shots, shot_index


def _toy_bundle():
    t = 20
    v1 = np.random.randn(t, 90).astype(np.float32)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    mask = np.ones(t, dtype=bool)
    mask[8] = False
    labels = np.zeros(t, dtype=np.int64)
    p = np.zeros((t, 3), dtype=np.float32)
    p[:, 0] = 1
    return {"a": {"v1": v1, "labels": labels, "mask": mask, "p": p}}


def test_shot_index_skips_censor_gaps():
    bundle = _toy_bundle()
    idx = shot_index(bundle, window=4, horizon=2)
    # a shot covering [t-3, t+2] cannot include frame 8
    for _, t in idx:
        assert 8 not in range(t - 3, t + 3)
    assert idx  # some shots still exist on each side of the gap


def test_collate_shapes():
    bundle = _toy_bundle()
    idx = shot_index(bundle, window=4, horizon=2)
    batch = collate_shots(bundle, idx[:5], window=4, horizon=2,
                          device=torch.device("cpu"))
    assert batch["windows"].shape == (5, 4, 90)
    assert batch["v1_future"].shape == (5, 2, 90)
    assert batch["p0"].shape == (5, 3)
    assert batch["labels_future"].shape == (5, 2)
