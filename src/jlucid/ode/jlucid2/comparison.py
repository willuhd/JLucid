"""Model comparison A/B/C (plan Phase C / T4).

All three models operate in the SAME latent space produced by the trained
encoder (P4 fix):

- A: global linear dynamics  z_dot = A z,  A fit by least squares on finite
  differences of the encoder latent path (usable consecutive frames).
- B: switching linear dynamics z_dot = A_{s(t)} z, per-state least squares.
- C: the nonlinear Neural ODE (Phase B), evaluated through its own
  integrated path and decoder.

Metrics are computed on the fixed 10% val split: next-step R2 (V1 recon),
multi-step R2 up to delta=5, and next-state accuracy (softmax over k=3).
Bootstrap CIs (subjects) are reported for each metric.
"""

from __future__ import annotations

import numpy as np
import torch

from jlucid.ode.jlucid2.models import encode_no_ode
from jlucid.ode.jlucid2.train import load_subject_tensors


def fit_linear_dynamics(z: np.ndarray, labels: np.ndarray, mask: np.ndarray,
                        k_states: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares A for z_dot = A z on usable consecutive frame pairs.

    Returns (A_global (d,d), A_state (k,d,d)); A_state[s] uses only pairs
    whose starting frame has label s; states with no pairs are NaN.
    """
    d = z.shape[1]
    pairs = np.where(mask[:-1] & mask[1:])[0]
    X = z[pairs]
    Y = z[pairs + 1] - z[pairs]
    A_global, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A_state = np.full((k_states, d, d), np.nan)
    for s in range(k_states):
        sel = pairs[labels[pairs] == s]
        if len(sel) < d + 1:
            continue
        Xs, Ys = z[sel], z[sel + 1] - z[sel]
        A_state[s], *_ = np.linalg.lstsq(Xs, Ys, rcond=None)
    return A_global, A_state


def predict_linear(z0: np.ndarray, A: np.ndarray, steps: int,
                   labels: np.ndarray | None = None,
                   A_state: np.ndarray | None = None) -> np.ndarray | None:
    """Iterate z_{t+1} = z_t + A_s z_t for steps from z0 (d,)."""
    z = z0
    out = [z]
    for i in range(steps):
        a = A if A_state is None else A_state[labels[i]]
        if np.isnan(a).any():
            return None
        z = z + a @ z
        out.append(z)
    return np.stack(out)


@torch.no_grad()
def latent_path(model, h5, sid, stats, device, use_ode: bool = True) -> np.ndarray:
    v1, labels, mask, p, wmask = load_subject_tensors(h5, sid, stats, device)
    if use_ode:
        out = model(v1, p, sample=False)
        return out["z_hat"].detach().cpu().numpy()
    return encode_no_ode(model, v1).detach().cpu().numpy()


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def evaluate_model(model, h5, meta, stats, device, sids, k_states: int = 3,
                   max_delta: int = 5) -> list[dict]:
    """Evaluate A/B/C on the given subjects; returns per-subject metric rows."""
    rows = []
    decoder = model.decoder
    for sid in sids:
        v1, labels, mask, p, wmask = load_subject_tensors(h5, sid, stats, device)
        v1_np = v1.cpu().numpy()
        lab_np = labels.cpu().numpy()
        m = mask.cpu().numpy()
        t = len(lab_np)
        z_enc = encode_no_ode(model, v1).detach().cpu().numpy()
        start = model.window - 1
        z_pos = np.arange(start, t)
        A_global, A_state = fit_linear_dynamics(
            z_enc, lab_np[z_pos], m[z_pos], k_states)
        out = model(v1, p, sample=False)
        v1_hat_c = out["v1_hat"].detach().cpu().numpy()
        logits_c = out["state_logits"].detach().cpu().numpy()

        usable = m[start:t]
        idx = np.where(usable)[0] + start
        for delta in [1, 2, 3, 5]:
            if idx.size == 0:
                continue
            ii = idx[idx + delta < t]
            if ii.size == 0:
                continue
            target_t = ii + delta
            keep = m[target_t]
            if not keep.any():
                continue
            ii = ii[keep]
            tt = target_t[keep]
            y_true = v1_np[tt]
            r2_c = r2(y_true, v1_hat_c[tt])
            preds_a = []
            preds_b = []
            for j in range(len(ii)):
                za = predict_linear(z_enc[ii[j] - start], A_global, delta)
                preds_a.append(None if za is None else
                               decoder(torch.from_numpy(za[-1]).float().to(device))
                               [0].detach().cpu().numpy())
                zb = predict_linear(z_enc[ii[j] - start], A_global, delta,
                                    lab_np[ii[j]:ii[j] + delta], A_state)
                preds_b.append(None if zb is None else
                               decoder(torch.from_numpy(zb[-1]).float().to(device))
                               [0].detach().cpu().numpy())
            pa = np.stack([x for x in preds_a if x is not None])
            pb = np.stack([x for x in preds_b if x is not None])
            ja = [i for i, x in enumerate(preds_a) if x is not None]
            jb = [i for i, x in enumerate(preds_b) if x is not None]
            rows.append({
                "sid": sid, "delta": delta,
                "r2_c": r2_c,
                "r2_a": r2(y_true[ja], pa) if len(ja) else np.nan,
                "r2_b": r2(y_true[jb], pb) if len(jb) else np.nan,
            })
        ii = idx[idx + 1 < t]
        tt = ii + 1
        keep = m[tt]
        if keep.any():
            ii, tgt = ii[keep], lab_np[tt[keep]]
            acc_c = float((logits_c[tgt].argmax(axis=1) == tgt).mean())
            pred_a, pred_b = [], []
            for j in range(len(ii)):
                za = predict_linear(z_enc[ii[j] - start], A_global, 1)
                pred_a.append(None if za is None else int(
                    decoder(torch.from_numpy(za[-1]).float().to(device))
                    [1].detach().cpu().numpy().argmax()))
                zb = predict_linear(z_enc[ii[j] - start], A_global, 1,
                                    lab_np[ii[j]:ii[j] + 1], A_state)
                pred_b.append(None if zb is None else int(
                    decoder(torch.from_numpy(zb[-1]).float().to(device))
                    [1].detach().cpu().numpy().argmax()))
            ja = [i for i, x in enumerate(pred_a) if x is not None]
            jb = [i for i, x in enumerate(pred_b) if x is not None]
            acc_a = float(np.mean([pred_a[i] == tgt[i] for i in ja])) if ja else np.nan
            acc_b = float(np.mean([pred_b[i] == tgt[i] for i in jb])) if jb else np.nan
            rows.append({"sid": sid, "delta": 1, "acc_c": acc_c,
                         "acc_a": acc_a, "acc_b": acc_b})
    return rows


def bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float("nan"),) * 3
    means = np.array([rng.choice(v, size=len(v), replace=True).mean()
                      for _ in range(n)])
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
