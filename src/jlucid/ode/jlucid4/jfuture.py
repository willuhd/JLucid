"""Routing / future-output Jacobians of packed JLucid 4 dest/when + stay decode.

This is the plan Phase-7 object the A(p) stay-field Jacobian is *not*:

    J_route(t) = ∂ [flip_logit, dest_logits] / ∂ z_dest(t)     (4 × 12)
    J_out(t)   = ∂ V̂_1(t+1) / ∂ z_stay(t)                     (90 × 12)

J_route depends on dest_z through the tanh MLP, so a subject's mean
J_route is not an occupancy mix of three shared matrices. H3/H4/H5
in ``scripts/45_jfuture.py`` use J_route as the primary lens.
"""
from __future__ import annotations

import numpy as np
import torch

from jlucid.ode.jlucid4.jspace import (
    apply_zero,
    concentration,
    effective_rank,
    random_orthonormal,
    svd_right,
)
from jlucid.ode.jlucid4.models import ThesisModel
from jlucid.ode.jlucid4.shots import collate_shots, shot_index, stay_index


def dest_jacobian_batch(model: ThesisModel, z: torch.Tensor,
                        extras: torch.Tensor) -> torch.Tensor:
    """J_route for a batch. ``z`` (B,d), ``extras`` (B,3) → (B, 1+k, d).

    Closed form of the packed dest MLP ``Linear → tanh → Linear``.
    """
    net = model.dest_router.net
    lin1, lin2 = net[0], net[2]
    feat = torch.cat([z, extras], dim=-1)
    pre = feat @ lin1.weight.T + lin1.bias
    sech2 = 1.0 - torch.tanh(pre).pow(2)
    # d y / d feat = W2 @ diag(sech2) @ W1   batched
    # W2: (out, hidden), W1: (hidden, in)
    w2 = lin2.weight
    w1 = lin1.weight
    # (B, out, hidden) = W2 * sech2
    mid = w2.unsqueeze(0) * sech2.unsqueeze(1)
    dy_dfeat = mid @ w1  # (B, out, in)
    d = z.shape[-1]
    return dy_dfeat[:, :, :d]


def stay_output_jacobian_batch(
    stay, z: torch.Tensor, p: torch.Tensor, v_now: torch.Tensor,
    last_disp: torch.Tensor | None,
) -> torch.Tensor:
    """∂ V̂ / ∂ z0 of one stay step (γ-chord + Dec). Shape (B, 90, d)."""
    rows = []
    for i in range(z.shape[0]):
        zi = z[i].detach()
        pi = p[i: i + 1]
        vi = v_now[i: i + 1]
        di = None if last_disp is None else last_disp[i: i + 1]

        def _f(u, _p=pi, _v=vi, _d=di):
            z1 = stay.step(u.unsqueeze(0), _p)
            g = stay.gamma(u.unsqueeze(0))
            extra = None if _d is None or stay.gamma_mode == "none" else g * _d
            return stay.decode(z1, _v, extra).squeeze(0)

        rows.append(torch.autograd.functional.jacobian(_f, zi))
    return torch.stack(rows, dim=0)


def route_from_mean_J(J_bar: np.ndarray, *, k: int = 3, n_frames: int = 0) -> dict:
    """SVD / C_k / r_eff of a (possibly rectangular) subject-mean Jacobian."""
    J_bar = np.asarray(J_bar, dtype=float)
    U, s, V = svd_right(J_bar)
    d_right = V.shape[0]
    kk = min(int(k), V.shape[1])
    V_k = V[:, :kk]
    rec = {
        "J": J_bar,
        "U": U,
        "svals": s,
        "V": V,
        "V_k": V_k,
        "C_k": concentration(s, kk),
        "C_1": concentration(s, 1),
        "r_eff": effective_rank(s),
        "n": int(n_frames),
        "latent": int(d_right),
        "k": int(kk),
        "jacobian_kind": "route",
    }
    if J_bar.shape[0] == J_bar.shape[1]:
        rec["A_J"] = V_k.T @ J_bar @ V_k
    else:
        rec["A_J"] = V_k.T @ (J_bar.T @ J_bar) @ V_k
    return rec


def occupancy_from_labels(labels: np.ndarray, n_states: int = 3) -> np.ndarray:
    lab = np.asarray(labels, dtype=int)
    ok = lab >= 0
    out = np.zeros(n_states, dtype=float)
    if not ok.any():
        return out
    for s in range(n_states):
        out[s] = float((lab[ok] == s).mean())
    return out


def grassmann_mean_overlap(Vs: list[np.ndarray], V_ref: np.ndarray) -> float:
    """Mean principal-angle cosine of bootstrap V_k vs reference V_k."""
    if not Vs:
        return float("nan")
    ref = np.asarray(V_ref, dtype=float)
    q_ref, _ = np.linalg.qr(ref)
    acc = []
    for V in Vs:
        q, _ = np.linalg.qr(np.asarray(V, dtype=float))
        s = np.linalg.svd(q.T @ q_ref, compute_uv=False)
        acc.append(float(s.mean()))
    return float(np.mean(acc))


def bootstrap_axis_stability(Js: np.ndarray, *, k: int = 3, n_boot: int = 20,
                             seed: int = 42) -> float:
    """Stability of right-J-space axes under frame bootstrap of J(t)."""
    Js = np.asarray(Js, dtype=float)
    if Js.ndim != 3 or len(Js) < 8:
        return float("nan")
    _, _, V = svd_right(Js.mean(axis=0))
    V_k = V[:, :k]
    rng = np.random.default_rng(int(seed))
    boots = []
    n = len(Js)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        _, _, Vb = svd_right(Js[idx].mean(axis=0))
        boots.append(Vb[:, :k])
    return grassmann_mean_overlap(boots, V_k)


@torch.no_grad()
def _encode_dest(model, windows, v1_now, v2_now, gap_now):
    z = model.dest_encoder(windows)
    extras = model.dest_router.extras(windows, v1_now, v2_now, gap_now)
    return z, extras


def collect_route_subject(
    model: ThesisModel,
    bundle: dict,
    sid: str,
    device,
    *,
    window: int,
    batch: int,
    n_states: int = 3,
) -> dict | None:
    """All usable Δ=1 shots: dest J_route, labels, dest_z, extras."""
    rec = bundle[sid]
    idx = shot_index({sid: rec}, window, 1)
    if not idx:
        return None
    zs, extras, Js, labs, nows, nxts, lab1, times = [], [], [], [], [], [], [], []
    model.eval()
    for s0 in range(0, len(idx), batch):
        bi = idx[s0: s0 + batch]
        b = collate_shots(bundle, bi, window, 1, device)
        z, extra = _encode_dest(
            model, b["windows"], b["v1_now"], b["v2_now"], b["gap_now"])
        J = dest_jacobian_batch(model, z, extra)
        zs.append(z.detach().cpu().numpy())
        extras.append(extra.detach().cpu().numpy())
        Js.append(J.detach().cpu().numpy())
        labs.append(b["labels_now"].detach().cpu().numpy())
        lab1.append(b["labels_future"][:, 0].detach().cpu().numpy())
        nows.append(b["v1_now"].detach().cpu().numpy())
        nxts.append(b["v1_future"][:, 0].detach().cpu().numpy())
        times.extend(int(t) for _, t in bi)
    J = np.concatenate(Js, axis=0)
    labels = np.concatenate(labs, axis=0).astype(int)
    labels_next = np.concatenate(lab1, axis=0).astype(int)
    z = np.concatenate(zs, axis=0)
    rec_out = route_from_mean_J(J.mean(axis=0), n_frames=int(J.shape[0]))
    rec_out["sid"] = sid
    rec_out["J_t"] = J.astype(np.float64)
    rec_out["z"] = z.astype(np.float64)
    rec_out["extras"] = np.concatenate(extras, axis=0).astype(np.float64)
    rec_out["labels"] = labels
    rec_out["labels_next"] = labels_next
    rec_out["v1_now"] = np.concatenate(nows, axis=0).astype(np.float32)
    rec_out["v1_next"] = np.concatenate(nxts, axis=0).astype(np.float32)
    rec_out["times"] = np.asarray(times, dtype=int)
    rec_out["dwell"] = rec["dwell"][rec_out["times"]].astype(np.float32)
    rec_out["occ"] = occupancy_from_labels(labels, n_states)
    rec_out["switch_rate"] = float((labels != labels_next).mean())
    rec_out["axis_stability"] = bootstrap_axis_stability(J, k=rec_out["k"])
    rec_out["z_var"] = float(z.var())
    rec_out["J_t_var"] = float(J.var())
    # origin-conditioned routing energy: mean |∂ dest_logit_s / ∂z| when in s
    row_energy = np.linalg.norm(J, axis=-1)  # (n, 1+k)
    rec_out["flip_row_norm"] = float(row_energy[:, 0].mean())
    rec_out["dest_row_norm"] = float(row_energy[:, 1:].mean())
    dmn_leave = []
    for s in range(n_states):
        sel = labels == s
        rec_out[f"row_norm_origin_{s}"] = (
            float(row_energy[sel].mean()) if sel.any() else float("nan"))
        if s == 1 and sel.any():
            # dest logits for non-DMN targets (cols 1+0 and 1+2 of J, 0-index:
            # col 0 flip, 1 dest0, 2 dest1, 3 dest2)
            dmn_leave.append(row_energy[sel][:, [1, 3]].mean())
    rec_out["dmn_leave_energy"] = (
        float(np.mean(dmn_leave)) if dmn_leave else float("nan"))
    return rec_out


def collect_stay_out_subject(
    model: ThesisModel,
    bundle: dict,
    sid: str,
    device,
    *,
    window: int,
    batch: int,
    max_frames: int,
    seed: int = 42,
) -> dict | None:
    """Sampled in-dwell ∂V̂/∂z (90×12). Secondary lens."""
    rec = bundle[sid]
    idx = stay_index({sid: rec}, window, 1)
    if not idx:
        return None
    rng = np.random.default_rng(int(seed) + (sum(ord(c) for c in str(sid)) % 10_000))
    if len(idx) > int(max_frames):
        pick = rng.choice(len(idx), size=int(max_frames), replace=False)
        idx = [idx[int(i)] for i in np.sort(pick)]
    Js, zs = [], []
    stay = model.stay
    model.eval()
    for s0 in range(0, len(idx), batch):
        bi = idx[s0: s0 + batch]
        b = collate_shots(bundle, bi, window, 1, device)
        z0 = model.encoder(b["windows"])
        last_disp = None
        if b["windows"].shape[1] >= 2:
            last_disp = b["windows"][:, -1] - b["windows"][:, -2]
        J = stay_output_jacobian_batch(
            stay, z0, b["p0"], b["v1_now"], last_disp)
        Js.append(J.detach().cpu().numpy())
        zs.append(z0.detach().cpu().numpy())
    J = np.concatenate(Js, axis=0)
    rec_out = route_from_mean_J(J.mean(axis=0), n_frames=int(J.shape[0]))
    rec_out["sid"] = sid
    rec_out["jacobian_kind"] = "stay_out"
    rec_out["z"] = np.concatenate(zs, axis=0)
    return rec_out


def dest_logits_from_z(model: ThesisModel, z: torch.Tensor,
                       extras: torch.Tensor) -> torch.Tensor:
    feat = torch.cat([z, extras], dim=-1)
    return model.dest_router.net(feat)


def route_arrays_from_z(
    z: np.ndarray,
    extras: np.ndarray,
    origin: np.ndarray,
    dest_true: np.ndarray,
    dwell: np.ndarray,
    v1_now: np.ndarray,
    v1_next: np.ndarray,
    raw_logits: np.ndarray,
    *,
    dgap: np.ndarray | None = None,
    centroids: np.ndarray | None = None,
    flip_thresh: float = -0.5,
) -> dict:
    """Build the when-head feature arrays after dest logits are known."""
    from jlucid.ode.jlucid4.discrete import dest_margin, persist_prob
    from jlucid.config import FLIP_COS_THRESH

    thresh = float(flip_thresh if flip_thresh is not None else FLIP_COS_THRESH)
    origin = np.asarray(origin, dtype=int)
    dest_true = np.asarray(dest_true, dtype=int)
    raw = np.asarray(raw_logits, dtype=np.float64)
    flip_logit = raw[:, 0]
    dest_logits = raw[:, 1:]
    dest_m = dest_logits.copy()
    dest_m[np.arange(len(origin)), origin.clip(0, dest_logits.shape[1] - 1)] = -1e9
    dest_hat = dest_m.argmax(-1)
    persist_cos = (np.asarray(v1_now) * np.asarray(v1_next)).sum(-1)
    n = len(origin)
    if centroids is None:
        cmargin = np.zeros(n, dtype=np.float64)
    else:
        ccos = np.asarray(v1_now, dtype=np.float64) @ np.asarray(centroids, dtype=np.float64).T
        origin_c = ccos[np.arange(n), origin.clip(0, ccos.shape[1] - 1)]
        tmp = ccos.copy()
        tmp[np.arange(n), origin.clip(0, ccos.shape[1] - 1)] = -1e9
        cmargin = origin_c - tmp.max(axis=-1)
    extras = np.asarray(extras, dtype=np.float64)
    return {
        "origin": origin,
        "dest": dest_true,
        "dest_masked": dest_hat.astype(np.int64),
        "flip": persist_cos < thresh,
        "last_cos": extras[:, 0].astype(np.float32),
        "v1v2": extras[:, 1].astype(np.float32),
        "gap": extras[:, 2].astype(np.float32),
        "dgap": (np.zeros(n, dtype=np.float32) if dgap is None
                 else np.asarray(dgap, dtype=np.float32)),
        "dwell": np.asarray(dwell, dtype=np.float32),
        "dest_margin": dest_margin(dest_logits, origin).astype(np.float32),
        "persist_p": persist_prob(dest_logits, origin).astype(np.float32),
        "centroid_margin": np.asarray(cmargin, dtype=np.float32),
        "dest_logits": dest_logits.astype(np.float32),
        "flip_logit": flip_logit.astype(np.float32),
        "flip_score": (1.0 / (1.0 + np.exp(-np.clip(flip_logit, -40, 40)))).astype(np.float32),
        "z0": np.asarray(z, dtype=np.float32),
    }


def dest_on_subset(origin, dest_true, dest_hat, mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float((np.asarray(dest_hat)[mask] == np.asarray(dest_true)[mask]).mean())


def route_logits_numpy(model: ThesisModel, z: np.ndarray, extras: np.ndarray,
                       device) -> np.ndarray:
    zt = torch.from_numpy(np.asarray(z, dtype=np.float32)).to(device)
    et = torch.from_numpy(np.asarray(extras, dtype=np.float32)).to(device)
    with torch.no_grad():
        return dest_logits_from_z(model, zt, et).cpu().numpy()
