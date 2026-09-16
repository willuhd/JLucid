"""Closed-form J = A(p) smoke, spectral summary, compact J-space peek."""

from __future__ import annotations

import numpy as np
import torch

from jlucid.config import PROCESSED_DIR, SHOT_BATCH, WINDOW_TR
from jlucid.ode.jlucid4.models import autograd_field_jacobian
from jlucid.ode.jlucid4.shots import collate_shots, stay_index


STATE_NAMES = {0: "VIS-DAN", 1: "DMN-LIMBIC", 2: "SMN-VAN"}


def jacobian_smoke(model, bundle, sids, device, *, window=None,
                   n: int = 8) -> dict:
    window = int(window or getattr(model, "window", WINDOW_TR))
    sub = {s: bundle[s] for s in sids}
    idx = stay_index(sub, window, 1)
    if not idx:
        return {"ok": False, "reason": "no stay shots"}
    b = collate_shots(bundle, idx[:n], window, 1, device)
    model.eval()
    z0 = model.encoder(b["windows"])
    p0 = b["p0"]
    stay = model.stay
    J_auto = autograd_field_jacobian(stay.field, z0, p0)
    J_closed = stay.field_jacobian(z0, p0)
    d = int(z0.shape[-1])
    match = bool(torch.allclose(J_auto, J_closed, atol=1e-5, rtol=1e-4))
    finite = bool(torch.isfinite(J_auto).all() and torch.isfinite(J_closed).all())
    max_abs = float(J_auto.abs().max())
    return {
        "ok": bool(finite and match and J_auto.shape[-1] == d
                   and J_auto.shape[-2] == d and max_abs > 1e-12),
        "shape": list(J_auto[0].shape),
        "batch_shape": list(J_auto.shape),
        "finite": finite,
        "matches_closed_form": match,
        "all_zero": bool(max_abs < 1e-12),
        "max_abs": max_abs,
        "mean_abs": float(J_auto.abs().mean()),
        "n": int(z0.shape[0]),
        "latent": d,
        "eps": 0.0,
        "gamma_mode": stay.gamma_mode,
    }


def spectral_report(model) -> dict:
    rows = model.stay.spectral_summary()
    return {
        "per_state": rows,
        "max_op_norm": max(r["op_norm"] for r in rows),
        "max_cond_IplusA": max(r["cond_IplusA"] for r in rows),
        "max_spectral_radius": max(r["spectral_radius"] for r in rows),
        "sane_linearization": bool(max(r["cond_IplusA"] for r in rows) < 10.0),
        "identifiable": True,
        "jacobian_is": "A(p) = sum_s p_s A_s  (12x12, eps=0, chord in decode)",
    }


@torch.no_grad()
def jspace_peek(model, bundle, sids, device, *, window=None,
                eps: float = 0.1, n_sv: int = 3, n_mean: int = 256) -> dict:
    """SVD of each A_s; Dec residual along right singular vectors.

    Descriptive, not a clinical test.
    """
    window = int(window or getattr(model, "window", WINDOW_TR))
    stay = model.stay
    A = stay.A.detach().cpu().numpy()
    k, d, _ = A.shape
    cents_path = PROCESSED_DIR / "state_centroids_k3.npy"
    cents = None
    if cents_path.exists():
        cents = np.load(cents_path).astype(np.float64)
        cents = cents / np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12)

    sub = {s: bundle[s] for s in sids}
    idx = stay_index(sub, window, 1)
    mean_z = torch.zeros(d, device=device)
    if idx:
        zs = []
        model.eval()
        for s0 in range(0, min(len(idx), n_mean), SHOT_BATCH):
            bi = idx[s0:s0 + SHOT_BATCH]
            b = collate_shots(bundle, bi, window, 1, device)
            zs.append(model.encoder(b["windows"]))
        mean_z = torch.cat(zs, dim=0).mean(dim=0)

    per_state = []
    for s in range(k):
        As = A[s]
        _u, svals, vt = np.linalg.svd(As, full_matrices=False)
        p = torch.zeros(1, k, device=device)
        p[0, s] = 1.0
        probes = []
        for at, z_np in (("zero", torch.zeros(d, device=device)),
                         ("mean_stay", mean_z)):
            dirs = []
            z = z_np.reshape(1, d)
            dec0 = stay.decoder(stay.step(z, p)).squeeze(0).cpu().numpy()
            for i in range(min(n_sv, d)):
                v = torch.from_numpy(vt[i].astype(np.float32)).to(device)
                zp = z + float(eps) * v.reshape(1, d)
                dec1 = stay.decoder(stay.step(zp, p)).squeeze(0).cpu().numpy()
                resid = dec1 - dec0
                rn = float(np.linalg.norm(resid))
                row = {"sv": int(i), "sval": float(svals[i]), "residual_norm": rn}
                if cents is not None and rn > 1e-12:
                    ures = resid / rn
                    row["cos_centroids"] = (cents @ ures).astype(float).tolist()
                dirs.append(row)
            probes.append({"at": at, "eps": float(eps), "dirs": dirs})
        per_state.append({
            "state": int(s),
            "name": STATE_NAMES.get(s, str(s)),
            "svals": svals.astype(float).tolist(),
            "top3_svals": svals[:n_sv].astype(float).tolist(),
            "top3_frob_frac": float(
                np.sqrt((svals[:n_sv] ** 2).sum())
                / max(float(np.linalg.norm(As)), 1e-12)
            ),
            "frob": float(np.linalg.norm(As)),
            "op_norm": float(svals[0]),
            "decode_probes": probes,
        })
    return {"eps": float(eps), "n_sv": int(n_sv), "per_state": per_state,
            "clinical": False}
