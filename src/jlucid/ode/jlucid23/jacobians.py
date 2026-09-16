"""Stay-field Jacobians and a first model-based J-space peek.

J_dyn(t) = ∂f_stay/∂z |_{z0,p0}          (12 × 12)
J_v1(t)  = ∂v_stay/∂z0                    (90 × 12)
J_future is the stay-path V1 Jacobian; discrete invert is not differentiated.

Per-subject Jbar = mean_t J_v1(t) over *stay* frames. SVD of Jbar gives
the model-based Jacobian subspace (right singular vectors in latent z).
"""

from __future__ import annotations

import numpy as np
import torch

from jlucid.config import YEO_NETWORKS
from jlucid.io.athena import load_network_membership
from jlucid.ode.jlucid23.models import StaySplitModel, stay_mask
from jlucid.ode.jlucid23.shots import collate_shots, shot_index


def j_dyn(model: StaySplitModel, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """(B, d, d) Jacobian of f_stay, one matrix per row."""
    z = z.detach().requires_grad_(True)
    p = p.detach()
    f = model.f_stay(z, p)
    d = z.shape[-1]
    rows = []
    for i in range(d):
        grad = torch.autograd.grad(f[:, i].sum(), z, retain_graph=True)[0]
        rows.append(grad)
    return torch.stack(rows, dim=1)


def concentration(svals: np.ndarray, k: int = 3) -> float:
    s2 = np.asarray(svals, dtype=float) ** 2
    tot = float(s2.sum())
    if tot <= 0:
        return float("nan")
    return float(s2[:k].sum() / tot)


def effective_rank(svals: np.ndarray) -> float:
    s = np.asarray(svals, dtype=float)
    s = s[s > 0]
    if s.size == 0:
        return float("nan")
    return float((s.sum() ** 2) / (s ** 2).sum())


def _network_masks() -> dict[str, np.ndarray]:
    mem = load_network_membership()
    out = {}
    for g in ["Subcortical"] + YEO_NETWORKS:
        idx = mem.loc[mem.network_group == g, "idx"].astype(int).to_numpy() - 1
        m = np.zeros(90, dtype=bool)
        m[idx] = True
        out[g] = m
    return out


@torch.no_grad()
def _stay_indices(bundle, sids, window: int) -> list[tuple[str, int]]:
    idx = shot_index({s: bundle[s] for s in sids}, window, 1)
    keep = []
    for sid, t in idx:
        rec = bundle[sid]
        v0, v1 = rec["v1"][t], rec["v1"][t + 1]
        lab0, lab1 = rec["labels"][t], rec["labels"][t + 1]
        if bool(stay_mask(torch.as_tensor(v0[None]), torch.as_tensor(v1[None]),
                          torch.as_tensor([lab0]), torch.as_tensor([lab1]))[0]):
            keep.append((sid, t))
    return keep


def subject_jspace(model: StaySplitModel, bundle: dict, sid: str, device,
                   window: int | None = None, max_frames: int = 80) -> dict:
    """Mean stay-path V1 Jacobian and its SVD for one subject."""
    window = int(window or model.window)
    idx = _stay_indices(bundle, [sid], window)
    if not idx:
        return {"sid": sid, "n": 0}
    if len(idx) > max_frames:
        rng = np.random.default_rng(42)
        pick = rng.choice(len(idx), size=max_frames, replace=False)
        idx = [idx[int(i)] for i in pick]
    model.eval()
    jd = []
    for chunk in range(0, len(idx), 16):
        bi = idx[chunk:chunk + 16]
        b = collate_shots(bundle, bi, window, 1, device)
        z0 = model.encode(b["windows"])
        jd.append(j_dyn(model, z0, b["p0"]).detach().cpu().numpy())
    J = np.concatenate(jd, axis=0).mean(axis=0)  # (12, 12)
    u, s, vt = np.linalg.svd(J, full_matrices=False)
    nets = _network_masks()
    # network loading of right singular vectors mapped by stay residual
    # decoder: use mean |stay - vel| as a proxy ROI pattern
    b_all = collate_shots(bundle, idx[: min(32, len(idx))], window, 1, device)
    with torch.no_grad():
        z0 = model.encode(b_all["windows"])
        z1 = model.stay_step(z0, b_all["p0"])
        prev = model.v_prev(b_all["windows"], b_all["v1_now"])
        stay = model.decode_stay(z1, b_all["v1_now"], prev)
        vel = stay  # already includes velocity; residual = stay - vel_persist
        velp = torch.nn.functional.normalize(
            2.0 * b_all["v1_now"] - prev, dim=-1)
        resid = (stay - velp).abs().mean(dim=0).cpu().numpy()
    loadings = {g: float(resid[m].mean()) if m.any() else float("nan")
                for g, m in nets.items()}
    fpn_dan = 0.5 * (loadings.get("FPN", 0.0) + loadings.get("DAN", 0.0))
    dmn_c = loadings.get("DMN", 0.0) / (fpn_dan + 1e-8)
    return {
        "sid": sid, "n": len(idx),
        "svals": s.tolist(),
        "C3": concentration(s, 3),
        "r_eff": effective_rank(s),
        "dmn_contamination": float(dmn_c),
        "loadings": loadings,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(J)))),
    }


def cohort_jspace(model, bundle, sids, qc, device, window=None) -> dict:
    rows = []
    for sid in sids:
        rec = subject_jspace(model, bundle, str(sid), device, window=window)
        if rec.get("n", 0) == 0:
            continue
        qrow = qc[qc["raw_id"].astype(str) == str(sid)]
        rec["dx_group"] = (qrow["dx_group"].iloc[0] if len(qrow) else None)
        rec["site"] = (qrow["site"].iloc[0] if len(qrow) else None)
        rows.append(rec)
    by = {}
    for grp in ("ADHD", "TDC"):
        sub = [r for r in rows if r.get("dx_group") == grp]
        if not sub:
            continue
        by[grp] = {
            "n": len(sub),
            "C3": float(np.mean([r["C3"] for r in sub])),
            "r_eff": float(np.mean([r["r_eff"] for r in sub])),
            "dmn_contamination": float(np.mean([r["dmn_contamination"] for r in sub])),
        }
    return {"n": len(rows), "per_subject": rows, "by_group": by}
