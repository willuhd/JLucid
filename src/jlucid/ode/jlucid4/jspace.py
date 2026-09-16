"""Brain J-space downstream of the frozen JLucid 4 stay field.

Primary Jacobian is the closed-form stay-field Jacobian

    J_dyn(t) = A(p(t)) = Σ_s p_s A_s     (12 × 12, ε = 0)

never an invert residual and never dest/when or the γ-chord decode.
Subject-level ``J̄ = mean_t A(p)`` over stay frames; Brain J-space axes
are the leading **right** singular vectors (latent z, 12-d):

    J̄ = U Σ Vᵀ ,   z_J = V_kᵀ z ,   A_J = V_kᵀ J̄ V_k

This is a model-based Jacobian subspace related to, not the same as,
Anthropic's J-space. Carrier: JLucid 4 ``A(p)``.

Pure array helpers live at the top so tests can plant spectra and ``V_k``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jlucid.config import (
    JSPACE_DMN_EPS,
    JSPACE_HURWITZ_MARGIN,
    JSPACE_K,
    N_PERMUTATIONS,
    SEED,
    SHOT_BATCH,
    WINDOW_TR,
    YEO_NETWORKS,
)
from jlucid.control.controllability import (
    average_controllability,
    modal_controllability,
    network_masks,
)
from jlucid.control.energy import solve_lqr_tracking
from jlucid.control.stabilize import hurwitz_shift
from jlucid.ode.jlucid4.models import ThesisModel, autograd_field_jacobian
from jlucid.ode.jlucid4.shots import collate_shots, stay_index
from jlucid.stats.effects import cohens_d
from jlucid.stats.permutation import fdr_bh, two_sample_perm_test_residual


STATE_NAMES = {0: "VIS-DAN", 1: "DMN-LIMBIC", 2: "SMN-VAN"}
H3_METRICS = ("C_k", "r_eff", "dmn_contamination")


# ---------------------------------------------------------------------------
# Spectral / SVD (array-only)
# ---------------------------------------------------------------------------

def concentration(svals: np.ndarray, k: int = JSPACE_K) -> float:
    """Top-k singular-value concentration C_k = (Σ_{i=1}^k σ_i²) / (Σ σ_i²)."""
    s = np.asarray(svals, dtype=float).reshape(-1)
    s2 = s ** 2
    tot = float(s2.sum())
    if tot <= 0.0:
        return float("nan")
    kk = min(max(int(k), 0), s2.size)
    return float(s2[:kk].sum() / tot)


def effective_rank(svals: np.ndarray) -> float:
    """r_eff = (Σ σ)² / (Σ σ²).  1 if rank-1; d if a flat spectrum."""
    s = np.asarray(svals, dtype=float).reshape(-1)
    den = float((s ** 2).sum())
    if den <= 0.0:
        return float("nan")
    return float((s.sum() ** 2) / den)


def svd_right(J: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD with the right-vector convention used for J-space.

    Returns ``(U, s, V)`` such that ``J = U @ diag(s) @ V.T`` and the
    columns of ``V`` are the right singular vectors (latent z).
    ``z_J = V_k.T @ z``.  Works for square or rectangular J.
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2:
        raise ValueError(f"J must be 2-d, got {J.shape}")
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    return U, s, Vt.T


def jspace_coords(z: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    """z_J = V_kᵀ z.  ``z`` is ``(..., d)``, ``V_k`` is ``(d, k)``."""
    z = np.asarray(z, dtype=float)
    V_k = np.asarray(V_k, dtype=float)
    return np.einsum("...d,dk->...k", z, V_k)


def project_jacobian(J: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    """A_J = V_kᵀ J V_k."""
    J = np.asarray(J, dtype=float)
    V_k = np.asarray(V_k, dtype=float)
    return V_k.T @ J @ V_k


def mix_A(A_states: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Closed-form A(p) = Σ_s p_s A_s.  ``A_states`` (k,d,d), ``p`` (...,k)."""
    A_states = np.asarray(A_states, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.einsum("...s,sij->...ij", p, A_states)


# ---------------------------------------------------------------------------
# Ablation maps (array-only)
# ---------------------------------------------------------------------------

def _rows(z: np.ndarray) -> tuple[np.ndarray, bool]:
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        return z.reshape(1, -1), True
    if z.ndim != 2:
        raise ValueError(f"z must be (d,) or (n, d), got {z.shape}")
    return z, False


def apply_zero(z: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    """z − V_k V_kᵀ z  (zero the J-space coordinates)."""
    z, squeeze = _rows(z)
    V_k = np.asarray(V_k, dtype=float)
    out = z - (z @ V_k) @ V_k.T
    return out[0] if squeeze else out


def apply_ablate(z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Zero coordinates along an arbitrary orthonormal frame Q (d, k)."""
    return apply_zero(z, Q)


def apply_swap(z: np.ndarray, V_k: np.ndarray, z_J_donor: np.ndarray) -> np.ndarray:
    """Replace J-space coords with ``z_J_donor``; keep the orthogonal complement.

    z_new = V_k z_J_donor + (I − V_k V_kᵀ) z
    """
    z, squeeze = _rows(z)
    V_k = np.asarray(V_k, dtype=float)
    donor = np.asarray(z_J_donor, dtype=float)
    if donor.ndim == 1:
        donor = np.broadcast_to(donor, (z.shape[0], V_k.shape[1]))
    elif donor.shape != (z.shape[0], V_k.shape[1]):
        raise ValueError(
            f"z_J_donor shape {donor.shape} incompatible with z {z.shape} "
            f"and V_k {V_k.shape}"
        )
    orth = z - (z @ V_k) @ V_k.T
    out = orth + donor @ V_k.T
    return out[0] if squeeze else out


def random_orthonormal(d: int, k: int, seed: int = SEED) -> np.ndarray:
    """Random orthonormal frame in R^d of dimension k (QR of a Gaussian)."""
    rng = np.random.default_rng(int(seed))
    q, _ = np.linalg.qr(rng.normal(size=(int(d), int(k))))
    return q[:, : int(k)]


def swap_donors(n: int, seed: int = SEED) -> np.ndarray:
    """Derangement-like pairing: a permutation with no fixed points if n>1."""
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(n))
    if n > 1:
        fixed = np.where(order == np.arange(n))[0]
        for i in fixed:
            j = (int(i) + 1) % n
            order[i], order[j] = order[j], order[i]
    return order.astype(int)


# ---------------------------------------------------------------------------
# Decoder-mapped network loadings
# ---------------------------------------------------------------------------

def map_axes_decoder(R: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    """w_k = R v_k.  ``R`` is (n_roi, d), ``V_k`` is (d, k) → (n_roi, k)."""
    return np.asarray(R, dtype=float) @ np.asarray(V_k, dtype=float)


def pattern_network_loadings(
    pattern: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    """L_net = mean |pattern| on the network's ROIs."""
    pat = np.abs(np.asarray(pattern, dtype=float).reshape(-1))
    out: dict[str, float] = {}
    for g, m in masks.items():
        m = np.asarray(m, dtype=bool)
        out[g] = float(pat[m].mean()) if m.any() else float("nan")
    return out


def dmn_contamination(
    loadings: dict[str, float],
    eps: float = JSPACE_DMN_EPS,
) -> float:
    """L_DMN / (L_FPN/DAN + ε) with L_FPN/DAN = mean(L_FPN, L_DAN)."""
    dmn = float(loadings.get("DMN", float("nan")))
    fpn = float(loadings.get("FPN", float("nan")))
    dan = float(loadings.get("DAN", float("nan")))
    fpn_dan = 0.5 * (fpn + dan)
    return float(dmn / (fpn_dan + float(eps)))


def decoder_jacobian(decoder: torch.nn.Module, z: np.ndarray) -> np.ndarray:
    """Jacobian of Dec(z) at a single latent point. Shape (n_roi, d)."""
    z_t = torch.as_tensor(np.asarray(z, dtype=np.float32).reshape(-1))
    jac = torch.autograd.functional.jacobian(decoder, z_t)
    return jac.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Subject assembly from planted / precomputed arrays
# ---------------------------------------------------------------------------

def assemble_subject(
    J_bar: np.ndarray,
    *,
    R: np.ndarray | None = None,
    masks: dict[str, np.ndarray] | None = None,
    k: int = JSPACE_K,
    n_frames: int = 0,
) -> dict:
    """SVD, C_k, r_eff, decoder-mapped loadings from a subject ``J̄``."""
    J_bar = np.asarray(J_bar, dtype=float)
    U, s, V = svd_right(J_bar)
    d = J_bar.shape[1]
    kk = min(int(k), V.shape[1])
    V_k = V[:, :kk]
    rec: dict = {
        "J": J_bar,
        "U": U,
        "svals": s,
        "V": V,
        "V_k": V_k,
        "C_k": concentration(s, kk),
        "r_eff": effective_rank(s),
        "n": int(n_frames),
        "latent": int(d),
        "k": int(kk),
        "A_J": project_jacobian(J_bar, V_k),
    }
    if R is not None and masks is not None:
        W = map_axes_decoder(R, V_k)
        pattern = np.abs(W).mean(axis=1)
        loadings = pattern_network_loadings(pattern, masks)
        rec["loadings"] = loadings
        rec["dmn_contamination"] = dmn_contamination(loadings)
        rec["W"] = W
    else:
        rec["loadings"] = {}
        rec["dmn_contamination"] = float("nan")
    return rec


# ---------------------------------------------------------------------------
# Covariates + H3
# ---------------------------------------------------------------------------

def covariates_from_table(df: pd.DataFrame) -> pd.DataFrame:
    """Step-2 covariate class: age, sex, site dummies, mean FD."""
    motion = df["mean_fd"] if "mean_fd" in df.columns else df["max_motion_mm"]
    sex = df["gender"] if "gender" in df.columns else df["sex"]
    cov = pd.DataFrame({
        "age": pd.to_numeric(df["age"], errors="coerce"),
        "sex": pd.to_numeric(sex, errors="coerce"),
        "motion": pd.to_numeric(motion, errors="coerce"),
        "site": df["site"].astype(str),
    })
    return pd.get_dummies(cov, columns=["site"], drop_first=False)


def h3_group_stats(
    metrics: pd.DataFrame,
    *,
    n_perm: int = N_PERMUTATIONS,
    seed: int = SEED,
    metric_names: tuple[str, ...] = H3_METRICS,
) -> pd.DataFrame:
    """ADHD vs TDC residualized permutation + FDR on J-space metrics.

    ``metrics`` must have ``dx_group`` plus the named metric columns and
    the same covariate columns as :func:`covariates_from_table` (age/sex
    or gender, site, mean_fd).  A null result is an honest pass.
    """
    cov = covariates_from_table(metrics)
    rows = []
    for name in metric_names:
        if name not in metrics.columns:
            continue
        work = metrics[["dx_group", name]].copy()
        work["value"] = pd.to_numeric(work[name], errors="coerce")
        work = work.dropna(subset=["value"])
        adhd = work[work.dx_group == "ADHD"]
        tdc = work[work.dx_group == "TDC"]
        empty = {
            "label": f"H3_{name}",
            "metric": name,
            "n_adhd": int(len(adhd)),
            "n_tdc": int(len(tdc)),
            "observed": float("nan"),
            "p_raw": float("nan"),
            "p_fdr": float("nan"),
            "cohens_d": float("nan"),
            "cohens_d_raw": float("nan"),
            "hypothesis": "H3",
        }
        if len(adhd) < 3 or len(tdc) < 3:
            rows.append(empty)
            continue
        grouped = pd.concat([adhd, tdc])
        ccov = cov.loc[grouped.index].reset_index(drop=True)
        x = adhd["value"].to_numpy()
        y = tdc["value"].to_numpy()
        res = two_sample_perm_test_residual(
            x, y, ccov, n_perm=n_perm, seed=seed)
        rows.append({
            "label": f"H3_{name}",
            "metric": name,
            "n_adhd": int(len(x)),
            "n_tdc": int(len(y)),
            "observed": float(res["observed"]),
            "p_raw": float(res["p_value"]),
            "p_fdr": float("nan"),
            "cohens_d": float(cohens_d(res["resid_x"], res["resid_y"])),
            "cohens_d_raw": float(cohens_d(x, y)),
            "hypothesis": "H3",
        })
    out = pd.DataFrame(rows)
    if len(out) and out["p_raw"].notna().any():
        p = out["p_raw"].to_numpy(dtype=float)
        q = np.full_like(p, np.nan, dtype=float)
        ok = np.isfinite(p)
        if ok.any():
            q[ok] = fdr_bh(p[ok])
        out["p_fdr"] = q
    return out


# ---------------------------------------------------------------------------
# Controllability / latent energy
# ---------------------------------------------------------------------------

def control_from_AJ(
    A_J: np.ndarray,
    margin: float = JSPACE_HURWITZ_MARGIN,
) -> dict:
    """Hurwitz-shift A_J (not FC stabilize) and avg/modal controllability."""
    A_h = hurwitz_shift(A_J, margin=margin)
    phi = average_controllability(A_h)
    psi = modal_controllability(A_h)
    return {
        "A_J": np.asarray(A_J, dtype=float),
        "A_J_hurwitz": A_h,
        "avg": np.asarray(phi, dtype=float),
        "modal": np.asarray(psi, dtype=float),
        "avg_mean": float(phi.mean()),
        "modal_mean": float(psi.mean()),
        "max_real_eig": float(np.linalg.eigvals(A_h).real.max()),
    }


def latent_pair_energy(
    A_h: np.ndarray,
    z_from: np.ndarray,
    z_to: np.ndarray,
) -> float:
    """LQR energy between two J-space state means (S = I)."""
    x0 = np.asarray(z_from, dtype=float).reshape(-1)
    x_t = np.asarray(z_to, dtype=float).reshape(-1)
    s = np.eye(x0.shape[0])
    sol = solve_lqr_tracking(A_h, x0, x_t, s)
    return float(sol["energy_total"])


# ---------------------------------------------------------------------------
# Stay-path cosine (ablation diagnostic; not mixed next-V1)
# ---------------------------------------------------------------------------

def stay_path_cosines(
    stay: torch.nn.Module,
    z0: torch.Tensor,
    p0: torch.Tensor,
    v1_now: torch.Tensor,
    last_disp: torch.Tensor,
    v1_next: torch.Tensor,
) -> np.ndarray:
    """One-step in-dwell stay cosine. Differentiates neither dest nor when."""
    z1 = stay.step(z0, p0)
    g = stay.gamma(z0)
    extra = g * last_disp if stay.gamma_mode != "none" else None
    hat = stay.decode(z1, v1_now, extra)
    cos = (hat * v1_next).sum(dim=-1)
    return cos.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Competence gate (official checkpoint + locked val IDs)
# ---------------------------------------------------------------------------

def competence_gate(
    model: ThesisModel,
    bundle: dict,
    val_ids: list[str],
    device,
    *,
    n_boot: int = 2000,
    cond_max: float = 10.0,
) -> dict:
    """Stay-safe Δ=1 + J=A(p) + sane cond(I+A). Fail closed if any check fails."""
    from jlucid.ode.jlucid4.eval import evaluate
    from jlucid.ode.jlucid4.jacobians import jacobian_smoke, spectral_report

    ev = evaluate(
        model, bundle, val_ids, device, horizons=(1,), n_boot=n_boot)
    h1 = ev["horizons"]["1"]
    js = jacobian_smoke(model, bundle, val_ids, device)
    spec = spectral_report(model)
    reasons: list[str] = []
    gain = h1.get("gain_subject")
    vs_half = h1.get("gain_vs_half_subject")
    if gain is None or not (gain > 0):
        reasons.append(f"Δ=1 does not beat persist (gain={gain})")
    if vs_half is None or not (vs_half > 0):
        reasons.append(f"Δ=1 does not beat half-step (vs_half={vs_half})")
    if not h1.get("gain_excludes_0"):
        reasons.append(f"95% CI vs persist does not exclude 0 ({h1.get('gain_subject_ci95')})")
    if not js.get("ok") or not js.get("matches_closed_form"):
        reasons.append(f"J≠A(p) or smoke failed: {js}")
    cond = float(spec.get("max_cond_IplusA", float("inf")))
    if not spec.get("sane_linearization") or cond >= cond_max:
        reasons.append(f"cond(I+A) not sane: {cond}")
    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "delta1": h1,
        "jacobian_smoke": js,
        "spectral": {
            "max_cond_IplusA": cond,
            "max_op_norm": spec.get("max_op_norm"),
            "sane_linearization": spec.get("sane_linearization"),
            "jacobian_is": spec.get("jacobian_is"),
        },
        "n_val": int(len(val_ids)),
        "carrier": "JLucid 4 A(p)",
    }


def sample_j_equals_ap(
    model: ThesisModel,
    bundle: dict,
    sids: list[str],
    device,
    *,
    n: int = 8,
) -> dict:
    """Assert closed-form A(p) equals autograd J at sampled stay frames."""
    idx = stay_index({s: bundle[s] for s in sids}, int(model.window), 1)
    if not idx:
        return {"ok": False, "n": 0, "max_abs_diff": float("nan")}
    b = collate_shots(bundle, idx[:n], int(model.window), 1, device)
    z0 = model.encoder(b["windows"])
    p0 = b["p0"]
    J_auto = autograd_field_jacobian(model.stay.field, z0, p0)
    J_closed = model.stay.field_jacobian(z0, p0)
    A_np = mix_A(
        model.stay.A.detach().cpu().numpy(),
        p0.detach().cpu().numpy(),
    )
    J_auto = J_auto.detach()
    J_closed = J_closed.detach()
    match_auto = bool(torch.allclose(J_auto, J_closed, atol=1e-5, rtol=1e-4))
    match_np = bool(np.allclose(
        J_closed.cpu().numpy(), A_np, atol=1e-5, rtol=1e-4))
    finite = bool(torch.isfinite(J_closed).all())
    diff = float((J_auto - J_closed).abs().max())
    return {
        "ok": bool(match_auto and match_np and finite),
        "n": int(z0.shape[0]),
        "max_abs_diff": diff,
        "matches_closed_form": match_auto,
        "matches_mix_A": match_np,
        "finite": finite,
        "shape": list(J_closed[0].shape),
    }


# ---------------------------------------------------------------------------
# Per-subject pass over the frozen stay field
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_stay_frames(
    model: ThesisModel,
    bundle: dict,
    sid: str,
    device,
    *,
    window: int | None = None,
    batch: int = SHOT_BATCH,
) -> dict | None:
    """Encode stay frames and closed-form J=A(p) for one subject."""
    window = int(window or getattr(model, "window", WINDOW_TR))
    rec = bundle[sid]
    idx = stay_index({sid: rec}, window, 1)
    if not idx:
        return None
    zs, ps, Js, labs = [], [], [], []
    nows, nxts, disps = [], [], []
    model.eval()
    for s0 in range(0, len(idx), batch):
        bi = idx[s0:s0 + batch]
        b = collate_shots(bundle, bi, window, 1, device)
        z0 = model.encoder(b["windows"])
        p0 = b["p0"]
        J = model.stay.field_jacobian(z0, p0)
        zs.append(z0.detach().cpu().numpy())
        ps.append(p0.detach().cpu().numpy())
        Js.append(J.detach().cpu().numpy())
        labs.append(b["labels_now"].detach().cpu().numpy())
        nows.append(b["v1_now"].detach().cpu().numpy())
        nxts.append(b["v1_future"][:, 0].detach().cpu().numpy())
        if b["windows"].shape[1] >= 2:
            disps.append((b["windows"][:, -1] - b["windows"][:, -2]).cpu().numpy())
        else:
            disps.append(np.zeros_like(b["v1_now"].cpu().numpy()))
    z = np.concatenate(zs, axis=0)
    p = np.concatenate(ps, axis=0)
    J = np.concatenate(Js, axis=0)
    labels = np.concatenate(labs, axis=0).astype(int)
    return {
        "sid": sid,
        "z": z.astype(np.float64),
        "p": p.astype(np.float64),
        "J": J.astype(np.float64),
        "labels": labels,
        "v1_now": np.concatenate(nows, axis=0).astype(np.float32),
        "v1_next": np.concatenate(nxts, axis=0).astype(np.float32),
        "last_disp": np.concatenate(disps, axis=0).astype(np.float32),
        "n": int(z.shape[0]),
    }


def subject_from_frames(
    model: ThesisModel,
    frames: dict,
    masks: dict[str, np.ndarray],
    *,
    k: int = JSPACE_K,
    n_states: int = 3,
) -> dict:
    """Build one subject's J-space record from :func:`collect_stay_frames`."""
    J_bar = frames["J"].mean(axis=0)
    z_mean = frames["z"].mean(axis=0)
    R = decoder_jacobian(model.stay.decoder, z_mean)
    rec = assemble_subject(
        J_bar, R=R, masks=masks, k=k, n_frames=frames["n"])
    rec["sid"] = frames["sid"]
    rec["z_mean"] = z_mean
    rec["R"] = R
    d = z_mean.shape[0]
    z_state = np.full((n_states, d), np.nan, dtype=float)
    n_state = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        sel = frames["labels"] == s
        n_state[s] = int(sel.sum())
        if sel.any():
            z_state[s] = frames["z"][sel].mean(axis=0)
    rec["z_state"] = z_state
    rec["n_state"] = n_state
    return rec


def state_transition_pairs(n_states: int = 3) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_states) for j in range(n_states) if i != j]


def official_checkpoint_ok(path: Path) -> tuple[bool, str]:
    """True iff ``path`` exists and is not the pack-only ode39 archive."""
    p = Path(path)
    parts = p.parts + (p.resolve().parts if p.exists() else ())
    if "packonly_ode39" in parts:
        return False, f"pack-only ode39 archive is not the carrier: {p}"
    if not p.exists():
        return False, f"missing checkpoint {p}"
    return True, str(p.resolve())
