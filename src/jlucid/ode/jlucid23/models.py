"""JLucid 2.3: stay velocity field + hard invert gate.

Stay law (causal, trained only on dwells)::

    v_prev = window[:, -2]
    v_vel  = unit(2 v - v_prev)          # last-step velocity persist
    z_dot  = f_θ(z, p)                   # 12-d residual, identifiable J
    z1     = z + z_dot
    v_stay = unit(v_vel + tan(Dec(z1)))  # residual on the velocity prior

Flip action is invert (−v), never a learned jump (2.2 leaked that into
stays). The gate is hard at eval.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jlucid.config import (
    FLIP_COS_THRESH,
    HIDDEN_SIZE,
    LATENT_DIM,
    PROCESSED_DIR,
    STEP3_INPUT_DIM,
    STEP3_K_COND,
    WINDOW_TR,
)


def _mlp(in_dim: int, out_dim: int, hidden: int = HIDDEN_SIZE,
         layers: int = 2) -> nn.Sequential:
    dims = [in_dim] + [hidden] * layers + [out_dim]
    seq: list[nn.Module] = []
    for i in range(len(dims) - 1):
        seq.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            seq.append(nn.Tanh())
    return nn.Sequential(*seq)


def tangent(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project ``u`` into the tangent plane of unit ``v``."""
    return u - (u * v).sum(dim=-1, keepdim=True) * v


def velocity_persist(v_now: torch.Tensor, v_prev: torch.Tensor) -> torch.Tensor:
    """unit(2 v_now − v_prev): continue the last observed displacement."""
    return F.normalize(2.0 * v_now - v_prev, dim=-1)


class WindowEncoder(nn.Module):
    """Deterministic GRU; same layout as JLucid 2 so we can load μ-head weights."""

    def __init__(self, in_dim: int = STEP3_INPUT_DIM, latent: int = LATENT_DIM,
                 hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.latent = latent
        self.gru = nn.GRU(in_dim, hidden, batch_first=False, num_layers=1)
        self.head = nn.Linear(hidden, latent)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(windows.transpose(0, 1))
        return self.head(h[-1])

    def load_j2_encoder(self, state_dict: dict) -> None:
        gru = {k[len("encoder.gru."):]: v for k, v in state_dict.items()
               if k.startswith("encoder.gru.")}
        self.gru.load_state_dict(gru)
        w = state_dict["encoder.head.weight"][: self.latent]
        b = state_dict["encoder.head.bias"][: self.latent]
        with torch.no_grad():
            self.head.weight.copy_(w)
            self.head.bias.copy_(b)


class StayField(nn.Module):
    """Identifies J_dyn = ∂f/∂z of a stay-only Euler step."""

    def __init__(self, latent: int = LATENT_DIM, k: int = STEP3_K_COND,
                 hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.net = _mlp(latent + k, latent, hidden=hidden, layers=2)

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, p], dim=-1))


class StaySplitModel(nn.Module):
    def __init__(self, latent: int = LATENT_DIM, window: int = WINDOW_TR,
                 in_dim: int = STEP3_INPUT_DIM, hidden: int = HIDDEN_SIZE,
                 k: int = STEP3_K_COND):
        super().__init__()
        self.latent = latent
        self.window = window
        self.k = k
        self.encoder = WindowEncoder(in_dim=in_dim, latent=latent, hidden=hidden)
        self.f_stay = StayField(latent=latent, k=k, hidden=hidden)
        self.stay_dec = _mlp(latent, in_dim, hidden=hidden, layers=2)
        # extras: last_cos, centroid margin → high-precision flip score
        self.flip_head = nn.Sequential(
            nn.Linear(latent + 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        cents = np.load(PROCESSED_DIR / "state_centroids_k3.npy").astype(np.float32)
        cents = cents / np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12)
        self.register_buffer("cents", torch.from_numpy(cents))

    def extras(self, windows: torch.Tensor, v_now: torch.Tensor) -> torch.Tensor:
        if windows.shape[1] >= 2:
            last = (windows[:, -1] * windows[:, -2]).sum(dim=-1, keepdim=True)
        else:
            last = torch.ones(v_now.shape[0], 1, device=v_now.device, dtype=v_now.dtype)
        dots = v_now @ self.cents.T
        top2 = torch.topk(dots, k=min(2, self.cents.shape[0]), dim=-1).values
        margin = (top2[:, 0] - top2[:, -1]).unsqueeze(-1)
        return torch.cat([last, margin], dim=-1)

    def encode(self, windows: torch.Tensor) -> torch.Tensor:
        return self.encoder(windows)

    def stay_step(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """One Euler step of the stay field. J_dyn = ∂this/∂z − I."""
        return z + self.f_stay(z, p)

    def decode_stay(self, z1: torch.Tensor, v_now: torch.Tensor,
                    v_prev: torch.Tensor) -> torch.Tensor:
        # Residual on persist, not on velocity. Last-step velocity overshoots
        # on this atlas (val stay 0.915 vs persist 0.918); pulling toward it
        # was how 2.3 v1 taxed the one thing we must not tax.
        raw = self.stay_dec(z1)
        return F.normalize(v_now + tangent(raw, v_now), dim=-1)

    def flip_logit(self, z0: torch.Tensor, windows: torch.Tensor,
                   v_now: torch.Tensor) -> torch.Tensor:
        return self.flip_head(torch.cat([z0, self.extras(windows, v_now)], dim=-1)).squeeze(-1)

    def v_prev(self, windows: torch.Tensor, v_now: torch.Tensor) -> torch.Tensor:
        if windows.shape[1] >= 2:
            return windows[:, -2]
        return v_now

    def forward(self, windows: torch.Tensor, v_now: torch.Tensor, p0: torch.Tensor,
                hard_tau: float | None = None,
                last_cos_max: float | None = None) -> dict:
        z0 = self.encode(windows)
        z1 = self.stay_step(z0, p0)
        prev = self.v_prev(windows, v_now)
        stay = self.decode_stay(z1, v_now, prev)
        vel = velocity_persist(v_now, prev)
        logit = self.flip_logit(z0, windows, v_now)
        g = torch.sigmoid(logit)
        invert = -v_now
        if hard_tau is None:
            mix = (1.0 - g.unsqueeze(-1)) * stay + g.unsqueeze(-1) * invert
            vhat = F.normalize(mix, dim=-1)
            called = g >= 0.5
        else:
            called = g >= hard_tau
            if last_cos_max is not None and windows.shape[1] >= 2:
                last = (windows[:, -1] * windows[:, -2]).sum(dim=-1)
                called = called & (last < last_cos_max)
            on = called.to(stay.dtype).unsqueeze(-1)
            vhat = F.normalize((1.0 - on) * stay + on * invert, dim=-1)
        return {
            "z0": z0, "z1": z1, "z_dot": z1 - z0,
            "stay": stay, "vel": vel, "invert": invert,
            "logit": logit, "g": g, "called": called, "v1_hat": vhat,
        }


def flip_target(v_now: torch.Tensor, v_next: torch.Tensor,
                thresh: float = FLIP_COS_THRESH) -> torch.Tensor:
    return (v_now * v_next).sum(dim=-1) < thresh


def stay_mask(v_now: torch.Tensor, v_next: torch.Tensor,
              lab0: torch.Tensor, lab1: torch.Tensor,
              thresh: float = FLIP_COS_THRESH) -> torch.Tensor:
    """True dwell: same LEiDA label and not an antipodal flip."""
    return (lab0 == lab1) & ~flip_target(v_now, v_next, thresh)


def split_loss(out: dict, v_next: torch.Tensor, v_now: torch.Tensor,
               lab0: torch.Tensor, lab1: torch.Tensor,
               w_bce: float = 1.0, w_pull: float = 0.35,
               thresh: float = FLIP_COS_THRESH) -> dict:
    """Stay residual only on dwells; flip head on every shot; no soft invert."""
    flip = flip_target(v_now, v_next, thresh).float()
    stay_m = stay_mask(v_now, v_next, lab0, lab1, thresh).float()
    n_pos = flip.sum().clamp(min=1.0)
    n_neg = (1.0 - flip).sum().clamp(min=1.0)
    n_stay = stay_m.sum().clamp(min=1.0)
    pos_weight = (n_neg / n_pos).detach()
    bce = F.binary_cross_entropy_with_logits(out["logit"], flip, pos_weight=pos_weight)
    stay_err = (1.0 - (out["stay"] * v_next).sum(dim=-1))
    vel_err = (1.0 - (out["vel"] * v_next).sum(dim=-1))
    stay_fit = (stay_err * stay_m).sum() / n_stay
    # persist pull: residual may only move a dwell if it helps
    pull = ((1.0 - (out["stay"] * v_now).sum(dim=-1)) * stay_m).sum() / n_stay
    total = stay_fit + w_pull * pull + w_bce * bce
    hard = torch.where(out["g"].unsqueeze(-1) >= 0.5, out["invert"], out["stay"])
    hard = F.normalize(hard, dim=-1)
    cos = (hard * v_next).sum(dim=-1)
    return {
        "total": total, "stay_fit": stay_fit, "pull": pull, "bce": bce,
        "vel_fit": (vel_err * stay_m).sum() / n_stay,
        "cos1": cos.mean(),
        "g_flip": (out["g"] * flip).sum() / n_pos,
        "g_stay": (out["g"] * stay_m).sum() / n_stay,
        "n_stay": n_stay, "n_flip": n_pos,
    }
