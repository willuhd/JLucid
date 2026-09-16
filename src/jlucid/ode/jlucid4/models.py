"""JLucid 4: identifiable A(p) stay field + frozen dest/when.

ε is not a knob. Invert is not in the field. Dest and when are discrete
and never update the stay Jacobian.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jlucid.config import (
    FLIP_COS_THRESH,
    HIDDEN_LAYERS,
    HIDDEN_SIZE,
    LATENT_DIM,
    STEP3_INPUT_DIM,
    STEP3_K_COND,
    WINDOW_TR,
)


def _mlp(in_dim: int, out_dim: int, hidden: int = HIDDEN_SIZE,
         layers: int = HIDDEN_LAYERS) -> nn.Sequential:
    dims = [in_dim] + [hidden] * layers + [out_dim]
    seq: list[nn.Module] = []
    for i in range(len(dims) - 1):
        seq.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            seq.append(nn.Tanh())
    return nn.Sequential(*seq)


def horizon_weights(n_steps: int, raw: list[float] | None = None) -> torch.Tensor:
    if raw is None:
        w = [1.0 if h == 0 else 2.0 for h in range(n_steps)]
    else:
        if len(raw) < n_steps:
            raw = list(raw) + [raw[-1]] * (n_steps - len(raw))
        w = list(raw[:n_steps])
    t = torch.tensor(w, dtype=torch.float32)
    return t / t.sum().clamp_min(1e-8)


def stay_loss(v_hat: torch.Tensor, v_next: torch.Tensor) -> torch.Tensor:
    return (1.0 - (v_hat * v_next).sum(dim=-1)).mean()


def stay_loss_weighted(v_hat: torch.Tensor, v_true: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
    cos = (v_hat * v_true).sum(dim=-1)
    w = weights.to(device=cos.device, dtype=cos.dtype).reshape(1, -1)
    return ((1.0 - cos) * w).sum(dim=-1).mean()


def stay_safe_penalty(v_hat: torch.Tensor, v_true: torch.Tensor,
                      v_now: torch.Tensor, margin: float = 0.002) -> torch.Tensor:
    pred = (v_hat[:, 0] * v_true[:, 0]).sum(dim=-1)
    persist = (v_now * v_true[:, 0]).sum(dim=-1)
    return torch.relu(persist - margin - pred).mean()


class WindowEncoder(nn.Module):
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


class StayAffineChord(nn.Module):
    """Switching-linear stay field; chord lives only in decode.

    ``z_{t+1} = z + A(p) z + b(p)``, ``A(p) = Σ_s p_s A_s``.
    ``∂f/∂z = A(p)`` exactly.
    """

    def __init__(self, latent: int = LATENT_DIM, k: int = STEP3_K_COND,
                 hidden: int = HIDDEN_SIZE, layers: int = HIDDEN_LAYERS,
                 in_dim: int = STEP3_INPUT_DIM, gamma_mode: str = "learned",
                 gamma_value: float = 0.5, decode_mode: str = "recursive",
                 small_decode_init: bool = True):
        super().__init__()
        if gamma_mode not in {"learned", "fixed", "none"}:
            raise ValueError(f"gamma_mode must be learned|fixed|none, got {gamma_mode}")
        if decode_mode not in {"recursive", "from_now"}:
            raise ValueError(f"decode_mode must be recursive|from_now, got {decode_mode}")
        self.latent = latent
        self.k = k
        self.eps = 0.0
        self.gamma_mode = gamma_mode
        self.gamma_value = float(gamma_value)
        self.decode_mode = decode_mode
        self.A = nn.Parameter(torch.zeros(k, latent, latent))
        self.b = nn.Parameter(torch.zeros(k, latent))
        self.decoder = _mlp(latent, in_dim, hidden=hidden, layers=layers)
        if small_decode_init:
            last = self.decoder[-1]
            nn.init.zeros_(last.bias)
            nn.init.normal_(last.weight, std=1e-3)
        self.gamma_head = nn.Linear(latent, 1)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)

    def mix_A(self, p: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bs,sij->bij", p, self.A)

    def mix_b(self, p: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bs,si->bi", p, self.b)

    def field(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """f(z, p) = A(p) z + b(p). Shape (B, d)."""
        return torch.einsum("bij,bj->bi", self.mix_A(p), z) + self.mix_b(p)

    def step(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return z + self.field(z, p)

    def gamma(self, z0: torch.Tensor) -> torch.Tensor:
        if self.gamma_mode == "none":
            return z0.new_zeros(z0.shape[0], 1)
        if self.gamma_mode == "fixed":
            return z0.new_full((z0.shape[0], 1), self.gamma_value)
        return torch.sigmoid(self.gamma_head(z0))

    def decode(self, z: torch.Tensor, anchor: torch.Tensor,
               extra: torch.Tensor | None = None) -> torch.Tensor:
        base = anchor if extra is None else F.normalize(anchor + extra, dim=-1)
        return F.normalize(base + self.decoder(z), dim=-1)

    def rollout(self, z0: torch.Tensor, p0: torch.Tensor, v1_now: torch.Tensor,
                n_steps: int = 1, decode_mode: str | None = None,
                last_disp: torch.Tensor | None = None) -> torch.Tensor:
        mode = decode_mode or self.decode_mode
        z = z0
        v = v1_now
        g = self.gamma(z0)
        prev = (v1_now - last_disp) if last_disp is not None else v1_now
        out = []
        for h in range(n_steps):
            z = self.step(z, p0)
            if mode == "recursive":
                extra = g * (v - prev) if self.gamma_mode != "none" else None
                v_new = self.decode(z, v, extra)
                prev, v = v, v_new
            else:
                extra = None
                if self.gamma_mode != "none" and last_disp is not None:
                    extra = g * float(h + 1) * last_disp
                v = self.decode(z, v1_now, extra)
            out.append(v)
        return torch.stack(out, dim=1)

    def field_jacobian(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.mix_A(p)

    def spectral_summary(self) -> list[dict]:
        A = self.A.detach().cpu().numpy()
        b = self.b.detach().cpu().numpy()
        rows = []
        eye = np.eye(self.latent, dtype=A.dtype)
        for s in range(self.k):
            As = A[s]
            svals = np.linalg.svd(As, compute_uv=False)
            eigs = np.linalg.eigvals(As)
            Ia = eye + As
            svals_ia = np.linalg.svd(Ia, compute_uv=False)
            eigs_ia = np.linalg.eigvals(Ia)
            rows.append({
                "state": int(s),
                "op_norm": float(svals[0]),
                "frob": float(np.linalg.norm(As)),
                "cond_A": float(svals[0] / max(float(svals[-1]), 1e-12)),
                "cond_IplusA": float(svals_ia[0] / max(float(svals_ia[-1]), 1e-12)),
                "max_real_eig": float(eigs.real.max()),
                "min_real_eig": float(eigs.real.min()),
                "spectral_radius": float(np.abs(eigs).max()),
                "IplusA_spectral_radius": float(np.abs(eigs_ia).max()),
                "b_norm": float(np.linalg.norm(b[s])),
                "svals": svals.astype(float).tolist(),
                "top3_svals": svals[:3].astype(float).tolist(),
                "top3_frob_frac": float(
                    np.sqrt((svals[:3] ** 2).sum()) / max(float(np.linalg.norm(As)), 1e-12)
                ),
                "eigs_real": eigs.real.astype(float).tolist(),
                "eigs_imag": eigs.imag.astype(float).tolist(),
            })
        return rows


class DestRouter(nn.Module):
    """Flip logit + next-state logits. Not a NODE. Frozen after pack."""

    def __init__(self, latent: int = LATENT_DIM, k: int = STEP3_K_COND,
                 hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(latent + 3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1 + k),
        )

    def extras(self, windows: torch.Tensor, v1_now: torch.Tensor,
               v2_now: torch.Tensor, gap_now: torch.Tensor) -> torch.Tensor:
        if windows.shape[1] >= 2:
            last = (windows[:, -1] * windows[:, -2]).sum(dim=-1, keepdim=True)
        else:
            last = torch.zeros(v1_now.shape[0], 1, device=v1_now.device,
                               dtype=v1_now.dtype)
        align = (v1_now * v2_now).sum(dim=-1, keepdim=True)
        gap = gap_now.reshape(-1, 1)
        return torch.cat([last, align, gap], dim=-1)

    def forward(self, z0, windows, v1_now, v2_now, gap_now) -> dict:
        feat = torch.cat(
            [z0, self.extras(windows, v1_now, v2_now, gap_now)], dim=-1)
        raw = self.net(feat)
        return {"flip_logit": raw[:, 0], "state_logits": raw[:, 1:]}


@dataclass
class SwitchConfig:
    in_dim: int = 27
    hidden: int = 32
    linear: bool = False
    k: int = STEP3_K_COND
    latent: int = LATENT_DIM

    def asdict(self) -> dict:
        return asdict(self)


class SwitchHead(nn.Module):
    """Binary switch logit from a fixed feature vector. Not a NODE."""

    def __init__(self, cfg: SwitchConfig | None = None, **kwargs):
        super().__init__()
        self.cfg = cfg or SwitchConfig(**kwargs)
        c = self.cfg
        if c.linear:
            self.net = nn.Linear(c.in_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(c.in_dim, c.hidden),
                nn.Tanh(),
                nn.Linear(c.hidden, 1),
            )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


class ThesisModel(nn.Module):
    """Stay field + its own dest encoder/router + when-head.

    Dest/when are frozen at train. Stay encoder is not the dest encoder.
    """

    def __init__(self, latent: int = LATENT_DIM, window: int = WINDOW_TR,
                 k: int = STEP3_K_COND, hidden: int = HIDDEN_SIZE,
                 gamma_mode: str = "learned", gamma_value: float = 0.5,
                 decode_mode: str = "recursive",
                 small_decode_init: bool = True,
                 switch_hidden: int = 32, switch_in_dim: int = 27):
        super().__init__()
        self.latent = latent
        self.window = window
        self.k = k
        self.eps = 0.0
        self.gamma_mode = gamma_mode
        self.decode_mode = decode_mode
        self.encoder = WindowEncoder(latent=latent, hidden=hidden)
        self.stay = StayAffineChord(
            latent=latent, k=k, hidden=hidden, gamma_mode=gamma_mode,
            gamma_value=gamma_value, decode_mode=decode_mode,
            small_decode_init=small_decode_init)
        self.dest_encoder = WindowEncoder(latent=latent, hidden=hidden)
        self.dest_router = DestRouter(latent=latent, k=k, hidden=hidden)
        self.when = SwitchHead(SwitchConfig(
            in_dim=switch_in_dim, hidden=switch_hidden, latent=latent, k=k))
        self.register_buffer("switch_mean", torch.zeros(switch_in_dim))
        self.register_buffer("switch_std", torch.ones(switch_in_dim))
        self.switch_tau = 0.5
        self.centroids = None
        self._discrete_packed = False

    def stay_params(self):
        return list(self.encoder.parameters()) + list(self.stay.parameters())

    def freeze_discrete(self) -> None:
        for p in list(self.dest_encoder.parameters()) + list(
                self.dest_router.parameters()) + list(self.when.parameters()):
            p.requires_grad = False

    def forward(self, windows, p0, v1_now, v2_now, gap_now,
                n_steps: int = 1, decode_mode: str | None = None) -> dict:
        z0 = self.encoder(windows)
        last_disp = None
        if windows.shape[1] >= 2:
            last_disp = windows[:, -1] - windows[:, -2]
        v_stay = self.stay.rollout(
            z0, p0, v1_now, n_steps=n_steps, decode_mode=decode_mode,
            last_disp=last_disp)
        dest = self.dest_route(windows, p0, v1_now, v2_now, gap_now)
        return {
            "z0": z0,
            "v_stay": v_stay,
            "gamma": self.stay.gamma(z0),
            **dest,
        }

    def dest_route(self, windows, p0, v1_now, v2_now, gap_now) -> dict:
        z = self.dest_encoder(windows)
        route = self.dest_router(z, windows, v1_now, v2_now, gap_now)
        return {"dest_z0": z, **route}


def flip_target(v1_now: torch.Tensor, v1_next: torch.Tensor,
                thresh: float = FLIP_COS_THRESH) -> torch.Tensor:
    return (v1_now * v1_next).sum(dim=-1) < thresh


def autograd_field_jacobian(field_fn, z: torch.Tensor, p: torch.Tensor
                            ) -> torch.Tensor:
    rows = []
    for i in range(z.shape[0]):
        zi = z[i].detach()
        pi = p[i:i + 1]
        ji = torch.autograd.functional.jacobian(
            lambda u, _p=pi: field_fn(u.unsqueeze(0), _p).squeeze(0),
            zi,
        )
        rows.append(ji)
    return torch.stack(rows, dim=0)
