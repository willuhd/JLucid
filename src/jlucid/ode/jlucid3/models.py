"""JLucid 3: stay-only residual NODE + discrete flip/next-state router."""

from __future__ import annotations

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


class StayField(nn.Module):
    """f(z, p) Euler residual on the sphere. Train on dwell pairs only."""

    def __init__(self, latent: int = LATENT_DIM, k: int = STEP3_K_COND,
                 hidden: int = HIDDEN_SIZE, layers: int = HIDDEN_LAYERS,
                 in_dim: int = STEP3_INPUT_DIM):
        super().__init__()
        self.dynamics = _mlp(latent + k, latent, hidden=hidden, layers=layers)
        self.decoder = _mlp(latent, in_dim, hidden=hidden, layers=layers)

    def step(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return z + self.dynamics(torch.cat([z, p], dim=-1))

    def decode(self, z: torch.Tensor, v1_now: torch.Tensor) -> torch.Tensor:
        raw = self.decoder(z)
        return F.normalize(v1_now + raw, dim=-1)

    def rollout(self, z0: torch.Tensor, p0: torch.Tensor, v1_now: torch.Tensor,
                n_steps: int = 1) -> torch.Tensor:
        z = z0
        out = []
        for _ in range(n_steps):
            z = self.step(z, p0)
            out.append(self.decode(z, v1_now))
        return torch.stack(out, dim=1)  # (B, H, 90)


class DiscreteRouter(nn.Module):
    """Flip logit + next-state logits. Not a NODE.

    Features: z0, eigen-gap, V1·V2, last-step cosine.
    """

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


class ThesisModel(nn.Module):
    def __init__(self, latent: int = LATENT_DIM, window: int = WINDOW_TR,
                 k: int = STEP3_K_COND, hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.latent = latent
        self.window = window
        self.k = k
        self.encoder = WindowEncoder(latent=latent, hidden=hidden)
        self.stay = StayField(latent=latent, k=k, hidden=hidden)
        self.router = DiscreteRouter(latent=latent, k=k, hidden=hidden)

    def forward(self, windows, p0, v1_now, v2_now, gap_now,
                n_steps: int = 1) -> dict:
        z0 = self.encoder(windows)
        v_stay = self.stay.rollout(z0, p0, v1_now, n_steps=n_steps)
        route = self.router(z0, windows, v1_now, v2_now, gap_now)
        return {"z0": z0, "v_stay": v_stay, **route}


def flip_target(v1_now: torch.Tensor, v1_next: torch.Tensor,
                thresh: float = FLIP_COS_THRESH) -> torch.Tensor:
    return (v1_now * v1_next).sum(dim=-1) < thresh


def stay_loss(v_hat: torch.Tensor, v_next: torch.Tensor) -> torch.Tensor:
    return (1.0 - (v_hat * v_next).sum(dim=-1)).mean()


def router_loss(out: dict, v1_now, v1_next, lab_next,
                thresh: float = FLIP_COS_THRESH) -> dict:
    flip = flip_target(v1_now, v1_next).float()
    n_pos = flip.sum().clamp(min=1.0)
    n_neg = (1.0 - flip).sum().clamp(min=1.0)
    pos_weight = (n_neg / n_pos).detach()
    bce = F.binary_cross_entropy_with_logits(
        out["flip_logit"], flip, pos_weight=pos_weight)
    ce = F.cross_entropy(out["state_logits"], lab_next.long())
    return {"bce": bce, "ce": ce, "flip": flip}
