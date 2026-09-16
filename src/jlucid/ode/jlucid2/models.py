"""Latent VAE + Neural ODE (short-shot protocol).

Same modules as the original run (GRU encoder, 2-layer tanh MLP field,
MLP decoder + state head). Training protocol is what changed:

- encode a w-TR window ending at t -> z0
- Euler-integrate H steps with p(t) HELD (no future labels)
- residual sphere decode: V1_hat(t+h) = unit(V1(t) + Dec(z_h))
- loss is 1-cosine on those short forecasts, not 200-step MSE from t=0

The old full-scan forward/loss stay so checkpoint_full.pt can still be
loaded and scored with the honest local-rollout evaluator.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jlucid.config import (
    BETA_KL,
    CONDITION_MODE,
    ETA_SMOOTH,
    GAMMA_STATE,
    HIDDEN_LAYERS,
    HIDDEN_SIZE,
    HORIZON_TR,
    L_CONSIST_WEIGHT,
    L_NOW_WEIGHT,
    LATENT_DIM,
    ODESOLVER,
    PREDICT_RESIDUAL,
    STEP3_INPUT_DIM,
    STEP3_K_COND,
    UNIT_SPHERE,
)

try:
    from torchdiffeq import odeint as _odeint
    HAS_TORCHDIFFEQ = True
except Exception:  # pragma: no cover
    _odeint = None
    HAS_TORCHDIFFEQ = False


def _mlp(in_dim: int, out_dim: int, hidden: int = HIDDEN_SIZE,
         layers: int = HIDDEN_LAYERS, act: str = "tanh") -> nn.Sequential:
    acts = {"tanh": nn.Tanh, "relu": nn.ReLU}
    seq = []
    dims = [in_dim] + [hidden] * layers + [out_dim]
    for i in range(len(dims) - 1):
        seq.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            seq.append(acts[act]())
    return nn.Sequential(*seq)


class WindowEncoder(nn.Module):
    """GRU over w consecutive V1 frames -> mu, logvar (VAE initial state)."""

    def __init__(self, in_dim: int = STEP3_INPUT_DIM, latent: int = LATENT_DIM,
                 hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=False, num_layers=1)
        self.head = nn.Linear(hidden, 2 * latent)

    def forward(self, windows: torch.Tensor):
        """windows: (B, w, d) -> (mu, logvar) each (B, latent)."""
        h, _ = self.gru(windows.transpose(0, 1))   # (w, B, hidden)
        mu_logvar = self.head(h[-1])
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar


class DynamicsMLP(nn.Module):
    """f_theta(z, p): concatenated MLP (plan Phase 6)."""

    def __init__(self, latent: int = LATENT_DIM, k: int = STEP3_K_COND,
                 hidden: int = HIDDEN_SIZE, layers: int = HIDDEN_LAYERS):
        super().__init__()
        self.net = _mlp(latent + k, latent, hidden=hidden, layers=layers)

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, p], dim=-1))


class DecoderMLP(nn.Module):
    """z -> V1_hat (90-dim) plus a linear state-logit head."""

    def __init__(self, latent: int = LATENT_DIM, out_dim: int = STEP3_INPUT_DIM,
                 k: int = STEP3_K_COND, hidden: int = HIDDEN_SIZE,
                 layers: int = HIDDEN_LAYERS):
        super().__init__()
        self.net = _mlp(latent, out_dim, hidden=hidden, layers=layers)
        self.state_head = nn.Linear(latent, k)

    def forward(self, z: torch.Tensor):
        """Returns (v1_hat (B,90), state_logits (B,k))."""
        return self.net(z), self.state_head(z)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar.clamp(-20, 4))
    return mu + torch.randn_like(std) * std


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu.pow(2) + logvar.exp().clamp(max=50) - 1 - logvar).sum(dim=-1).mean()


class LatentODEModel(nn.Module):
    """Encoder + dynamics ODE + decoder.

    ``predict_residual`` / ``condition`` / ``unit_sphere`` select the short-shot
    protocol. Defaults match the new training setup; pass
    predict_residual=False, unit_sphere=False to score the original checkpoint.
    """

    def __init__(self, latent: int = LATENT_DIM, k: int = STEP3_K_COND,
                 window: int = 7, in_dim: int = STEP3_INPUT_DIM,
                 hidden: int = HIDDEN_SIZE, layers: int = HIDDEN_LAYERS,
                 solver: str = ODESOLVER, rtol: float = 1e-3, atol: float = 1e-5,
                 predict_residual: bool = PREDICT_RESIDUAL,
                 condition: str = CONDITION_MODE,
                 unit_sphere: bool = UNIT_SPHERE):
        super().__init__()
        self.latent = latent
        self.k = k
        self.window = window
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.predict_residual = bool(predict_residual)
        self.condition = condition
        self.unit_sphere = bool(unit_sphere)
        self.encoder = WindowEncoder(in_dim=in_dim, latent=latent, hidden=hidden)
        self.dynamics = DynamicsMLP(latent=latent, k=k, hidden=hidden, layers=layers)
        self.decoder = DecoderMLP(latent=latent, out_dim=in_dim, k=k,
                                  hidden=hidden, layers=layers)

    def encode_windows(self, v1: torch.Tensor):
        """Sliding windows over (T, d) -> (mu, logvar) for window-ending frames."""
        w = self.window
        t = v1.shape[0]
        windows = torch.stack([v1[i:i + w] for i in range(t - w + 1)])
        return self.encoder(windows)

    def _ode_func(self, p: torch.Tensor):
        """Returns func(t, z) using nearest-frame p for adaptive solvers."""
        k = self.k

        def func(t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            idx = min(max(int(round(float(t))), 0), p.shape[0] - 1)
            pt = p[idx:idx + 1].expand(z.shape[0], k)
            return self.dynamics(z, pt)

        return func

    def integrate(self, z0: torch.Tensor, p: torch.Tensor,
                  times: torch.Tensor) -> torch.Tensor:
        """Integrate f_theta from z0 over times (T,). Returns (T, latent).

        Single-series helper used by the original full-scan forward.
        """
        if self.solver == "euler" or not HAS_TORCHDIFFEQ:
            z = z0.clone()
            out = [z]
            for i in range(len(times) - 1):
                pt = p[i:i + 1].expand(z.shape[0], self.k)
                z = z + self.dynamics(z, pt)
                out.append(z)
            return torch.stack(out)[:, 0]
        sol = _odeint(self._ode_func(p), z0, times, method=self.solver,
                      rtol=self.rtol, atol=self.atol)
        return sol[:, 0]

    def integrate_steps(self, z0: torch.Tensor, p: torch.Tensor,
                        n_steps: int, hold_p: bool = True) -> torch.Tensor:
        """Batched Euler. z0 (B, d); p is (B, k) if hold_p else (n_steps, B, k).

        Returns z including z0, shape (n_steps+1, B, d).
        """
        z = z0
        out = [z]
        for i in range(n_steps):
            pt = p if hold_p else p[i]
            if self.condition == "none":
                pt = torch.zeros(z.shape[0], self.k, device=z.device, dtype=z.dtype)
            z = z + self.dynamics(z, pt)
            out.append(z)
        return torch.stack(out)

    def decode_path(self, z: torch.Tensor, v1_now: torch.Tensor | None = None):
        """z (H+1, B, d) -> unit/raw V1 (H+1, B, 90) and logits (H+1, B, k)."""
        h1, b, d = z.shape
        raw, logits = self.decoder(z.reshape(h1 * b, d))
        raw = raw.view(h1, b, -1)
        logits = logits.view(h1, b, -1)
        if self.predict_residual:
            if v1_now is None:
                raise ValueError("residual decode needs v1_now (B, 90)")
            v1_hat = F.normalize(v1_now.unsqueeze(0) + raw, dim=-1)
        elif self.unit_sphere:
            v1_hat = F.normalize(raw, dim=-1)
        else:
            v1_hat = raw
        return v1_hat, logits, raw

    def forward_shots(self, windows: torch.Tensor, p0: torch.Tensor,
                      n_steps: int = HORIZON_TR, sample: bool = True,
                      v1_now: torch.Tensor | None = None) -> dict:
        """Short-shot forward. windows (B, w, 90), p0 (B, k)."""
        mu, logvar = self.encoder(windows)
        z0 = reparameterize(mu, logvar) if sample else mu
        if self.condition == "none":
            p_use = torch.zeros_like(p0)
        else:
            p_use = p0
        z = self.integrate_steps(z0, p_use, n_steps, hold_p=True)
        if v1_now is None:
            v1_now = windows[:, -1]
        v1_hat, logits, raw = self.decode_path(z, v1_now)
        return {
            "z": z, "mu": mu, "logvar": logvar,
            "v1_hat": v1_hat, "state_logits": logits, "raw": raw,
            "v1_now": v1_now,
        }

    def forward_ode(self, v1: torch.Tensor, p: torch.Tensor,
                    sample: bool = True) -> dict:
        """Integrate-then-reconstruct over one subject series.

        v1: (T, d) normalized; p: (T, k). Returns dict with z_hat, z_enc,
        mu/logvar, v1_hat, state_logits, all (T, ...) aligned.
        """
        t = v1.shape[0]
        w = self.window
        mu, logvar = self.encode_windows(v1)
        z0 = reparameterize(mu[:1], logvar[:1]) if sample else mu[:1]
        times = torch.arange(t, dtype=torch.float32, device=v1.device)
        z_hat = self.integrate(z0, p, times)
        v1_hat, state_logits = self.decoder(z_hat)
        z_enc = torch.full_like(z_hat, float("nan"))
        z_enc[w - 1:] = mu
        return {
            "z_hat": z_hat, "z_enc": z_enc, "mu": mu, "logvar": logvar,
            "v1_hat": v1_hat, "state_logits": state_logits,
        }

    def forward(self, v1: torch.Tensor, p: torch.Tensor,
                sample: bool = True) -> dict:
        return self.forward_ode(v1, p, sample=sample)


def model_loss(model: LatentODEModel, out: dict, v1: torch.Tensor,
               labels: torch.Tensor, mask: torch.Tensor,
               wmask: torch.Tensor, beta: float = BETA_KL,
               gamma: float = GAMMA_STATE, eta: float = ETA_SMOOTH,
               w_consist: float = L_CONSIST_WEIGHT) -> dict:
    """Step-3 loss on one subject (masked to usable frames)."""
    m = mask
    v1_hat = out["v1_hat"]
    z_hat = out["z_hat"]
    recon = F.mse_loss(v1_hat[m], v1[m])
    kl = kl_divergence(out["mu"], out["logvar"])
    lab = labels[m].long()
    state_ce = F.cross_entropy(out["state_logits"][m], lab)
    valid_pairs = m[:-1] & m[1:]
    smooth = ((z_hat[1:] - z_hat[:-1])[valid_pairs]).pow(2).mean()
    z_enc = out["z_enc"]
    ok = wmask & torch.isfinite(z_enc[:, 0])
    if ok.any():
        consist = ((z_hat[ok] - z_enc[ok]) ** 2).mean()
    else:
        consist = torch.zeros((), device=v1.device)
    total = recon + beta * kl + gamma * state_ce + eta * smooth + w_consist * consist
    return {
        "total": total, "recon": recon, "kl": kl, "state_ce": state_ce,
        "smooth": smooth, "consist": consist,
    }


def encode_no_ode(model: LatentODEModel, v1: torch.Tensor) -> torch.Tensor:
    """Encoder-only latent path (used by the no-ODE comparison / ablations)."""
    mu, _ = model.encode_windows(v1)
    return mu


def shot_loss(out: dict, v1_future: torch.Tensor, labels_future: torch.Tensor,
              v1_now: torch.Tensor, beta: float = BETA_KL,
              gamma: float = GAMMA_STATE, eta: float = ETA_SMOOTH,
              now_w: float = L_NOW_WEIGHT) -> dict:
    """Short-shot loss. v1_future (B, H, 90), labels_future (B, H)."""
    pred = out["v1_hat"][1:].transpose(0, 1)          # (B, H, 90)
    cos = (pred * v1_future).sum(dim=-1)
    pred_loss = (1.0 - cos).mean()
    now = out["v1_hat"][0]
    now_loss = (1.0 - (now * v1_now).sum(dim=-1)).mean()
    kl = kl_divergence(out["mu"], out["logvar"])
    k = out["state_logits"].shape[-1]
    logits = out["state_logits"][1:].transpose(0, 1).reshape(-1, k)
    ce = F.cross_entropy(logits, labels_future.reshape(-1).long())
    dz = out["z"][1:] - out["z"][:-1]
    smooth = dz.pow(2).mean()
    total = pred_loss + now_w * now_loss + beta * kl + gamma * ce + eta * smooth
    return {
        "total": total, "pred": pred_loss, "now": now_loss,
        "kl": kl, "state_ce": ce, "smooth": smooth,
        "cos1": cos[:, 0].mean() if cos.shape[1] else cos.mean(),
    }
