"""Load stay / dest / when weights into a single JLucid 4 checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from jlucid.ode.jlucid4.models import SwitchConfig, ThesisModel


def pack_discrete(model: ThesisModel, dest_ckpt: Path, when_ckpt: Path) -> None:
    """Copy dest encoder/router and when-head into ``model``. Stay is untouched."""
    dest = torch.load(dest_ckpt, map_location="cpu", weights_only=False)
    sd = dest["state_dict"]
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    route = {k[len("router."):]: v for k, v in sd.items() if k.startswith("router.")}
    model.dest_encoder.load_state_dict(enc)
    model.dest_router.load_state_dict(route)

    when = torch.load(when_ckpt, map_location="cpu", weights_only=False)
    cfg = SwitchConfig(**(when.get("config") or {}))
    if (cfg.in_dim != model.when.cfg.in_dim
            or cfg.hidden != model.when.cfg.hidden):
        from jlucid.ode.jlucid4.models import SwitchHead
        model.when = SwitchHead(cfg)
        model.register_buffer("switch_mean", torch.zeros(cfg.in_dim))
        model.register_buffer("switch_std", torch.ones(cfg.in_dim))
    model.when.load_state_dict(when["state_dict"])
    mean = np.asarray(when["scaler_mean"], dtype=np.float32)
    std = np.asarray(when["scaler_std"], dtype=np.float32)
    with torch.no_grad():
        model.switch_mean.copy_(torch.from_numpy(mean))
        model.switch_std.copy_(torch.from_numpy(std))
    model.switch_tau = float(when["tau"])
    cents = when.get("centroids")
    if cents is not None:
        model.centroids = np.asarray(cents, dtype=np.float32)
    model._discrete_packed = True
    model.freeze_discrete()


def load_thesis_checkpoint(path: Path, device) -> tuple[ThesisModel, dict]:
    """Load a packed JLucid 4 checkpoint. Refuses the pack-only ode39 archive."""
    path = Path(path)
    if "packonly_ode39" in path.resolve().parts:
        raise ValueError(
            f"refusing pack-only ode39 archive {path}; "
            "use data/processed/step3/ode4/checkpoint.pt"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    from jlucid.config import WINDOW_TR
    model = ThesisModel(
        latent=int(cfg.get("latent", 12)),
        window=int(cfg.get("window", WINDOW_TR)),
        gamma_mode=str(cfg.get("gamma_mode", "learned")),
        gamma_value=float(cfg.get("gamma_value", 0.5)),
        decode_mode=str(cfg.get("decode_mode", "recursive")),
        small_decode_init=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.switch_tau = float(ckpt.get("switch_tau", model.switch_tau))
    model._discrete_packed = bool(ckpt.get("discrete_packed", False))
    if "switch_mean" in ckpt.get("state_dict", {}):
        model._discrete_packed = True
    if ckpt.get("centroids") is not None:
        model.centroids = np.asarray(ckpt["centroids"])
    model.eval()
    return model, ckpt


def load_stay_into(model: ThesisModel, stay_ckpt: Path) -> None:
    """Copy encoder + stay (+ leftover router if present) from a stay ckpt."""
    ck = torch.load(stay_ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    stay = {k[len("stay."):]: v for k, v in sd.items() if k.startswith("stay.")}
    model.encoder.load_state_dict(enc)
    model.stay.load_state_dict(stay)
