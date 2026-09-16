"""Deterministic seeding helpers (D9)."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed numpy and the stdlib random module; sklearn is seeded per-call."""
    np.random.seed(seed)
    random.seed(seed)
