"""PennLEAD dataset — isolated module (single-site Prisma TR 0.8s).
Does NOT import from src-controllability or src-leida at import time to avoid
cross-package GIL/locking when running parallel subagents.
Torch 2.13 CPU, nibabel 5.4.2, same stats as Paper 6 §2.6.
"""
__version__ = "0.1.0-pennlead"
