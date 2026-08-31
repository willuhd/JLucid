"""Site harmonization — ComBat for ADHD-200 multi-site (8 sites).
Lightweight fallback: per-site z-score if neuroCombat not installed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

def combat_harmonize(features: np.ndarray, sites: np.ndarray, covariates: Optional[np.ndarray]=None) -> np.ndarray:
    """ComBat harmonization for controllability metrics across sites.
    features: (n_subjects, n_features) e.g. avg/modal per subject or per network
    sites: (n_subjects,) site labels (e.g. Peking, KKI, NYU...)
    covariates: (n_subjects, n_cov) e.g. age, sex, motion
    If neuroCombat not installed, fallback to per-site residualization.
    """
    try:
        from neuroCombat import neuroCombat
        data = features.T  # neuroCombat expects (features, subjects)
        covars = pd.DataFrame({"site": sites, "batch": sites})
        if covariates is not None:
            for i in range(covariates.shape[1]):
                covars[f"cov{i}"] = covariates[:, i]
        out = neuroCombat(dat=data, covars=covars, batch_col="batch")
        return out["data"].T
    except ImportError:
        print("[harmonization] neuroCombat not installed → per-site z-score fallback (pip install neuroCombat)")
        # per-site demean
        harmonized = features.copy().astype(float)
        for s in np.unique(sites):
            mask = sites == s
            m = harmonized[mask].mean(axis=0, keepdims=True)
            harmonized[mask] -= m
            # optional: rescale to grand mean
            harmonized[mask] += features.mean(axis=0, keepdims=True)
        return harmonized
    except Exception as e:
        print(f"[harmonization] ComBat failed {e} → returning unharmonized")
        return features
