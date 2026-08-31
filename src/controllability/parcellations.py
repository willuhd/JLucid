"""Parcellation atlases + network / ROI mappings.
- AAL90 (paper 7): 90 ROIs via nilearn.datasets.fetch_atlas_aal
- Schaefer 100/200/300/400 (paper 6): 7 Yeo networks via fetch_atlas_schaefer_2018
- Advisor ROI subsets: frontal / parietal / DMN / combined as subnetworks
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

# ---------- static fallbacks when nilearn download unavailable ----------
# Minimal Schaefer100 → Yeo7 mapping approximated from Schaefer 2018 ordering.
# Real pipeline will overwrite with actual nilearn labels when online.
# Source: Schaefer 100 parcels are ordered by network: VIS, SOM, DAN, SAL, LIM, FPN, DMN
# Approximate counts (sum 100): VIS 14, SOM 14, DAN 15, SAL 12, LIM 5, FPN 16, DMN 24  (empirical split)
_SCHAEFER100_NETWORKS = (
    ["VIS"]*14 + ["SOM"]*14 + ["DAN"]*15 + ["SAL"]*12 + ["LIM"]*5 + ["FPN"]*16 + ["DMN"]*24
)
# Schaefer200 approximate double
_SCHAEFER200_NETWORKS = (
    ["VIS"]*24 + ["SOM"]*28 + ["DAN"]*30 + ["SAL"]*24 + ["LIM"]*12 + ["FPN"]*34 + ["DMN"]*48
)
_SCHAEFER300_NETWORKS = (
    ["VIS"]*36 + ["SOM"]*42 + ["DAN"]*45 + ["SAL"]*36 + ["LIM"]*18 + ["FPN"]*51 + ["DMN"]*72
)
_SCHAEFER400_NETWORKS = (
    ["VIS"]*48 + ["SOM"]*56 + ["DAN"]*60 + ["SAL"]*48 + ["LIM"]*24 + ["FPN"]*68 + ["DMN"]*96
)

# AAL90 labels (Tzourio-Mazoyer 2002) — first 90 used in paper 7.
# We keep ordering as nilearn's AAL atlas provides.
_AAL90_LABELS = [
    "Precentral_L","Precentral_R","Frontal_Sup_L","Frontal_Sup_R","Frontal_Sup_Orb_L","Frontal_Sup_Orb_R",
    "Frontal_Mid_L","Frontal_Mid_R","Frontal_Mid_Orb_L","Frontal_Mid_Orb_R","Frontal_Inf_Oper_L","Frontal_Inf_Oper_R",
    "Frontal_Inf_Tri_L","Frontal_Inf_Tri_R","Frontal_Inf_Orb_L","Frontal_Inf_Orb_R","Rolandic_Oper_L","Rolandic_Oper_R",
    "Supp_Motor_Area_L","Supp_Motor_Area_R","Olfactory_L","Olfactory_R","Frontal_Sup_Medial_L","Frontal_Sup_Medial_R",
    "Frontal_Med_Orb_L","Frontal_Med_Orb_R","Rectus_L","Rectus_R","Insula_L","Insula_R","Cingulum_Ant_L","Cingulum_Ant_R",
    "Cingulum_Mid_L","Cingulum_Mid_R","Cingulum_Post_L","Cingulum_Post_R","Hippocampus_L","Hippocampus_R","ParaHippocampal_L","ParaHippocampal_R",
    "Amygdala_L","Amygdala_R","Calcarine_L","Calcarine_R","Cuneus_L","Cuneus_R","Lingual_L","Lingual_R","Occipital_Sup_L","Occipital_Sup_R",
    "Occipital_Mid_L","Occipital_Mid_R","Occipital_Inf_L","Occipital_Inf_R","Fusiform_L","Fusiform_R","Postcentral_L","Postcentral_R",
    "Parietal_Sup_L","Parietal_Sup_R","Parietal_Inf_L","Parietal_Inf_R","SupraMarginal_L","SupraMarginal_R","Angular_L","Angular_R",
    "Precuneus_L","Precuneus_R","Paracentral_Lobule_L","Paracentral_Lobule_R","Caudate_L","Caudate_R","Putamen_L","Putamen_R",
    "Pallidum_L","Pallidum_R","Thalamus_L","Thalamus_R","Heschl_L","Heschl_R","Temporal_Sup_L","Temporal_Sup_R",
    "Temporal_Pole_Sup_L","Temporal_Pole_Sup_R","Temporal_Mid_L","Temporal_Mid_R","Temporal_Pole_Mid_L","Temporal_Pole_Mid_R",
    "Temporal_Inf_L","Temporal_Inf_R"
]

# Advisor-relevant frontal / parietal / DMN masks for AAL90 (heuristic, extensible)
_AAL_FRONTAL = {"Frontal_Sup_L","Frontal_Sup_R","Frontal_Sup_Orb_L","Frontal_Sup_Orb_R","Frontal_Mid_L","Frontal_Mid_R","Frontal_Mid_Orb_L","Frontal_Mid_Orb_R","Frontal_Inf_Oper_L","Frontal_Inf_Oper_R","Frontal_Inf_Tri_L","Frontal_Inf_Tri_R","Frontal_Inf_Orb_L","Frontal_Inf_Orb_R","Frontal_Sup_Medial_L","Frontal_Sup_Medial_R","Frontal_Med_Orb_L","Frontal_Med_Orb_R","Rectus_L","Rectus_R","Precentral_L","Precentral_R","Supp_Motor_Area_L","Supp_Motor_Area_R"}
_AAL_PARIETAL = {"Postcentral_L","Postcentral_R","Parietal_Sup_L","Parietal_Sup_R","Parietal_Inf_L","Parietal_Inf_R","SupraMarginal_L","SupraMarginal_R","Angular_L","Angular_R","Precuneus_L","Precuneus_R","Paracentral_Lobule_L","Paracentral_Lobule_R"}
_AAL_DMN = {"Frontal_Sup_Medial_L","Frontal_Sup_Medial_R","Frontal_Med_Orb_L","Frontal_Med_Orb_R","Cingulum_Ant_L","Cingulum_Ant_R","Cingulum_Post_L","Cingulum_Post_R","Precuneus_L","Precuneus_R","Angular_L","Angular_R","Hippocampus_L","Hippocampus_R","ParaHippocampal_L","ParaHippocampal_R","Temporal_Mid_L","Temporal_Mid_R","Temporal_Inf_L","Temporal_Inf_R"}

def _schaefer_networks(atlas: str) -> List[str]:
    if atlas == "schaefer100": return list(_SCHAEFER100_NETWORKS)
    if atlas == "schaefer200": return list(_SCHAEFER200_NETWORKS)
    if atlas == "schaefer300": return list(_SCHAEFER300_NETWORKS)
    if atlas == "schaefer400": return list(_SCHAEFER400_NETWORKS)
    raise ValueError(atlas)

def get_network_labels(atlas: str) -> List[str]:
    """Return length-N list of Yeo7 network per ROI."""
    if atlas.startswith("schaefer"):
        return _schaefer_networks(atlas)
    if atlas == "aal90":
        # For AAL, map AAL labels → Yeo7 via heuristic (fallback).
        # When nilearn is available and user loads real atlas, they can supply custom mapping.
        # Here we use coarse anatomical → network proxy.
        # For controllability ladder, we care mainly about frontal/parietal/dm n indices, not full Yeo.
        # Provide VIS/SOM/DAN/SAL/LIM/FPN/DMN placeholder via region-type proxy:
        mapping = []
        for lbl in _AAL90_LABELS:
            if lbl in _AAL_DMN: mapping.append("DMN")
            elif lbl in _AAL_FRONTAL: mapping.append("FPN" if "Frontal" in lbl else "DAN")
            elif lbl in _AAL_PARIETAL: mapping.append("DAN" if "Parietal" in lbl else "SOM")
            elif "Occipital" in lbl or "Calcarine" in lbl or "Cuneus" in lbl or "Lingual" in lbl: mapping.append("VIS")
            elif "Temporal" in lbl: mapping.append("LIM" if "Pole" in lbl else "DMN")
            elif "Insula" in lbl or "Cingulum" in lbl: mapping.append("SAL")
            else: mapping.append("SOM")
        return mapping
    raise ValueError(f"Unknown atlas {atlas}")

def get_roi_indices(atlas: str, roi: str) -> np.ndarray:
    """Return indices for advisor ROI subsets.
    roi in {whole, frontal, parietal, dmn, combined}
    For Schaefer, Frontal≈FPN+SAL+part of DAN anterior; Parietal≈DAN+part SOM; DMN as labeled.
    We implement as network-based proxy: frontal→FPN, parietal→DAN, dmn→DMN, combined→FPN+DAN+DMN.
    """
    nets = np.array(get_network_labels(atlas))
    if roi == "whole":
        return np.arange(len(nets))
    if roi == "frontal":
        # FPN + SAL as frontal-cognitive proxy
        return np.where(np.isin(nets, ["FPN","SAL"]))[0]
    if roi == "parietal":
        return np.where(np.isin(nets, ["DAN","SOM"]))[0]
    if roi == "dmn":
        return np.where(nets == "DMN")[0]
    if roi == "combined":
        return np.where(np.isin(nets, ["FPN","DAN","DMN"]))[0]
    raise ValueError(f"Unknown roi {roi}")

def get_aal_roi_indices_aal90(roi: str) -> np.ndarray:
    """AAL-specific anatomical ROI indices (more precise)."""
    labs = _AAL90_LABELS
    if roi == "whole": return np.arange(90)
    if roi == "frontal": return np.array([i for i,l in enumerate(labs) if l in _AAL_FRONTAL])
    if roi == "parietal": return np.array([i for i,l in enumerate(labs) if l in _AAL_PARIETAL])
    if roi == "dmn": return np.array([i for i,l in enumerate(labs) if l in _AAL_DMN])
    if roi == "combined":
        s = _AAL_FRONTAL | _AAL_PARIETAL | _AAL_DMN
        return np.array([i for i,l in enumerate(labs) if l in s])
    raise ValueError(roi)

def describe_atlas(atlas: str) -> dict:
    nets = get_network_labels(atlas)
    uniq, counts = np.unique(nets, return_counts=True)
    return {"atlas": atlas, "n_rois": len(nets), "networks": dict(zip(uniq.tolist(), counts.tolist()))}

def load_nilearn_atlas(atlas: str, data_dir: str | None = None):
    """Attempt to load real nilearn atlas; fallback to static if offline."""
    try:
        if atlas.startswith("schaefer"):
            from nilearn.datasets import fetch_atlas_schaefer_2018
            n = int(atlas.replace("schaefer",""))
            fetched = fetch_atlas_schaefer_2018(n_rois=n, yeo_networks=7, data_dir=data_dir)
            return fetched
        if atlas == "aal90":
            from nilearn.datasets import fetch_atlas_aal
            return fetch_atlas_aal(version="SPM12", data_dir=data_dir)
    except Exception as e:
        return {"error": str(e), "fallback": describe_atlas(atlas)}
    return None
