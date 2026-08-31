"""Central config: parcellation choices, c-sweep, ROI definitions."""
from dataclasses import dataclass, field
from typing import List, Literal

AtlasName = Literal["aal90", "schaefer100", "schaefer200", "schaefer300", "schaefer400"]
NormName = Literal["none", "fisher", "global_zscore", "row_zscore"]
RoiScope = Literal["whole", "frontal", "parietal", "dmn", "combined"]

@dataclass
class PipelineConfig:
    atlas: AtlasName = "schaefer100"
    c: float = 1.0
    c_sweep: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])
    fc_norm: NormName = "none"
    roi: RoiScope = "whole"
    n_permutations: int = 5000
    fdr_q: float = 0.05
    use_abs_fc: bool = True  # if False, Eq.4 without abs (robustness, paper 6 §2.7)
    tr: float = 2.0
    drop_first_n_vols: int = 10
    bandpass_low: float = 0.01
    bandpass_high: float = 0.1
    fwhm: float = 6.0
    seed: int = 42

# Yeo 7-network names as in Schaefer 100 (paper 6 §2.4)
YEO7 = ["VIS","SOM","DAN","SAL","LIM","FPN","DMN"]

# Advisor-relevant frontal/parietal/DMN groupings — mapped in parcellations.py
ADVISOR_ROIS = ["frontal", "parietal", "dmn", "combined"]
