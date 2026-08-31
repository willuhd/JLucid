# leida — ADHD dispersion analysis

The finding: inter-individual DISPERSION of normative phase-dynamics/controllability
deviations is elevated in pediatric ADHD — cross-dataset (ADHD-200 rest + PennLEAD
n-back), motion-robust, ADHD-specific. See `../../docs/REPORT.md`.

## What's here

| File group                         | What it does                                                                                                                                           |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `build_*.py`, `gate0_*.py`         | Feature construction (LEiDA co-leadership, phase-diff variance, Hancock-VAR, signed-A controllability, amplitude family) + split-half reliability gate |
| `run_gate1_*.py`, `run_gate2_*.py` | Oracle and LOSO-CV screening (batches 1-5)                                                                                                             |
| `run_batch6..9_*.py`               | DX, subtype, medication, PennLEAD ESWAN, nback screens                                                                                                 |
| `run_batch10..15_*.py`             | State-modulation, brain-age, single-site, amplitude, fALFF, FC-decomposition, and the certified heterogeneity/dispersion analysis                      |
| `run_pilot*.py`                    | Geometry pilot scans (0.04-0.095 Hz sweep)                                                                                                             |
| `make_figures.py`                  | Regenerates all report figures from cached artifacts in `../../results/`                                                                               |

## Requirements

numpy, scipy, pandas, scikit-learn, matplotlib, nibabel, nilearn. Run from repo root:

    PYTHONPATH=src python src/leida/make_figures.py

All scripts read from `data/` (ADHD-200 CC200 TCs, PennLEAD ptseries, Schaefer atlas)
and `results/` (feature caches, V1/V2 wavelet caches), writing outputs
back into `results/`.

## Provenance

Ported from JLucid2 `exp/37_leida` without functional changes — only path
constants were rebased (JLucid2 -> JLucid; exp subfolders -> results/37_leida).
The experiment-container docs (PREREG, SIEVE_TABLE, batch logs) live in `../../docs/`
and `../../results/` with results preserved verbatim.
