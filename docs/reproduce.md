# Reproduce

Environment: .venv/bin/python -c with requirements.txt (numpy, scipy, pandas, sklearn, statsmodels, torch, matplotlib, h5py). No GPU needed (stay field trains ~1 s/epoch on CPU).

Generated artifacts go to ignored flat `outputs/`; bulky intermediates go to ignored `data/processed/`. Committed paper figures stay under `essay/figs/`.

1. Download data (docs/data-download.md) into data/.
2. o4 effect: .venv/bin/python src/o4_occupancy.py
   - Needs retained `data/e123_adhd200_denovo_labels.npz`, `data/e123_adhd200_denovo_C.npy`, phenotypics, per-site motion tables, plus `outputs/field_table.csv` (via `scripts/build_field_table.py`).
   - Writes flat `outputs/o4_subject.csv`, `outputs/o4_refits.csv`, `outputs/o4_summary.json`.
   - Expect d=-0.302, perm p=0.001 floor, SB=0.586, refit matched -0.299, literal -0.035, recovery 9/40.
   - The recon labels rebuild via `scripts/rebuild_filelens_labels.py`; the seed1 lens rebuilds via `scripts/rebuild_seed1_labels.py` (independent fit, permuted into denovo order).
3. Motion: .venv/bin/python scripts/essay_motion_stats.py
   - Needs `outputs/o4_subject.csv` plus per-site motion tables.
   - Writes flat `outputs/motion_stats.json`.
   - The effect survives motion matching; exact pair count depends on the motion-table join.
4. Rank-2/gauge: .venv/bin/python src/rank2_math.py ; .venv/bin/python src/gauge_artifact.py
   - Rank identity spot checks need retained CC200 runs.
   - The full operator sweep needs the legacy CC200/Yeo mapping, absent here.
   - Gauge checks use retained HBN inputs and `data/pilot_out/leida_centroids_190.npy`, not legacy `pilot/out/...`.
   - Expect sigma3/sigma1 ~3e-16 and gauge-flip contingency, not full figure regeneration.
5. J-space: audit + figures via src/jspace/ and essay figure logic in docs
   - Needs external J metrics/checkpoints absent from this checkout.
   - Expect the paper's bound: J stay-field metrics are occupancy functions; group means null.
   - Do not treat missing external J inputs as regenerable from retained data alone.

Figures: essay/figs/*.png are committed for the paper build; regenerate only when all missing inputs are supplied.
