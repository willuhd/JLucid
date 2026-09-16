# merge_jlucid2 — minimal JLucid2 subset staged for codebase merge
Staged 2026-09-06 from `/Volumes/thinkplus/Code/JLucid2` (read-only source; nothing modified there).
Total: ~3.3 MB. Relative paths mirror the source repo so moves are trivial.

## Contents and paper mapping
- `jlucid/ode/jlucid4/` — finalized ThesisModel (switching-linear stay field
  `A(p)`, GRU encoder, cosine-loss training, J-space SVD/ablation/energy;
  §3 "J-space as trained"). Includes `models.py jspace.py jacobians.py eval.py
  train.py discrete.py jfuture.py shots.py pack.py`.
- `jlucid/ode/{jlucid2,jlucid3,jlucid23}/` — earlier model generations
  (comparison context for §3; small).
- `jlucid/control/` (`controllability.py energy.py stabilize.py state_fc.py
  targets.py`) — paper-6 AC/MC/energy machinery backing §2/§4.
- `jlucid/stats/` (`effects.py permutation.py symptoms.py`) — effect sizes,
  permutation/FDR utilities.
- `jlucid/config.py` — constants (JSPACE_K, seeds, window/TR, Yeo nets).
- `data/processed/step3/jspace_group_stats.csv` — J-space clinical nulls
  (Tab:jspace: C_k d=-0.085, r_eff +0.083, dmn +0.081; n=124/183).
- `data/processed/step3/jspace_metrics.csv` — per-subject J-space metrics +
  singular values (n=307; C_k sd 0.011 — the population-flow signature).
- `data/processed/step3/jspace_control.csv`, `latent_energy.csv`,
  `h4_symptom_scores.csv` — control/energy/symptom supporting tables.
- `data/processed/step3/jfuture/*.csv` — ablation/transition-energy metrics
  (ablation_per_subject, when_clinical, group_stats, metrics,
  impl_sanity_dest, hbn_attention_stats, hbn_cc200_fc).
- `data/processed/step3/{ode4,ode35,ode32}/*.pt` — trained checkpoints for the
  finalized generations (2.6 MB total; ode3/older generations excluded).

## Deliberately excluded (bloat)
- `exp/` (38 experiment dirs), `notebooks/`, `sbi-logs/`, `data/processed/`
  subject-level derivatives, `ode3` + older checkpoint generations (~24 MB),
  `results/`, tests/, docs. Nothing referenced by the paper lives there.

## Training reproducibility (added 2026-09-06)
- `scripts/41_train_ode4.py`, `scripts/42_eval_ode4.py` — launchers (all
  imports resolve inside the staged `jlucid/` tree: config, ode.data,
  ode.jlucid4.{models,pack,train}, utils.logging).
- `tests/` (112K) — executable spec incl. test_ode4*, test_jfuture*,
  test_controllability*, test_permutation*.
- `requirements.txt` — pinned deps (torch 2.13, numpy, scipy, sklearn...).
- Retraining inputs (window bundles under JLucid2 `data/`, 4.7 GB working
  area incl. the 505 MB Athena filtfix archive) are NOT staged: they regenerate
  from the same ADHD-200 preprocessed release already in JLucid `data/adhd200`
  via the staged loaders (`jlucid/io/athena.py`, `jlucid/ode/data.py`) + prep
  scripts. `RUN.md` documents the recipe (kept in JLucid2 only as reference).
- Deletion rule for JLucid2: safe to delete everything EXCEPT `data/` until a
  retrain has regenerated bundles from JLucid-side data; source, tests,
  weights, and results above are fully staged here.

## Dependency gaps (retraining only; results unaffected)
- `h5py` (bundle I/O in `ode/*/shots.py`, `ode/data.py`) and possibly
  `torchdiffeq` are NOT in JLucid's venv and NOT in the copied
  `requirements.txt` (which is itself incomplete upstream). Retraining needs
  `pip install h5py torchdiffeq` (versions per JLucid2's lockfile, if any).
- Verified 2026-09-06, module by module in JLucid's venv:
  `ode.jlucid4.models` (ThesisModel) OK; `control.controllability` OK;
  `stats.permutation` OK; `ode.jlucid4.jspace` needs h5py (via shots.py
  bundle loading). Train/eval drivers need `pip install h5py torchdiffeq`.
