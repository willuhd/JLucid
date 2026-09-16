# Note: what are the "ev" files in data/pennlead? (side question, 2026-09-04)

## What "ev" is locally
- `data/pennlead/manifest/availability.csv` has flags `nb / rest / ev` (plus `fd_nb / fd_rest`).
  There are **NO ev files on disk** — `data/pennlead/ptseries/` contains only
  `*_ses-1_rest.ptseries.nii.gz` (104) and `*_ses-1_nback.ptseries.nii.gz` (104), plus
  nback events TSVs and fmriprep-style motion confounds. `ev=True` for 106 subjects.
- The local loader (`src/pennlead/datasets.py`) only handles conditions "rest" and "nback".

## What the source dataset actually contains
- pennlead = **Penn LEAD** (Penn Longitudinal Executive functioning in Adolescent
  Development), Sevchik/Shafiei/Murtha et al., PennLINC.
- Raw data: OpenNeuro **ds007116** (v1.0.2, DOI 10.18112/openneuro.ds007116.v1.0.2).
  fMRI per participant: **two resting-state runs (522 and 383 volumes, TR=0.8 s) and ONE
  n-back run (fractal n-back, 522 volumes)**. There is **no "ev" fMRI task** in the raw
  release (no emotion/EMOID fMRI task; the fMRI tasks are rest + n-back only).
  (One subject's first rest run was renamed `task-restfilm` because a movie played.)
- XCP-D derivatives: ds006741 (fmriprep derivatives) and ds006779 (XCP-D parcellated time
  series / FC). Other derivatives: ds006732 (sMRI), ds006739/640 (DWI), ds006744 (ASL).
  Note in ds006779: some subjects' extra n-back runs were manually deleted.

## Best explanation of the local "ev" flag
- Given the source release has no third fMRI task, the local `ev` flag most plausibly refers
  to something the project's own builder counted as "available": candidates are (a) the
  **second rest run** (383 volumes) — mislabeled; (b) an **event-files** availability flag
  (nback events TSVs); or (c) a CNB *cognitive-task* battery availability (Penn CNB includes
  e.g. Penn Emotion Recognition for Children — "emo"/"ev"-ish acronym) that was mislabeled as
  imaging availability. The column has no `fd_ev` counterpart (unlike nb/rest), which argues
  against it being an fMRI run with confounds.
- Verdict: **no "ev" imaging data exists to be downloaded**. What IS extra on the source side
  beyond what we have locally: rest run-2 (383 vol) and the full XCP-D parcellated
  derivatives. Both are the same Penn LEAD dataset.

## Access route (if the user ever wants it)
- All Penn LEAD releases are on **OpenNeuro under CC0** (public GitHub mirrors:
  OpenNeuroDatasets/ds007116 etc.). Downloading needs only the OpenNeuro web/API —
  **no DUC / NDA / data-use agreement is required**. (Contrast: NDA-held datasets like ABCD
  require DUC approval; Penn LEAD does not.)
- Current project rule says "don't download external datasets", so nothing was downloaded;
  this note is informational for a later user decision (e.g., pulling rest run-2 would
  roughly double Penn LEAD rest data at zero new-subject cost).

## Practical implication for the LEiDA goal
- Penn LEAD gives us rest + n-back only; the n-back run (522 vol, TR 0.8 = ~7 min task)
  is the task-dynamics asset. The second rest run would improve rest reliability, not add
  task contrast.
