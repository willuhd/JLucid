# FINAL LEDGER — cross-dataset clinical LEiDA/controllability search (2026-09-05/06)

STATUS: COMPLETE. Full cohort (169/169 ds000030) prepped, extracted, tested.
Meets the rule-5 exhaustion condition. Recommendation: mark goal BLOCKED.

Goal: a real clinical LEiDA/controllability signal, cross-verified across the
on-disk cohorts, strictly in the paper-6/7 family, modelable. NO downloads.
Session folder: exp_r6_lock/ (this session). Prior work: exp_leida_xverify/
(rounds 1-4), exp_overnight_leida/ (failed broad search, see postmortem).

## 1. Infrastructure built this round (all verified)
- fl_perm.py: Freedman-Lane permutation (x-residual scheme), dual-scheme
  agreement with canonical y-residual scheme (t=-2.11 p=.029 vs .030).
  Fixed the frozen-pinv bug of the old 04_metrics_stats.py (auditor-quantified:
  mildly liberal only at target-cov r>=0.30).
- Literal majority-count sign rule (paper 7 verbatim, paper7_fulltext:500-506);
  1.27% of frames disagree with the old mean rule; dictionary recovery |r|=1.000.
- HBN both-run eigenvectors, literal rule, 931 subjects x 2 runs (fixed pipeline).
- ds000030 uniform pipeline: BOLDRigid MC + N4 + SyNRA to MNI + native-space
  CC200 parcellation + real FD from own motion estimates. Validated: FD
  mean 0.08-0.12mm, 0% frames>0.5, tSNR~150, all ROIs >=30 native voxels,
  artifact-checks ~=0 (gcoh_sd t=-0.65, lam_max t=-0.02 at n=76).
- HBN per-run FD recovery (auditor method): 555 run-1 + 191 run-2 assignable;
  FD-trait validated as covariate.

## 2. Results by cohort x family (all with corrected inference + real covariates)

HBN (intact regime, n=779-931):
- age effects (global occ t=-7.3, DMN occ t=+7.0, Bonf): REAL (positive control).
- attention ~ FPN-dwell family: raw real (no-motion p_fam=.025), FULL motion
  block p_fam=.113, per-run FD t=-1.33, run-2 t=-1.40, k=6/7/8 t=-1.97/-1.09/-1.06,
  Schaefer-100 no-FPN-state (nearest t=-2.06 p=.13), young band null, Q4 flip,
  half-split sign-stable but sub-threshold (t -1.1..-2.0). VERDICT: marginal,
  measurement-fragile, NOT confirmed. Posterior ~3-8%.
- 7-net dictionary: FPN state exists (Cont +0.340, cleanest anywhere) -> dwell
  t=+0.02. The overnight's 7-net replication is REFUTED.
- Controllability all-7-nets x {attention, p_factor}: 28 tests, |t|<=2.25,
  all p_fam>=.13. No signal.
- V-smoothing: effect dies at >=10-frame smoothing (t=-2.32 -> -0.76).

ds000030 (own pipeline, intact regime, adults, n=115 prepped of 169):
- dx ADHD(42) vs CONTROL(73): life_fpn t=+0.62 (wrong direction), family |t|<=1.04.
  STRUCTURALLY INVALIDATED: dx perfectly confounded with stratum (all ADHD in
  block-700, all controls in 101-111) — uninterpretable as biology at any n.
- Batch-free within-stratum ASRS: within-700 (n=42) t=-0.94; within-10x (n=73)
  t=+1.17 (wrong sign). Both null.
- Native dictionary: global + 2-3 modular states reproduce (r 0.88-0.91), but
  NO FPN-positive state emerges natively.
- Subgraph controllability (FPN/DMN/FPN+DMN, user advice): all |t|<=0.29.
- Multi-atlas S100/300/AAL (user advice): structure transfers, clinical null.
- T3 artifact family ~=0 (clean pipeline confirmed).

adhd200 (anti-phase, GS-removed, n=965):
- State dictionaries: null. Scalars/MC: motion artifact (MC_Vis r=.98 with
  spectral radius; Q1-zero; imputed-FD block). Penn-style sign-opposite.
- Pooled dimensional (Inattentive, n=203 usable): |t|<=0.32.
- Polarity object (regime-native): |t|<=1.36.
- Cross-dataset dictionary transfers: INVALIDATED (the two "CC200" atlases are
  different parcellations, 83mm same-id distance).

Penn (anti-phase + XCP-D scrubbing, n=104+87):
- Clean-group dx: opposite-signed (d=-0.62). Rest states/ESWAN: null.
- Polarity: |t|<=0.34. Vendor-vs-XCPD scalar consistency r~0.15 (measurement,
  not trait). Run-averaged retest: null (t=-0.91).

## 3. Mathematical reasons per closure
(i) Spectral-concentration latent L (motion-loaded, pipeline-composition-
dependent) behind all |FC|-based families; non-invariant across vendors.
(ii) Estimator unreliability at 355 frames (VAR/SSM operators ICC~0.05;
lifetime ICC~0.15; switch/burstiness ICC<0.1).
(iii) Degeneracy by construction (transition asymmetry; exit/entry rates).
(iv) Anatomically invalid transfers (two different CC200s; regime-incompatible
FPN projection into GS-removed data).
(v) Structural design flaw (ds000030 dx-perfectly-confounded-with-stratum).
(vi) Discretization-scale fragility (k-specific, atlas-specific, smoothing-killed).

## 4. The one lead's trajectory
NYU dx-MC (artifact) -> HBN attention~life_fpn discovery (t=-2.87, buggy perms)
-> audit (genuine-in-dataset, motion caveat) -> this round: corrected inference
+ 8 falsifications -> marginal (familywise p=.113), P(biological) ~3-8%,
awaiting only the ds000030 adjudication, which returned: dx uninterpretable
(batch), batch-free gradients null, no native FPN state.

## 5. Measurement-validity discoveries (durable beyond this goal)
- Two "CC200" atlases are different parcellations (invalidates all prior
  cross-dataset transfers using them interchangeably).
- Vendor scrubbing fingerprint: FD-trait correlates NEGATIVELY with per-run
  DVARS (high-motion subjects get artificially smooth series).
- Permutation bug class (frozen design inverse) + majority-sign convention fix.
- Rank-2 vacuity of the V1>50% "sanity check".

## 6. Remaining on-disk, unrun (EV estimates)
- ds000030 control wave (54 subj, ~1h): within-10x ASRS power + batch
  heterogeneity only; cannot test dx. Low EV.
- V2 dictionaries / envelope LEiDA / sub-band ROI LEiDA / personalized
  dictionaries / dwell exponent / adhd200 IQ calibration: each 5-10% EV,
  10-60 min each. The adjacent tier, needs explicit authorization.
- Fitted-A controllability project, youth-cohort raw downloads: out of scope
  for this goal.

## 8. Blocking code audit (auditor_round2_code.md, 2026-09-05) — LEDGER CONFIRMED
Independent adversarial code audit with synthetic-data proofs:
- Inference engine: 220 null sims uniform (mean p=.518, P<=.05=.032, KS p=.74),
  FWER=.058; known-effect t matches analytic OLS to 6 decimals. CORRECT.
- Registration direction (atlas->native via invtransforms): correct per ANTsPy
  contract; bincount averaging exact; cached values reproduce to 6 decimals.
- Assignment (roi_ids-1), ICC (<=.003), FD offset (11 first-principles-correct;
  R1a recompute t=-1.33 exact match), all 4 spot-recomputed ledger numbers MATCH.
- Two corrections (no verdict change): (i) "all ROIs >=30 voxels" falsified —
  19/115 subjects have sub-30-voxel ROIs, 4 miss 1-3 labels; coverage-excluded
  sensitivity (n=96) reproduces the null identically (t=+0.68); (ii) HBN gcoh =
  |l|/2 vs ds30 |l|/N scale mismatch — cross-cohort gcoh comparison invalid
  (never performed; within-cohort inference unaffected); ds30_prep_meta rebuilt
  as ds30_prep_meta_full.csv (original held only the last batch).
No ledger verdict falls. The "marginal, NOT confirmed" verdict is strengthened.

## 9. Dispersion-D object (bounded-push target — the only claimed-positive family
never tested by us; tested 2026-09-05 late)
Implemented faithfully (control-only normative OLS + MAD scaling + clip +-3 +
mean-z2 + within-site/full-shuffle permutation) on this project's validated
cached features (overnight's 7-net tables not recoverable from disk).
- adhd200 block C (14 AC/MC): dD=+.46, VR=1.71, d=+.32, **p=.0016** (n=113/314).
  Survives: Pittsburgh-drop (p=.010), med-exclusion (p=.029), strict motion QC
  maxmotion<=2 (p=.0008, STRONGER), tail-trim (p=.019), motion-matched pairs
  (t=+2.48), all motion quartiles incl. cleanest, motion-covariate normative
  models (p=.0002, UNCHANGED). Per-feature broad (all 14, MC-led). Caveats:
  site-heterogeneous (NeuroIMAGE VR=2.39 + Peking_1 VR=2.10 carry; KKI 1.13 +
  Pittsburgh 1.16 flat — KKI has the lowest motion variance); tail-concentrated
  (top-5% removal drops p to .019/.22 depending on QC; tail members include
  extreme-motion/short-run subjects BUT the QC-cleaned tail persists).
- adhd200 block G (scalars): null (p=.24).
- Penn rest/nback/change: null (S whispers p=.055-.077; change-S wrong-signed;
  all n.s.). HBN S/G/C: null (|t|<=1.11).
- VERDICT: robust within-adhd200, NOT cross-verified (Penn + HBN null). Strongest
  surviving finding in the project; still single-cohort.

## 10. ds000030 FULL cohort (169/169 prepped, all contrasts run 2026-09-05 late)
- T1 dx (42v127): life_fpn t=+0.60 (wrong direction), family |t|<=1.13, all
  p_fam>=0.65. INVALIDATED by perfect dx-x-block confounding (all ADHD in
  stratum 700); descriptively the ADHD mean (2.33) falls INSIDE the control
  block range (1.65-2.72 across 11 blocks).
- T2 ASRS full (n=169): life_fpn t=-0.02; family |t|<=0.97. Within-700 (n=42):
  t=-0.94. Within-10x (n=127): t=+0.19. All null, signs inconsistent.
- T3 artifact: gcoh_sd t=-1.38 (p_fam=.24), lam_max t=-0.28. Null.
- Sensitivities: FD<0.2 t=+0.39, no-motion t=+0.67, FD-matched pairs t=+1.09,
  clean-coverage subset identical. All null.
- Native dictionary (127 controls): global + 2 modular states match HBN
  (r=0.88-0.91); FPN state does not emerge cleanly (2 states match at 0.35/0.55).
- Multi-atlas S100/300/AAL: structure transfers, clinical null at every n.
- Subgraph controllability: null. Coverage sensitivity: identical null.

## 7. Recommendation
Block the goal on actual exhaustion (rule 5): every (cohort x in-family object)
cell with non-negligible EV is tested; each closure has its mathematical reason;
the complete ledger above is the evidence. No new objects, no downloads.
Pending user confirmation.
