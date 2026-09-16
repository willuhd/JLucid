# LEiDA/Controllability sieve — final report (2026-08-29)

## What was run
10 preregistered screen batches (protocol: `prereg.md`, results: `sieve-table.md` and
`../results/` batch `*.json`/`*.log`) covering the COMPLETE paper-6/7 feature family on the
QC'd ADHD-200 (n=872; T-cohort 335, ADHD-Index 514, DX 721) and PennLEAD (rest n=87,
nback n=86; ESWAN n=85-87) cohorts:

| family | batches | variants | best honest result |
|---|---|---|---|
| co-leadership L (2nd moment), m (1st moment), λ-scalars | 1 | 23 | r_sc 0.128 fw_p 0.45 |
| phase-diff variance W1/W2, VAR/metastability V1/V2, TCE | 2 | 12 | r_sc 0.104 fw_p 0.74 (per-site 0.199 fw_p 0.0995) |
| per-site estimator re-entries | 3 | 12 | mean per-site r 0.199 fw_p 0.0995 |
| PennLEAD independent confirmation of best | 4 | 1 | 9/15 signs, 0 Holm — FAILED |
| Hancock-true VAR, magnetization, v_ab, persistence ACF, signed-A φ/μ | 5 | 8 (+1 dead Gate0, +1 invalidated pre-fit) | r_sc 0.077 fw_p 0.75 |
| DX case-control | 6 | 11 | dAUC +0.003 |
| subtypes (inattentive/combined/unmedicated) + medication | 7 | 33+ | dAUC +0.004 (C1) |
| PennLEAD ESWAN discovery (rest) | 8 | 27 | CV r 0.192 fw_p 0.98 |
| nback event-aligned imaging | 9 | — | NOT RUNNABLE (dataset artifact) |
| whole-run state modulation + DX + ESWAN | 10 | 12 + split-half | Lnet_rest×ESWAN r 0.409 fw_p 0.037 → multi-seed p 0.056 → split-half p 0.43 — NOT CONFIRMED |

## Certified genuine signals (positive controls, honest CV, permutation-tested)
1. **d'(nback behavior) → ESWAN total: CV r=0.35, fw_p=0.01** (batch 9) — behavior↔symptom.
2. **Lnet28 (LEiDA co-leadership) → age within-site maturation: r=0.108, p=0.005** (ADHD-200,
   200 within-site perms; new for the DYNAMIC family; static age was exp/35).
3. exp/28 IQ r=0.208; exp/35 age r=0.64 (prior sessions, static features).

## Certified nulls (assay-validated negatives)
ADHD symptoms (Inatt/Hyper/Idx), ADHD diagnosis, ADHD subtypes, medication strata —
across both datasets, both task states, all geometries (0.05-0.095 Hz), both estimators —
NO genuine, cross-dataset-confirmable imaging signal in the LEiDA/controllability family
at detectable magnitudes (familywise null q95 ≈ 0.15-0.21 on ADHD-200, 0.28-0.40 on
PennLEAD). The one familywise-surviving candidate (Lnet_rest×ESWAN) failed multi-seed
and split-half confirmation — reported as NOT CONFIRMED, not as a hit.

## Interpretation for papers 6/7
- The feature family carries real, assay-detectable variance: reliability SB 0.38-0.56
  (Gate 0), age maturation r≈0.11 — but nothing ADHD-specific. This mirrors the
  meta-analytic ADHD null literature (Cortese 2020: no convergent FC alteration;
  Solodkin 2021 ALE null; Chen 2017 cross-site Dice ≤0.013; Shappell 2021 symptom-FDR-null
  despite group-state differences).
- Wang-2018-style phase variance features: reproducible-direction r≈0.10-0.20 in ADHD-200
  (all-site consistent) but below familywise bars and non-replicating in PennLEAD at n=85.
- Controllability: TCE and signed-A φ/μ are highly reliable (SB 0.48-0.56) but carry no
  ADHD signal in any contrast; CA-score on paper-6's A is mathematically ill-posed
  (A has |eig| up to 1.5 — discrete Gramian diverges) — a substantive methods finding.

## Cost accounting
All 10 batches: ~2.5 hours wall-clock on 4 cores. Features cached in
`../results/`{features,batch2_features,batch5_features,batch10_feats}.npz; every gate
result JSON + log preserved. Prereg discipline held throughout: every batch's variants
were frozen in `sieve-table.md` BEFORE any target fit; re-parameterizations were logged as
new variants and priced into the familywise null; the one discovered ALIVE cell was
audited (multi-seed, jackknife, Spearman, univariate, confounds) and honestly downgraded
when it failed confirmation.

## Addendum (batches 11-12 + moderation, 2026-08-29 late)

- Batch 11 (maturational lag / brain-age delta, Shaw-2007 hypothesis): ADHD−HC Δ=−0.28y
  (perm-p 0.062), Δ×Inatt r=−0.110 (p=0.048 nominal) — nothing survives Holm; PennLEAD
  does not confirm (Δ×ESWAN r=+0.067, p=0.60). NULL.
- Batch 12 (single-site site-5 discovery, n=187): best Lstr7 Inatt CVr=+0.172, fw_p=0.62;
  confound baseline alone r=0.202. NULL.
- Moderation model (Lnet×age interaction on Inatt): Δr=+0.013 over confound baseline. NULL.
- Certified positive controls stand: d'→ESWAN r=0.35 (fw_p=0.01); Lnet28→age
  within-site r=0.108 (p=0.005, ADHD-200); static IQ/age from prior exps.

The sieve has now exhausted: 3 target modalities × 2 datasets × 2 task states × 5
geometries × 3 estimator classes × 12 feature families × moderation terms. Under the
assay's demonstrated detection floor (honest CV |r|≥0.25 familywise, or |r|≥0.20 with
site-consistency), the paper-6/7 LEiDA/controllability family contains no genuine,
cross-dataset-confirmable ADHD signal in these cohorts. This is the modelable conclusion
of the screening program; the certified positive controls (esp. d'→ESWAN r=0.35) mark the
phenotype as sound and the null as a property of the imaging features, not the assay.

## NEXT-ROUND CANDIDATE DIRECTIONS (for goal continuation, round 4+)

Directions NOT yet tried, in order of literature support:
1. **Oscillatory power/amplitude (not phase)**: the ADHD EEG-fMRI literature's only reliable
   fMRI correlates are spectral (slow-3/4/5 ALFF differences; BMC Psychiatry 2025 AUC 0.755-
   0.783). LEiDA's wavelet machinery yields instantaneous AMPLITUDE per band for free;
   amplitude-network moments (analog of L but on |W(x,t)|) are untested and stay in the
   paper-6/7 spectral family.
2. **Cross-frequency coupling** (phase-amplitude, band-band coherence): PennLEAD TR=0.8s
   supports >0.1Hz bands that ADHD-200 cannot see; cross-dataset asymmetry documented but
   the 0.04-0.1 Hz CFC is computable in both.
3. **Heterogeneity-first modeling**: normative modeling (per-site HC z-scores then extreme-
   deviation counts), which handles the site-Dice≤0.013 problem by construction; deviation
   counts vs symptoms is the framing that rescued some EEG ADHD findings.
4. **Longer time windows / dwell beyond 1TR**: all batches used instantaneous per-TR
   features; 30-60s windowed versions of the winning-in-reliability families were only
   partially screened in exp/21-28 (documented nulls there).
5. **PennLEAD ESWAN subtype subscores** beyond total/inatt/hyper (if present in cohort.csv).

## ROUND-4 OUTCOME (2026-08-30): FIRST CERTIFIED CROSS-DATASET ADHD SIGNAL

See `dispersion.md`. Summary: after 14 null batches (all first-order screens),
the preregistered heterogeneity-first design (batch 15) found that ADHD shows ELEVATED
INTER-INDIVIDUAL DISPERSION of the paper-6/7 features (variance ratio ≈2.2): ADHD-200 rest
unmedicated n=192 vs 401, p=0.0005 (motion-matched p=0.023); PennLEAD nback n=41/45,
p=0.0025; within-subject state-change p=0.024. Robust to motion, normative covariates,
medication strata, sites. Not a severity biomarker (dimensional targets remain null) —
a group-level result consistent with the heterogeneity literature (Marquand 2019;
Wolfers 2020; Segal 2023).
Also closed this round: amplitude axis (batch 13/13b, 45 cells, cross-dataset null),
fALFF + phase×amplitude FC decomposition (batch 14, 15 cells null), cumulative score
(r=0.016), site×severity moderation. Positive controls re-verified: d'→ESWAN r=0.35
fw_p=0.01; age→Lnet28 r=0.108 p=0.005.
