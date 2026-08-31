# exp/37 — LEiDA/Controllability Family Sieve: FINAL VERDICT

**12 batches, ~60 variants, 3 gates each. THE PAPER-6/7 FAMILY IS COMPREHENSIVELY NULL FOR ADHD.**

This document closes the family per the batch-12 preregistered stop rule. Every claim below is backed by a saved script + JSON in this directory or `../pilot_geometry/`.

## What was screened (the full inventory)

| Batch | Family / variants | Outcome |
|---|---|---|
| 1 | Co-leadership L (28/7/190-dim), first-moment M, λ-scalars, 5 geometries, FUSED (23 Gate-2 cells) | NULL (best Inatt r_sc=0.128 fw_p=0.45) |
| 2 | Phase-diff variance W1/W2 (Wang-2018 analog), λ1-std (Farinha VAR), network metastability, transition-control energies TCE (12 cells) | NULL; W1 pooled-r pathology diagnosed |
| 3 | Per-site estimator re-entries X1–X5 | NULL (X1 Inatt mean-per-site +0.199, fw_p=0.0995 — closest miss of the screen) |
| 4 | PennLEAD independent directional confirmation of X1 | FAILED (0/28 Holm; 9/15 sign-consistency p=0.61) |
| 5 | True Hancock VAR (H3), anti-phase v_ab (H1), magnetization (H2), leader-persistence ACF (P1 — dead at Gate 0), signed-A φ/μ (K1), CA-score (K2 — invalidated: paper-6 A has |eig|>1, Gramian diverges) | NULL (0/8 Gate-2 cells) |
| 6 | DX case-control pivot, 11 blocks | NULL (ΔAUC ∈ [−0.042, +0.003], fw_p=1.0) |
| 7 | Subtype (DX1/DX3) + medication strata, 33 AUC cells | NULL |
| 8 | PennLEAD ESWAN discovery, 27 cells | NULL (oracle→CV collapse, fw_p≥0.98) |
| 9 | nback task-modulation | NOT RUNNABLE (events/TR unrecoverable); **behavioral positive control d′→ESWAN r=0.35 fw_p=0.01 — the only alive cell, not imaging** |
| 10 | Task-state modulation (rest/nback/contrast), 12 cells | Lnet_rest×ESWAN alive at seed 0 (r=0.409, fw_p=0.037) → decomposed to seed-mean +0.21±0.09 |
| 11 | Preregistered confirmation of batch-10 positive on ADHD-200 Index (n=514, 7× PennLEAD) | **NULL** (r_sc=+0.058 fw_p=0.56; stop rule invoked: L-family closed) |
| 12 | Final theorist batch: CGATE (Gate-0 dead, SB 0.02), EDIFF, DYNCTRL, EMAP, D′, C′, A′ | **NULL** (best EDIFF×Hyper +0.141 fw_p=0.23; IQ canary ≤0.11 for all) |

## The three mechanistic laws this sieve established (with numbers)

1. **Reliability is necessary, not sufficient.** The most reliable dynamic features ever measured here — EMAP SB 0.87, D′ 0.84, DYNCTRL 0.76, EDIFF 0.75 (vs occupancy 0.32–0.51, and the corrected λ1 0.49–0.52) — delivered r_sc ≤ 0.06 on symptoms. Meanwhile exp/28's FC/precision blocks (SB ~0.6–0.8) delivered r=0.21 on IQ. Reliability gates noise, not relevance.

2. **Oracle→LOSO collapse is the family's signature.** Every batch shows in-sample r 0.3–0.79 collapsing to |r_sc| ≤ 0.2 honest. The family's covariance structure contains ADHD-200 symptom variance only in the sample-specific direction — 28–190 dims at n=336 is overfit territory for a true effect ≤0.1.

3. **Direction-consistent but magnitude-below-bar, everywhere.** The family's best honest cells: W1 Inatt +0.199 (fw_p 0.10, PennLEAD-confirmed-fail), Lnet×ESWAN +0.21 seed-mean (n=70, Index-confirmation-fail at +0.058), EDIFF×Hyper +0.141 (fw_p 0.23), H3 +0.22 per-site mean (pooled 0.07). A weak, real-looking positive direction that never reaches r≈0.25 + fw_p<0.05 anywhere. At n=336–514 that is indistinguishable from the sampling distribution of the max-statistic under a true r≈0.08–0.12 — exactly the attenuation ceiling exp/36 predicted (ρ_feature 0.3–0.5 × ρ_target 0.85 caps observable r at 0.5–0.65× true; true r must be ≥0.3 for a detectable 0.15).

## What remains genuinely alive in the program
- **exp/28 IQ model** (r=0.208, 6-site, adversarially verified): the family's static-architecture cousins (FC/precision/φ/μ) carry real cognitive signal — the target, not the features, was the problem.
- **exp/35 age model** (r=0.64 OOS): the strongest cross-site prediction in the program.
- **Behavioral d′→ESWAN r=0.35** (batch 9 positive control): ESWAN phenotypes carry valid clinical variance.
- **exp/29 male φ/μ×Hyper +0.196 lead** (needs 2–4× males to certify).

## The honest final statement for the family
*On ADHD-200 CC200 + PennLEAD, every published formula of the LEiDA/state-dynamics and precision-controllability families — including all extractor geometries, state models, second-moment and variance summaries, signed networks, energetic quadratic forms, fusion and interaction hybrids, across symptom, severity-index, subtype, medication, and case-control endpoints, under pooled and per-site estimators with preregistered multiplicity — is familywise-null. The measurable ceiling for any single-subject ADHD-symptom effect in this data is r ≈ 0.15–0.20 direction-at-best, unreproducible across targets. The imaging in these cohorts carries robust normative structure (age r=0.64, IQ r=0.21, both cross-site) — the ADHD-specific deviation is not expressed in resting-state phase or control architecture at the detectable floor of these samples.*

Closing the family per the stop rule. Any future ADHD-symptom claim from this data class would need new data (HBN DUA package prepped in exp/24) or new cohorts.
