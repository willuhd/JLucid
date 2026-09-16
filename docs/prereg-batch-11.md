# Batch 11 (confirmation run): Lnet/Lstr7 × ADHD-Index, preregistered BEFORE any fit

**Confirmation of batch-10 screen-positive (Lnet_rest × ESWAN-total, PennLEAD n=70, CV r=+0.21 seed-mean / +0.41 luckiest-seed, all-seeds-positive, fw_p=0.037 at seed 0).** NOT a new screen — a directional confirmation on a cohort and target never touched by these cells.

## Why this is the correct confirmation cell
- Batch-10 positive: co-leadership network-block architecture (Lnet28, cache geometry g050 c5 K60) predicts PennLEAD ESWAN ADHD total.
- The L-family × ADHD-Index cells (sites 1/3/5, n=514) have NEVER been fitted (verified against `sieve-table.md` batches 1–10: batch-1 Gate 2 was T-cohort Inatt/Hyper only; batch-2 A-family was W1/W2 only; batch-5 A-cohort cells were H1/H3/K1 only). Registered as the t6 theorist's "V1 L-IDX" variant, never run.
- ADHD-Index is the largest ADHD-severity cohort in ADHD-200 (n=514), parent/teacher-rated Conners composite — a different RATER and DATASET from ESWAN self-ratings: a genuine direction test of the family.

## Frozen cells (2)
- Lnet28 × ADHD-Index (28-dim block, cache geometry, all 872-subject features.npz exists)
- Lstr7 × ADHD-Index (7-dim row-strength network means, same archive)

## Estimators (both preregistered; primary declared now)
- PRIMARY: LOSO site-centered r_sc (3-site LOSO over sites 1/3/5), ridge α inner-fold mode, fixed under perms.
- SECONDARY (co-reported, known pooled-r pathology fix from batch 3): mean per-site r (within-site z-scored predictions).
- CV seeds: 5 seeds (0–4) AVERAGED, reported with sd — the batch-10 lesson (seed-luck) is priced in: the statistic is the seed-mean, not the best seed.

## Success criteria (preregistered, BEFORE the run)
CONFIRMED if: seed-mean r_sc > +0.10 (direction-consistent with PennLEAD +0.21 at 7× n, attenuated) AND familywise p < 0.05 (200 perms, within-site permutation, max-stat over the 2 cells × 2 estimators) AND per-site r positive at ≥2/3 sites.
NULL if: seed-mean r_sc ≤ 0.05 OR fw_p ≥ 0.05.
AMBIGUOUS (0.05 < r_sc ≤ 0.10 or mixed sites): record as unconfirmed; family stays open at 'weak-positive, unconfirmed' status.

## Claim path
If CONFIRMED: combined evidence claim (PennLEAD screen + ADHD-200 Index confirmation) goes to the full 4-agent adversarial audit before any clinical statement. If NULL/AMBIGUOUS: batch-10 positive is recorded as small-n screen noise; the family verdict remains the comprehensive negative with one unconfirmed weak positive.

## Multiplicity honesty
This confirmation runs 2 cells (Lnet28, Lstr7) because both are the same family at different granularity — the max-stat null covers both (and both estimators). No other variants are run in this batch. If null, NO further L-family re-testing on Index (stop rule: the L-family is then closed on every symptom target, estimator, cohort, and dataset available to this program).
