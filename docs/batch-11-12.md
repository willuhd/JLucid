## Batch 11 — CLOSED 2026-08-30. Confirmation of batch-10 positive: NULL.

Prereg (`prereg-batch-11.md`, written before fit): Lnet28/Lstr7 x ADHD-Index
(sites 1/3/5, n=514), 5 CV seeds averaged, LOSO + per-site estimators, 200-perm max-stat
null over 2 cells x 2 estimators. Confirmation bar: r_sc>0.10 AND fw_p<0.05 AND >=2/3 sites.

- Lnet28: seed-mean r_sc=+0.0575 (fw_p=0.56); per-site mean +0.081 (fw_p=0.32);
  site r's +0.120/+0.070/+0.063 (all positive).
- Lstr7: r_sc=+0.066 (fw_p=0.47); site r's +0.072/+0.107/−0.023 (mixed).
- VERDICT: NULL (direction-consistent with PennLEAD but 2-4x below the bar; fw_p far from 0.05).
- STOP RULE invoked: L-family (Lnet28/Lstr7/Mnet7/Lrow/Mrow, all geometries, FUSED,
  per-site estimator) is CLOSED on every symptom target, cohort, and dataset available.
- Cross-dataset summary of the family's best cells: PennLEAD ESWAN r=+0.21 (seed-mean,
  n=70), ADHD-Index r=+0.06 (n=514), T-cohort Hyper r=+0.19 (n=336, per-site estimator).
  Consistent weak positive direction everywhere, never near clinical or preregistered
  significance — the family's honest ceiling is r~0.1-0.2 single-subject, unreproducible
  at gate level. (Behavioral positive control d'->ESWAN r=0.35 remains the only alive cell,
  not an imaging feature.)

## Batch 12 — PREREGISTERED 2026-08-30 (BEFORE any batch-12 fit): the 7 remaining
## theorist-proposed hybrid variants (workflow leida-family-discovery, t4/t5; t1/t2 returns
## were null/dropped by the provider — their directions folded into t3/t6 coverage).

Gate 0 (measured by theorists' pilots, to be re-verified by lead): C' occ-weighted
per-state controllability SB 0.578 raw / 0.315 siteres; D' template-state transition+entry
energies SB 0.843 raw / 0.544 siteres (highest ever measured in this program);
A' fused occ x age interaction 0.463 (siteres 0.456); CGATE/EDIFF/DYNCTRL/EMAP: forecasts
0.42-0.75 (t4 did not run pilots — their Gate 0 must be computed fresh before Gate 1).
- Variants: 4A CGATE (coherence-gated precision contrast, 70d), 4B EDIFF (difference-direction
  energy, 3d, DX-augmented-baseline caveat preregistered), 4C DYNCTRL (windowed precision
  strength mean/CV/persist, 15d), 4D EMAP (18d energy map incl. 7 template targets + 4
  system-contrast), D5-A' (occ x age, 5d, tested as interaction delta), D5-C' (14d),
  D5-D' (15d).
- Gates: identical protocol. Gate 2 familywise null includes ALL batch-12 entrants
  (max-stat; the null prices the whole batch). Targets: T-cohort Inatt/Hyper primary;
  IQ positive control (t6's Gate-0.5 canary idea adopted: any variant failing IQ CV<0.10
  is reported as low-content even if symptom-null).
- STOP: this is the last registered batch of the family; after it, the family verdict is
  final regardless of outcome.

