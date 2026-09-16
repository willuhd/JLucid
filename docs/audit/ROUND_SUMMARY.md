# ROUND SUMMARY — 2026-09-04 (rounds 1–3) — READ THIS FIRST

## Where things stand (one paragraph)
Papers 6/7 were extracted and used as the foundation (exact pipelines in
notes/papers_6_7_foundation.md). The overnight's failures were distilled
(notes/overnight_postmortem.md). I then built a paper-7-exact LEiDA pipeline (0.02–0.1 Hz
narrow band, windowless phase, leading eigenvector by largest |λ|, majority-negative
convention, k-means with Dunn sweep, occurrence/lifetime/switch) and ran it — plus the
paper-6 controllability family — over every usable dataset on disk:
adhd200 (965 subj, 8 sites, full use), pennlead/Penn LEAD (104 rest + 104 nback), HBN CPAC
(790 subjects). ds000030 (10/267 files) and ds005899 (14 files) are too partial to use.

## What works (pipeline validation)
- The pipeline is healthy: leading eigenvector explains >50% variance on 100% of frames in
  all datasets; HBN states reproduce paper 7's canonical repertoire (global/VIS/DMN/SMN-VAN/FPN);
  strong canonical age effects in HBN (global state ↓ with age t=−7.3, DMN ↑ t=+7.0, Bonf p≤.0022);
  split-half occupancy reliability ≈0.8. Where a signal CAN be detected, it is detected.

## The two genuine (within-dataset) clinical effects found this session
1. **NYU (adhd200), dx categorical — modal controllability of Visual/Limbic/SalVentAttn**:
   ADHD > TD, MC_Vis perm p=.0010 (Bonferroni-14 = .014), d≈0.44; robust to FD<0.5mm,
   medication, leave-25-out (t 3.11–3.29), both subtypes.
2. **HBN, attention factor — lifetime of the FPN state**: perm p=.0032 (Bonf-11 = .035),
   ρ=−.122; split-half sign-stable 10/10; replicates in the 7-net dictionary at k=5
   (p=.0090, ρ=−.103) though not at the k=4 Dunn optimum.

## Why neither is THE cross-dataset signal (the mathematical explanation)
Both effects are manifestations of a single latent L ("level of phase organization / FC
spectral concentration": gcoh, gcoh_sd, MC, AC all load on it, r≈0.6–0.8). L is
measurement-dependent: it tracks motion (r≈+0.5) and the vendor preprocessing (Athena vs
CPAC vs XCP-D; global-signal handling). The dx↔L association is + in NYU, 0/− in Penn (Penn
clean-group gcoh_sd is SIGN-OPPOSITE, t=−2.66 p=.0086), and 0/− in the other 4 adhd200 sites
(Pittsburgh's nominal p=.0005 is 4-ADHD leverage). A non-invariant latent cannot carry a
cross-dataset claim — six independent metric families all show the same NYU(+)/Penn(−) pattern
(including a data-FITTED dynamics operator, VAR(1) on the eigenvector trajectory: NYU +1.75,
Penn −0.69). Kuramoto generative modeling (paper 7 Eqs 9–17, machinery validated on synthetic
data) fails the identifiability gate on real data (R²≈0.01 vs floor ≈0.05) — the "fitted
dynamics" hypothesis from the postmortem is now closed at ROI level too.

## Decision needed from you (pick any)
A. **Authorize downloads** (changes the no-download rule): (i) Penn LEAD rest run-2 + XCP-D
   derivatives (ds007116/ds006779, CC0 — no DUC needed, see notes/pennlead_ev_provenance.md)
   to double Penn rest power; (ii) full ds000030 (~2.6 GB, paper 6's own dataset, ADHD n=43)
   to test the NYU-MC/gcoh families in a 4th cohort with MY OWN uniform preprocessing.
B. **Stay on-disk**: I continue with the remaining micro-cells (TPM/dwell-exponent variants,
   HBN k-sweep robustness, within-NYU subtype/Med decompositions) — low priors, honest.
C. **Reframe the goal**: accept the two within-dataset Bonferroni effects + the
   non-invariance explanation as the outcome (no cross-dataset verification claim possible
   with these pipelines; would need harmonized preprocessing, i.e., raw BOLD reprocessing
   which only exists for ds000030-partial/hbn-CPAC).

All numbers and receipts: notes/findings_log.md; review of the two hits is running
(notes/adversarial_review_hits.md when done).
