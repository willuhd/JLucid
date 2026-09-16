# Round 1 verdict ledger — exp_r6_lock (2026-09-05, ~21:30)

## What this round set out to do (user-approved in-order plan)
(1) fix flagged inference/measurement bugs, (2) HBN run-1/run-2 ICC, (3) one locked
familywise test of the HBN FPN-dwell lead, (4) ds000030 4th-cohort adjudication,
(5) Penn reliability. All executed except (5) (pending) and the final ds000030
contrast (ADHD wave prepping).

## 1. Bugs fixed (with verification)
- **Freedman–Lane permutation** (fl_perm.py): permute residualized target, refit full
  design per perm, fresh pinv. Dual-scheme check (x-res vs canonical y-res):
  t=-2.11 p=.029 vs t=-2.11 p=.030 — schemes agree. Old 04 bug confirmed real but
  small (auditor: liberal only at target-cov r=0.30).
- **Literal majority-sign rule**: matches paper 7 verbatim ("number of elements
  greater than 0 exceeds half... replace with -V1(t)", paper7_fulltext.txt:500-506).
  Disagreement with old mean-rule: 1.27% of frames; dictionary recovery |r|=1.000;
  immaterial.
- **HBN controllability network-name bug**: confirmed (only Limbic/Default ever
  ran); fixed by name translation — NOT yet re-run (deprioritized: the MC family
  is artifact-retired; run only if the ledger needs completeness).

## 2. ICC (n=931 both-run, literal-rule dictionary)
occ_global .49 | occ_fpn .29 | **life_fpn .146** | gcoh_mean .60 | gcoh_sd .56 |
switch .08. Lifetime = least reliable member; run-averaging mandatory (SB .255).
Continuous gcoh scalars are the most reliable traits (.56-.60).

## 3. FD provenance — recovered (auditor)
fd_rest tree = all-rest pick; the true FD<->DVARS relation is NEGATIVE (CPAC
SpikeRegression censoring). Edge-aligned test: census control t=-15.5, test group
t=-16.1. Per-run FD: 555 run-1 + 191 run-2 assignable; FD-trait (fd_rest mean)
validated as covariate (transfer check in/out of clean group ~identical).

## 4. LOCKED TEST v2 VERDICT (n=747, family {life_fpn, nlong8, p90, occ_global,
gcoh_mean} run-averaged, covs age+sex+fd_trait+spike+dvars, 10k FL max-|t|)
- **Primary life_fpn: t=-2.11, familywise p=.1129, partial rho=-.056 — FAILS the
  prespecified p<.05 rule.**
- No-motion block: t=-2.67, p_fam=.025 (raw association exists).
- Per-run FD block (n=555): t=-1.33. Run-2 alone: t=-1.40 (run-1-only effect).
- FD strata: Q1 -2.40 / Q2 -0.90 / Q3 -2.16 / Q4 +1.99 (sign flip in the
  highest-motion quartile). Youngest age band (5-9y): t=-0.28 (null).
- Release dummies / age cubic / n_frames / leave-25-out: no change.

## 5. Cheap discriminators (the round's key new knowledge)
- **D1 k-sweep: FAILS.** life_fpn t = -2.11 (k5) -> -1.97 (k6) -> -1.09 (k7) ->
  -1.06 (k8). The effect is k=5-specific — state-fragmentation signature. (The
  previous session's "replicates at k=6/7/8" claim does not survive corrected
  inference.)
- **D2 Schaefer-100 cross-atlas: FAILS.** No FPN-positive state emerges in the
  Schaefer-100 geometry (Cont centroid mean -0.009); nearest-analog dwell
  t=-2.06, p_fam=.1285. (Map recomputed directly on the HBN atlas — the stored
  cc200_to_schaefer100.npy is provenance-ambiguous and was NOT used.)
- FPN identity itself IS atlas-verified (21/24 HBN FPN parcels -> Schaefer-300
  Cont majority) — the label is real; the EFFECT does not transfer.
**Honest posterior on the HBN lead after D1+D2+motion+run2+band evidence:
P(biological) ~ 5-15%.**

## 6. Measurement-validity discoveries (cross-cutting)
- **The two "CC200" atlases are DIFFERENT parcellations** (adhd200 4mm vs HBN
  3mm: same-id centroid distance 83mm; Hungarian match 12.1mm; identity 4/190).
  All previous cross-dataset dictionary transfers between adhd200 and HBN were
  anatomically unfounded. The locked dictionary is HBN-internal — unaffected.
- HBN FD-trait <-> per-run DVARS correlation is NEGATIVE (-0.28..-0.41): the
  scrubbing fingerprint (high-motion subjects get smoother series). Vendor
  censoring distorts motion covariates in vendor-preprocessed data.

## 7. ds000030 (4th cohort) status
- 88/169 prepped (67 CONTROL + ~21 ADHD). Pipeline: BOLDRigid MC + N4 + SyNRA to
  HBN CC200 atlas (dice .79, visual QC ok) + native-space parcellation + real FD
  from own motion estimates. ~124s/subject serial; total throughput core-limited
  (~29 subj/h no matter the parallelization — measured 3 ways).
- Regime: intact (global state 65% of frames; modular states 6-12% each;
  bandpassed FC med 0.40, fracneg .05 — more global-dominated than HBN).
- n=76 first look: artifact-check ~0 (gcoh_sd t=-0.65, lam_max t=-0.02) — the
  clean-pipeline prediction holds. dx contrast at 9 ADHD = plumbing only
  (expected t~1.3 even under H1). ASRS within-controls: null (range-truncated).
- Remaining: 33 ADHD (in flight, ~55 min) -> definitive dx test 42v~76
  (t~2.0-2.3 expected under H1, power ~50-60%) -> +60 CONTROL for the fully
  powered test (55-70%).

## 9. Cheap-tier discriminators (ran 2026-09-05 late, corrected FL inference)
- **A1 V-smoothing (HBN run-1, same dictionary)**: attention~life_fpn t = -2.32
  (k=1) -> -2.34 (k=5) -> -0.76 (k=10) -> -1.41 (k=20). The clinical variance
  lives at the 5-10 frame (4-8s) bout scale; 10-frame aggregation destroys it.
  Against a stable-trait account (smoothing should help), for a
  discretization-scale-sensitive object.
- **D1 polarity object (anti-phase-native metastability)**: NYU dx (n=257)
  |t|<=1.36; Penn clean-group (n=49) |t|<=0.34. The regime's own native object
  carries nothing.
- **D2 pooled adhd200 dimensional**: only 203 usable Inattentive rows
  (Peking_1 123 + KKI 80; the auditor's n=660 is not in usable form on disk).
  Inattentive ~ {gcoh, gcoh_sd, MC_Vis} + site effects: all |t|<=0.32. No
  symptom gradient anywhere in adhd200 for the scalar families.
Combined with D1(k-sweep, k=5-specific) and D2(schaefer, no FPN state): five
independent cheap falsifications, all negative-leaning. Honest posterior on
the HBN lead: 3-8%.

## 11. Penn XCP-D repeated-measures (plan step 5, completed 2026-09-05)
87 XCP-D runs extracted with the same scalar pipeline (guarded for dead parcels).
- Vendor-vs-XCPD within-subject rank-consistency (n~23 repeated): gcoh r=0.126,
  gcoh_sd r=0.196. Compare same-pipeline HBN ICC 0.60 -> the anti-phase object
  is pipeline-measurement, not trait (pipeline x run confounded, but the gap
  0.60 -> 0.15 implicates the pipeline change).
- Clean-group ADHD vs TD retest, run-averaged where both runs exist (n=46 obs):
  gcoh_avg t=-0.91 p=.37; XCP-D-only t=-1.27; original-only t=+0.74. Null to
  weakly negative. The Penn dx evidence stays non-positive with doubled
  repeated-measures precision.
- Plan item 5 COMPLETE.

## 12. HBN control analyses CLOSED (2026-09-05 late, corrected FL + real FD block)
- **7-net dictionary**: FPN state exists with the cleanest network contrast seen
  anywhere (Cont +0.340) -> dwell t=+0.02, p=.98. The overnight's 7-net k=5
  'replication' (p=.0090, buggy perms + fake covariates) is REFUTED. 7-net k=4
  (Dunn optimum) has no FPN-positive state at all. A cleaner FPN state shows
  less, not more — against a physiological FPN-dwell mechanism.
- **HBN controllability all-7-nets** (paper-6 construction, name bug fixed,
  n=779, familywise over 14): attention |t|<=0.97; p_factor best AC_DorsAttn
  t=+2.25 p_fam=.13. The MC_Limbic whisper is dead (t=+1.81, p_fam=.31).
  No network controllability signal exists in HBN.
- **Penn polarity clean-group** (n=49): |t|<=0.34. **Pooled adhd200 Inattentive**
  (n=203 usable): |t|<=0.32.

## 10. Next-game tree (user-requested forward program; untested members ranked)
Paper-7-adjacent: V2 dictionaries; amplitude-envelope LEiDA; sub-band LEiDA;
personalized dictionaries + match quality; dwell-distribution exponent;
discovery/validation half-split; windowed (coarse-grained) LEiDA.
Paper-6-adjacent on fitted dynamics: factor-model/DMD-stabilized A + control
energies, B-specific (stimulate-FPN) target controllability, state-MDP
steering energy (requires a validated fitted pipeline — a project, not a mini
analysis). All need explicit authorization given accumulated evidence.

## 8. Decision state
- The ds000030 test is now a CLOSURE test, not a likely confirmation: P(positive)
  ~ 5-10% given the degraded HBN prior.
- If null (expected): the LEiDA-dwell family is exhausted across all on-disk
  cohorts (HBN fragile-and-specific, ds000030 null, adhd200/Penn structurally
  incompatible) -> recommend goal blocked per rule 5 with this ledger.
- If positive: two-cohort revival, reassess with the control wave + Penn item.
