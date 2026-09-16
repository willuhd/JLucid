# Auditor round 2 — CODE audit (adversarial, synthetic-proof, 2026-09-05)

Role: break the ledger's statistical conclusions by finding code bugs, each proven
with synthetic data. Workspace: `/Volumes/thinkplus/Code/JLucid`.
Rules obeyed: READ anything; WROTE ONLY this report +
`exp_r6_lock/results/audit2/` scratch (`a_fl.py`, `b_reg.py`, `b_reg2.py`,
`b_real.py`, `c_assign.py`, `d_icc.py`, `e_offset.py`, `e_flip.py`, `f_fast.py`,
`f_kmeans.py`, `g_hygiene.py`, plus `a_fl.npz`). No downloads, no new data, no
long pipelines (heaviest step: one k=7 KMeans rebuild ~minutes, explicitly
authorized by target F).

Context read first, in order: `notes/FINAL_LEDGER.md`, `notes/auditor_round1.md`,
`exp_leida_xverify/notes/papers_6_7_foundation.md`.

## TL;DR

I could NOT break: the inference engine (A), the registration direction (B), the
assignment indexing (C), the ICC formula (D), the FD-recovery alignment in any
verdict-changing sense (E), any of the 4 recomputed ledger numbers (F), or file
hygiene (G). The ledger's quantitative claims I tested all reproduce.
New minor findings (none change any ledger verdict): (1) HBN `gcoh` is on a ~60
scale (`|λ|/2`) while ds30 `gcoh` is a true fraction (`|λ|/N`) — cross-cohort
gcoh comparison invalid, within-cohort inference unaffected; (2) ds30
"all ROIs ≥30 voxels" is overstated — 19/115 subjects have a sub-30-voxel ROI
(min 2), 3 subjects are missing 1–3 labels; (3) `ds30_prep_meta.csv` holds only
the last batch (10 rows for 115 prepped subjects); (4) FD offset 11 is
first-principles-correct but empirically 12–13 separate slightly better
(likely scrubbing/filter smear) — verdict-flip rate vs offset 11 is ≈X%
(see §E), no sensitivity result changes; (5) the half-split `t=-1.98` has no
on-disk code provenance, but my two independent half-splits bracket it
(-1.56 / -2.25); (6) `fl_family_test` has no finite-guard on null t's (unlike
`fl_perm_test`) and silently complete-cases the whole family on any NaN —
documented constraint, benign here.

---

## A. INFERENCE ENGINE (`scripts/fl_perm.py`) — VERDICT: CONFIRMED

Synthetic validation (`audit2/a_fl.py`, null: `y = C@b + noise`, `x ⫫ y | C`,
correlated covariates, `n=800`):

- **A1 null uniformity** (`fl_perm_test`, 220 sims × 399 perms): `mean(p)=0.518`
  (expect 0.50), `P(p≤.05)=0.0318` (expect 0.05; MC se≈0.015 → 1.2σ, slightly
  conservative, not liberal), KS vs uniform `D=0.0455, p=0.736`. No miscalibration.
- **A1b y-res scheme** (60 sims): `mean(p)=0.495`, `P(≤.05)=0.0500`, KS `p=0.966`.
- **A2 familywise global null** (5 correlated metrics, 120 sims × 399 perms):
  FWER=`0.0583` (expect ≈0.05, MC se≈0.02). Error control holds with the
  shared-permutation max-|t| construction (`fl_perm.py:101-104,156`).
- **A3 known effect** (`y=0.3x+C@b+noise`): reported `t=9.356342` vs analytic OLS
  `t=9.356342`, diff `0.0e+00`. Matches to 6 decimals (spec: 3).
- **A4 y-res reconstruction**: `mean(y_res)=4e-16`, `y_recon+resid ≡ y`
  (`True`), t_obs identical across schemes (`9.3563` vs `9.3563`). The
  `y_recon + permuted-residuals` refit on the observed design is the canonical
  Freedman–Lane reduced-model scheme (fixed C, exchangeable residuals under H0).
- Round-1's verifiable claims re-checked: fresh `pinv(X'X)` per permutation
  (`fl_perm.py:42-48,55-58` — the old frozen-inverse bug is gone); two-sided
  `(b+1)/(B+1)` (`:61`); finite-guard present in `fl_perm_test` (`:49,:60`).
  Minor gap: `fl_family_test.fit_ts` (`:137-148`) has NO finite-guard — a
  degenerate (e.g. constant) metric yields garbage-but-finite t via min-norm
  lstsq rather than NaN (tested: constant metric → t=0.231, p=0.975, max-null
  100% finite). Benign here; recommend one-line guard for safety.
- No miscalibration ⇒ no corrected-recompute of attention~life_fpn needed.

## B. ds000030 REGISTRATION/PARCELLATION DIRECTION (`05_ds000030_prep.py`) — VERDICT: CONFIRMED (direction correct; validation claims overstated, see B3)

- **B1 direction proof** (`audit2/b_reg.py`, `b_reg2.py`): my empirical toy
  registrations did NOT converge (mutual-information on synthetic blobs;
  Translation returned the same file for fwd/inv and a wrong shift — reported
  honestly as INCONCLUSIVE, not as evidence). The decisive proof is the ANTsPy
  source contract (`.../ants/registration/registration.py:172-176`):
  `fwdtransforms`: "Transforms to move from **moving to fixed**",
  `invtransforms`: "Transforms to move from **fixed to moving**"; and
  `apply_transforms(fixed, moving, ...)`: "moving image to be mapped to **fixed
  space**". The code (`05:53-56`) registers `fixed=atlas, moving=n4_native`,
  so `invtransforms`: atlas→native, and applies them with `fixed=n4,
  moving=atlas` — resampling atlas labels into the native BOLD grid. **That is
  the correct direction** (the standard bring-atlas-to-subject pattern). A
  flipped direction would scatter labels across the FOV.
- **B2 bincount averaging** (`05:60-66`): synthetic 4×4×4×5 test —
  bincount-over-flattened-labels vs independent boolean-mask means agree to
  `0.00e+00`. Voxel order preserved (`ravel` on both data and labels). CORRECT.
- **B3 real-data spot-check** (2 subjects, `audit2/b_real.py`): an independent
  code path (no bincount, explicit `corrcoef`) reproduces cached `fc_mean`
  (0.481792/0.486071), `fracneg`, `tsnr` (148.1/154.4), and `dvars`
  (2.919157/3.551091) **to 6 decimals**. FD means (0.122/0.082) and
  `tsnr~150` match the ledger. A full `nilearn.resample_to_img` re-warp was
  correctly out of scope (would require re-registration = forbidden long
  pipeline); the doc-contract proof + internal-consistency proof jointly carry
  the verdict.
- **B3 caveats (NEW, minor, no verdict change)**:
  - Coverage: `ds30_prep_meta.csv` covers only the last batch (**10 rows for
    115 prepped subjects** — idempotent-resume overwrite in `05:110`, real
    hygiene bug). Across all 115 `ds30_tc/*.npz`: `n_live` min 197 (3 subjects
    missing 1–3 labels: sub-10692/-70061/-10429), `cov_min` min **2**, and
    **19/115 subjects have a sub-30-voxel ROI**. The ledger's "all ROIs ≥30
    native voxels" (§1) is overstated; median behavior (cov_med ~36–149) is
    fine and the dx contrast is null regardless, but the validation sentence
    needs correction.
  - One subject (sub-10524) has `n_frames=108` vs 132 elsewhere (short run) —
    correctly handled downstream by per-subject means; noted only.

## C. DICTIONARY ASSIGNMENT — VERDICT: CONFIRMED

- **C1 off-by-one** (`audit2/c_assign.py`): synthetic 4-state/6-label dict with
  known missing label — `dictz[:, roi_ids-1]` (`08:101`) recovers planted
  labels `[0 1 3]` exactly; forgetting `-1` raises hard `IndexError` (0-based
  proof), and a silent-case demo (5-col dict, in-bounds shift) gives different
  labels (`[2 2]` vs `[3 3]`, `differ=True`), proving the `-1` is load-bearing
  where silent. Column order also verified: `08:job` stores `tc` in `live_ids`
  order and `V` inherits it, so `roi_ids-1` selects matching dictionary
  columns. CORRECT.
- **C2 vs `KMeans.predict`**: manual full `(v−c)²` (the `03:100-102` einsum
  path) vs `sklearn predict` agree `1.0000` (raw and co-assignment) on 50 real
  frames. Centroid norms vary (0.761/0.478/0.452/0.467/0.484), confirming the
  full-square form is required and used — the `2−2v·c` proxy would be wrong.
- **C3 metric-row recompute**: sub-NDARAA948VFH both runs — occ_2, life_2,
  switch, gcoh reproduced **exactly** (6+ decimals) from stored `V/lam`.
- **NEW minor bug (no verdict impact)**: `gcoh` scale differs by construction
  between cohorts — `03:47` divides by `lam.shape[1]` (=2, → ~60 scale, see
  `hbn_state_metrics_bothruns.csv` gcoh_mean≈60.4) while `08:113` divides by
  `V.shape[1]` (=N, true fraction ≈0.65). Within-cohort t/ICC/familywise
  statistics are scale-invariant, so no number changes; but any cross-cohort
  gcoh comparison would be invalid. Fix: divide by N (or by trace) everywhere.

## D. ICC FORMULA (`03:54-69`) — VERDICT: CONFIRMED

Synthetic variance-components test (`audit2/d_icc.py`, `n=20000`, vs=1.0,
vr=0.25, ve=1.5, raters ±0.5): got `ICC21=0.3342` vs analytic `0.3333`
(diff 0.0009), `ICC31=0.3974` vs `0.4000` (diff 0.0026); noiseless case gives
exactly (1,1). (Note for reproducibility: with k=2 fixed raters at ±a, the
Shrout–Fleiss denominator carries `2·vr`, i.e. analytic ICC21=vs/(vs+2vr+ve) —
my first analytic pass used `1·vr` and "mismatched" by 0.056; the formula was
right, my expectation was wrong. Corrected and documented.)
Ledger spot-check from raw `hbn_state_metrics_bothruns.csv`: life_2
`ICC21=0.146`, gcoh_mean `0.598`, switch `0.081` — **exact match** to
`hbn_icc.csv` and the ledger (life .146, gcoh .60). Reliability claims stand.

## E. FD RECOVERY OFFSET (`06_fd_recovery.py:35-43`) — VERDICT: CONFIRMED (11 correct in principle, suboptimal by ~1–2 frames in practice, no verdict impact)

- **E0 derivation**: `dvars_proxy` diffs `arr[10:-10]` → `dv[i]` = motion
  between raw frames `10+i → 11+i`; Power `FD[t]` = displacement into frame
  `t`; FD file length = T. Correct segment is `FD[11:11+n]` — **offset 11,
  QED** (`audit2/e_offset.py`).
- **Empirical sweep** (486 both-run subjects: 291 census-known + 194 test):
  | off | known t | test t | known acc(|.|) | known acc(c1<c2) |
  |---|---|---|---|---|
  | 9 | −15.0 | −9.6 | 0.753 | 0.804 |
  | 10 | −15.2 | −9.7 | 0.753 | 0.821 |
  | **11 (used)** | **−15.5** | **−9.9** | **0.790** | 0.808 |
  | 12 | −16.1 | −10.5 | 0.773 | **0.828** |
  | 13 | −16.7 | −11.1 | 0.773 | 0.825 |
  Offsets 12–13 separate slightly better by |t| on both groups (likely
  bandpass/scrubbing smear of the motion signal across frames), but assignment
  accuracy peaks at 11 (|.| rule, the rule the code uses) and the accuracy
  spread is ±2pp. So 11 is the principled choice and within noise of optimal.
- **Impact quantification** (`audit2/e_flip.py` + R1a recompute, full
  746-subject cache; my off-11 run-1 count = **555, exactly matching the
  ledger's "555 run-1 assignable"**):
  verdict-flip rate vs offset 11: off-10 **8.2%**, off-12 **9.7%**, off-13
  **16.4%** — alignment is uncertain at ±1 frame (filter/scrub smear), honestly
  reported. Recomputed R1a-like sensitivity (run-1 metrics + run-1 FD block,
  own code): off-11 `n=555, t=-1.33` (**exact reproduction** of the ledger) vs
  off-12 `n=551, t=-0.78`. The point estimate ranges −0.78…−1.33 across
  defensible offsets — **all null, all far from the |t|≥2.5 guard**, and the
  shift goes toward zero at the empirically-better offset, i.e. against (not
  for) the lead. The headline (FD-trait block, unaffected by offsets) stands.
  The recovery stands; report the −0.78…−1.33 range as a robustness footnote.

## F. LEDGER SPOT-RECOMPUTE (own minimal code, no imports of their functions) — VERDICT: ALL MATCH

| # | Claim | Mine | Result |
|---|---|---|---|
| F1 | locked v2 primary life_fpn `t=-2.11, p_fam=.1129, n=747` | `n=747`, `t=-2.11`; own FL familywise (2000 perms) `p_fam=0.1059` (ledger .1129; MC se≈.007 — consistent) | **MATCH** |
| F2 | k-sweep `k=7 life t=-1.09` (full rebuild, seed 0, n_init 10, 344,741-frame pool) | `n=747`, `t=-1.10` (FPN s6, FPN-mean +0.016 — weak state, as ledger says) | **MATCH** (0.01 = rounding/float32 noise) |
| F3 | ds30 checkpoint life_fpn `t=+0.62` (n=115, 42 ADHD) | `n=115`, `t=+0.62` | **MATCH** |
| F4 | half-split B-dict→B-test `t=-1.98` | No on-disk code produces a half-split (provenance gap — likely interactive). My independent splits: A-dict→B-test `t=-1.56` (n=370), B-dict→A-test `t=-2.25` (n=377) | **CONSISTENT** (sign-stable, sub-threshold range reproduced; exact −1.98 untraceable) |

Also reproduced en passant: nlong8 `t=-1.91`, p90 `t=-1.96`, occ_global
`t=-0.25`, gcoh `t=+0.43` (F1 family, all within rounding of the locked run).

## G. META-FILE HYGIENE — VERDICT: CONFIRMED (clean, two notes)

- **Method-vs-column collisions**: precise grep for attribute-form
  `.(sub|file|count|mean|sum|min|max|std|var|age|sex)` over all scripts — every
  hit is numpy `.shape`/`.mean()` calls or safe names (`df.verdict`,
  `met.dx`, `man.sid` — none collide with DataFrame methods). Notably
  `part['sub']`/`met.merge(on='sub')` use bracket form throughout. The old bug
  class does not recur. (Amusingly it bit MY audit script: `dsm.sub` returned
  the `DataFrame.sub` method — fixed to `dsm['sub']`.)
- **Merge keys**: HBN manifest `sid` (`sub-…_ses-1`) → stripped `s` joins
  metrics/fd sids (`sub-…`, no suffix) — verified consistent; ds30
  `participant_id` (`sub-10159` on all three sides) merges losslessly
  (115/115, dx missing 0, asrs missing 0).
- **Run-label swap**: metrics `n_frames` vs meta `n_frames` agree `1.0000`
  (n=972 run-1, n=943 run-2); labels-vs-V length consistency 781/781.
- **Sign-rule consistency**: extraction (`02:50`, `08:55`) both use the literal
  majority-count rule; assignment is sign-free Euclidean. No inconsistency.
- Notes: (i) `ds30_prep_meta.csv` incomplete (10 rows / 115 subjects, §B3);
  (ii) `03:139` contains a dead placeholder line (`both = [... if False]`) —
  unused (`n_pairs` computed properly at `:142`), remove on touch.

---

## Ranked required fixes (all minor; none change any verdict)

1. **Correct the ds30 validation sentence** (FINAL_LEDGER §1) to "median
   coverage ~36–149 voxels; 19/115 subjects have a sub-30-voxel ROI (min 2),
   3 subjects missing 1–3 labels" — the "all ROIs ≥30" claim is falsified by
   `results/ds30_tc/*.npz`. No analysis change (null stands).
2. **Unify the `gcoh` scale** (`03:47`: `/lam.shape[1]` → `/N`) so HBN and
   ds30 `gcoh_mean` are comparable fractions; re-verify ICC table invariant
   (it will be — scale-free).
3. **Accumulate `ds30_prep_meta.csv`** across resume batches (`05:110-111`)
   instead of overwriting with the latest batch only.
4. **Add finite-guard to `fl_family_test.fit_ts`** (mirror `fl_perm_test:49`);
   delete dead `03:139` line.
5. **Record half-split provenance** (script + seed) if the `t=-1.98` number is
   cited again; my `audit2/f_kmeans.py` F4 block can serve as the template.
6. Optional: FD-sensitivity range footnote (R1a `t=-1.33` at offset 11 vs
   `-0.78` at offset 12) to close the alignment question permanently.

## Closing statement

**After this audit, the ledger claims I could NOT break are:** the FL
inference engine and every p-value built on it (null-uniform, familywise
control, t-to-3-decimals); the ds000030 atlas direction (contract-proven
correct, parcellation internally consistent); the restricted assignment and
all HBN metric rows; the ICC reliability values (life .146, gcoh .60);
the FD-recovery alignment (offset 11 principled, verdicts offset-proof); the
four recomputed headline numbers (F1/F2/F3 exact, F4 bracketed); and merge/run
hygiene throughout.
**The ledger claims that FALL are:** none as verdicts — but two validation
sentences fall as stated and must be reworded: (i) "all ROIs ≥30 native
voxels" (19/115 violations, min 2) and (ii) by implication any cross-cohort
gcoh comparison (scales differ ~100× by construction). The central verdict —
marginal, measurement-fragile, NOT confirmed — survives this audit intact;
if anything it is strengthened, since the most breakable machinery held.
