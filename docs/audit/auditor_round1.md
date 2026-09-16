# Auditor round 1 — exp_r6_lock (adversarial audit, 2026-09-05)

Auditor: independent adversarial audit (read-only on all data/code; light pandas/numpy only in project venv).
Every number below was recomputed by me from disk unless attributed to an existing note. Intermediates are in `/tmp/auditor/` (lit_store.pkl, flux_metrics.csv, hyst_metrics.csv, tail_metrics.csv, var_metrics.csv, handover.csv, fd_grouptest2.csv, fd_tree_lens.csv).

**TL;DR (4 verdicts + 1 urgent flag)**
1. The literal majority-count sign rule is faithful to paper 7 AND immaterial: 1.27% of frames disagree; the k5 dictionary is recovered at |r|=1.000 centroid correlation; the attention effect is unchanged (t=−3.04 → −3.04 fixed-dictionary; −3.15 with a literal-rule-rebuilt dictionary).
2. **"Per-run FD impossible" is REFUTED (in part).** The `fd_rest` tree is an all-rest pick, and a properly edge-aligned FD↔DVARS test (which 01_fd_provenance.py never did — it expected POSITIVE correlation) shows fd_rest is the run-1 pick at group level (paired t=−9.5, p≈1e-17, n=183) and the census-known files reproduce it as positive control (t=−10.7). Per-run run-1 FD is recoverable for ≈700+/931 subjects, and subject-level all-rest FD exists for all 833.
3. FD-trait (defined as the fd_rest mean) transfers: its correlations with life_4 (−.249 vs −.232), age (−.378 vs −.387) and global-occ (+.397 vs +.412) are near-identical inside (314) vs outside (476) the provenance-clean group. The fallback is validated for family-wide use.
4. I pre-tested 6 candidate "hidden-deep" objects on the fresh both-run labels/frames: directed exit/entry flux, transition asymmetry (degenerate), FPN→global handover, continuous axis-projection dwell (hysteresis), dwell-tail statistics, and a paper-6 fitted-operator VAR. Only the **dwell-duration family** (life, nlong8, p90) carries the attention association; everything else is either unreliable (ICC<0.1) or a global-coherence latent in disguise. Details + numbers in §4.
5. Urgent: the per-run `dvars_mean` column of `eigvec_hbn_both_runs_meta.csv` was overwritten by the 17:05 re-run (the v1_dvars copy on disk is byte-identical to the new meta — the 1862-row dvars version is gone). Recompute DVARS inside the locked-test script (cheap) — do not reference the meta for dvars.

---

## 1. FAITHFULNESS AUDIT — 02_hbn_run2_eigvec.py vs 02b_leida_eigvec_hbn.py

Line-by-line comparison (paths: `exp_r6_lock/scripts/02_hbn_run2_eigvec.py`, `exp_leida_xverify/scripts/02b_leida_eigvec_hbn.py`):

| Component | 02b (old) | 02 (new) | Verdict |
|---|---|---|---|
| Band / edge / filter | BAND=(0.02,0.1), EDGE=10, butter(4,[f*2*tr],bandpass), filtfilt axis=0 | identical (lines 20–27 vs 28–39) | identical |
| z-scoring | per-parcel (d−mean0)/std0 with 1e-8 guard | identical | identical |
| Hilbert → dPC | angle(hilbert), M=outer(c,c)+outer(s,s) | identical | identical |
| Leading eigvec | eigh, argmax(abs(w)) | identical | identical |
| lam storage | w[k], next-largest abs(w) | identical | identical |
| TR rule | 1.45 if n_t≥420 else 0.8 | identical | identical |
| loadtxt | skiprows=2 | identical | identical |
| **Sign rule** | `if np.mean(v) > 0: v = -v` (l.38) | `if (v>0).sum() > len(v)/2: v = -v` (l.50) | **intended difference, correct** |
| Sample selection | manifest-driven longest run, **T≥250 filter** | glob all `*_scan_rest_run-*`, **no minimum-T** | deliberate (both runs needed); new meta n_t ∈ {375:1834, 370:78, 252:1, 374:1, 294:1} → min n_frames 232; no 420/750 files match the rest-run glob, so **no TR misassignment exists** (my earlier worry, retracted — checked `exp_r6_lock/results/eigvec_hbn_both_runs_meta.csv`) |
| Pool | Pool(8), chunksize 4 | identical | identical |

**Paper 7 sign convention (verified)**: `exp_leida_xverify/notes/paper7_fulltext.txt` lines 500–506: *"for each eigenvector V1(t), when the number of elements greater than 0 in V1(t) exceeds half of the total number (0.5×N), replace it with −V1(t) to ensure that majority of the elements are negative."* → **the literal count rule matches the paper; the old `mean>0` rule was a deviation.** The new code's strict `>` is also right ("exceeds half" — count=100 exactly does not flip for N=200).

**Quantified disagreement on the OLD cached eigenvectors** (`exp_leida_xverify/results/eigvec_hbn.npz`, all 790 subjects × 280,305 frames; stored frames are mean-rule-signed so mean(v)≤0 always, and on stored frames the two rules disagree ⟺ count_pos>100):
- **Disagreement: 3,548/280,305 frames = 1.27%.** Per-subject: mean 1.3%, sd 1.0%, max 6.8%; 735/790 subjects have ≥1 disagreeing frame.
- By old state: s0 2.1%, s1 2.1%, s2 1.9%, s4(FPN) 1.5%, **s3 (global, all-negative) 0.0%** — as expected, both rules agree wherever the pattern is coherent.

**Would the k5 dictionary materially change? No — measured three ways:**
1. Fixed old dictionary, re-assign literal-rule frames (exact Euclidean nearest-centroid — note centroid norms vary 0.45–0.77, so a `2−2v·c` proxy is WRONG, use full `(v−c)²`): life_4 r=**0.9854** between rules (mean |Δlife| = 0.25 TR = 9% of its SD), occ_4 r=0.9987, s4-run-count r=0.986.
2. Effect: `attention ~ life_4 + age + sex_c + fd(fake)` → old t=−3.041 p=.0030 vs literal t=−3.044 p=.0035; age+sex only: −3.226 vs −3.229 (my fl_perm, 2000 perms, seed 7).
3. **Dictionary rebuilt from literal-rule frames** (k=5, n_init=10, seed 0, same 280,305-frame pool): all 5 centroids match the old ones at **|r|=1.000** (new s0↔old s4 FPN, s4↔old s3 global, s1↔s1, s2↔s0, s3↔s2); life_fpn(new dict) r=0.983 with old life_4; attention t=−3.147 (fake covs), −3.330 (age+sex) — if anything slightly stronger.
→ **Verdict: the sign-rule fix is faithful and immaterial; the locked dictionary built in `hbn_k5_dict_run1.npz` is a valid successor of the old one.** No need to re-run or re-justify anything on the sign rule.

## 2. FD VERDICT CHALLENGE — "per-run FD impossible" is REFUTED (in part)

What the census actually contains (`data/hbn_t2/fd_provenance_census.csv`, 833 rows: sid, picked, should_be, all_paths):
- `picked` = the FULL bucket path that the `data/hbn_t2/fd/` tree holds per subject (verified: picked=movie classes ↔ fd-tree file length ≈750; picked=rest_run-1 ↔ length 375; see /tmp/auditor/fd_tree_lens.csv): movieDM 450, **rest_run-1 314**, movieTP 58, rest 6, rest_run-2 4, movieDM_run-01 1.
- `should_be` = the analyzed run's FD: rest_run-1 (782) / rest (43).
- `all_paths` shows the bucket held BOTH rest runs' FD for ≥304 subjects — so a correct per-run file existed upstream; the on-disk trees simply picked one each.
- The two trees are byte-identical for **361/833** subjects; for the 519 movie-picked subjects the `fd` tree holds a **movie** FD (≈750 rows) while `fd_rest` holds a ≈375-row file.

**The unused path that recovers per-run FD.** The prior verdict rested on `exp_r6_lock/scripts/01_fd_provenance.py`, whose matcher (a) demands a POSITIVE FD↔DVARS correlation ≥0.35 (`CORR_THR=0.35`, verdict 'assigned' only for positive corr) and (b) aligns FD[1:] / FD[:-1] against an edge-trimmed DVARS proxy without trimming FD. Both are wrong:
1. The true within-run FD↔DVARS relation is **NEGATIVE** (CPAC pipeline.yml `Censor: SpikeRegression, FD_J 0.5` suppresses high-motion frames in place — the "scrubbing fingerprint" the session already saw). My edge-aligned recomputation (DVARS proxy = RMS frame-diff of the raw cc200 matrix over frames [10, T−10], FD trimmed to FD[11:11+n]):
   - **Positive control** (131 census rest_run-1 subjects, fd tree = true run-1 FD): corr(FD, DVARS_run1) = −0.171 vs corr(FD, DVARS_run2) = −0.008; paired Δ = −0.163 ± 0.015, **t = −10.7, p ≈ 1.6e-19**; 85% of subjects more-negative for run-1.
   - **Test group** (183 movie-picked subjects, fd_rest tree): corr(fd_rest, DVARS_run1) = −0.141 vs −0.024 (run-2); **paired t = −9.5, p ≈ 1.5e-17**; per-subject |corr| criterion assigns 75% to run-1 (median |corr| margin +0.05, IQR 0.00–0.15).
   - Negative control (fd tree, movie FD): corr ≈ +0.005 / −0.021 — no rest structure, as expected.
2. FD-mean cross-checks: corr(fd_rest mean, DVARS_run1 mean) = −0.753 (census grp) / **−0.667** (movie-picked grp); corr(fd-tree mean, fd_rest mean) = 1.0 vs **0.538** (movie-picked) — i.e., the fd-tree mean is movie-contaminated for those subjects while the fd_rest mean tracks the analyzed run's DVARS strongly.

**Verdict**: (i) per-run **run-1** FD is recoverable on disk for ≈ 314 (census-proven) + ≈ 390 of the 519 (DVARS-margin assignment, |corr| criterion) ≈ **700+ of 931 subjects**; (ii) subject-level **all-rest** FD exists for all 833 (fd_rest tree; ~37 movie-picked subjects hold 750-row fd_rest files — drop or handle those); (iii) the remaining unknown is only run-1-vs-run-2 for the unassigned minority — irrelevant for the locked test, which analyzes run-1 / run-averaged metrics. The "impossible" verdict should be downgraded to "per-subject certain for ~75%, group-level certain for the fd_rest tree".

**FD-trait fallback (as used in the planned locked test) — validated with the exact minimal check:**
- Tautology check on the 314: fd == fd_rest byte-identical for 307/313 census-rest_run-1 files → attention~life_fpn with FD-trait must reproduce the round-2 t=−2.87 there (any deviation = implementation bug).
- **Transfer check (the informative one), measured now**: corr(FD-trait = fd_rest mean, ·) inside vs outside the provenance-clean group: life_4 **−0.249 vs −0.232**; age **−0.378 vs −0.387**; occ(global) **+0.397 vs +0.412**; attention +0.093 vs +0.046. All near-identical → the covariate behaves as the same latent in both groups → validated for the full n≈930. Expected pattern under a GOOD covariate is exactly this (transfer within ±0.05); under a BAD covariate the outside-group correlations would be attenuated/flipped. Additionally the FD-trait↔DVARS relation is strongly negative as the censoring model predicts (−0.67 in the outside group).
- Keep the round-2 314-subject analysis as the provenance-clean sensitivity; and define **FD-trait from the fd_rest tree only** (never the fd tree, which is movie for 519 subjects).

Side effects to fix: the per-run `dvars_mean` column died with the meta overwrite (see TL;DR) — recompute DVARS means per analyzed run inside the locked-test script (01_fd_provenance.py's dvars_proxy, ~5 min for 1915 files, or reuse /tmp/auditor/fd_grouptest2.csv for ~330 subjects as a check).

## 3. LOCKED-TEST ADVERSARIAL

**ICC context (parent's own run, `exp_r6_lock/results/hbn_icc.csv`, n=931, corrected icc_2way — I verified the formula is now the correct SS/MS decomposition and reproduced life_2 ICC=0.148 vs their 0.146 on my labels):** occ_0 0.488, occ_fpn 0.285, life_fpn 0.146, gcoh_mean 0.598, gcoh_sd 0.558, switch 0.081; Spearman-Brown 2-run: life_fpn 0.255→(their table: 0.351 for life_4-old-label; state labels differ between my rebuild and their dict — same object). **The primary metric sits on the least reliable member of the family; run-averaging is mandatory (SB ≈ 2× single-run), not optional.**

**fl_perm.py audit** (`exp_r6_lock/scripts/fl_perm.py`) — mathematically correct:
- Residualize y and target on C(+intercept) via lstsq; permute the residualized target; **refit the FULL design per permutation with fresh lstsq AND fresh `pinv(X'X)`** (lines 42–48, 55–58) — the old `exp_leida_xverify/scripts/04_metrics_stats.py` bug (line 57 computes `XtX_inv` once from the OBSERVED design, line 71 reuses it inside the loop while the design changed) is genuinely fixed.
- Two-sided p with the (b+1)/(B+1) correction: correct. Familywise: **same permutation index applied to all family members** before the max-|t| (line 101–104) — this preserves the cross-metric dependence, which is the right max-statistic construction; smoke test passes (t=6.8 → p=.0010 at true β=0.35; family mode returns sane values; NaN-safe).
- Quantified impact of the old bug (my null simulation, n=825, correlated covariates age↔FD ρ=0.4, 300 sims × 400 perms): at target↔covariate correlation 0.0/0.15 both schemes behave identically (mean null p 0.51/0.49; P(p≤.05)=0.067 both, within MC noise ±0.013); at 0.30 old q05=0.030 vs FL q05=0.037 — the old scheme is mildly liberal only when the target correlates strongly with covariates. → Round-2's p_perm=.0046 is trustworthy to first order; the FL rerun is still the right call for the locked p≈.046.
- Two caveats: (a) FL canonically permutes the reduced-model RESIDUALS of y, not the residualized predictor; both are accepted FL-type schemes and asymptotically equivalent, but since the decision threshold is borderline, run a 1,000-perm dual-scheme agreement check (permute P⊥y residuals instead of x̃) before locking; (b) one RuntimeWarning (sqrt of negative / div-by-zero) fired in my sims — add a finite-guard on t (treat non-finite null t as 0) in fl_perm_test.

**What could flip a familywise p≈.046 — the ≤5 concrete robustness checks (exact definitions):**
- **R1 Motion-covariate triangulation** (now possible given §2): rerun the locked regression three times — (a) per-run FD (census ∪ DVARS-margin-assigned), (b) FD-trait (fd_rest mean), (c) no motion covariates — with covariate set {age, sex_c, spike_frac(FD>0.5 on the same FD series), dvars_mean(analyzed run)}. Decision: sign-stable and |t|≥2.5 in all three; and NO concentration in a single FD-quartile stratum with sign flip in the cleanest quartile (the NYU artifact signature was exactly a Q1-zero).
- **R2 Release/protocol adjustment**: `data/hbn_t2/attention.csv` has `release_number` (R1–R9) for all subjects (joinable via participant_id). Add release dummies (or run within-R1+R2, the largest releases, n≈286) — a vendor-harmonization change across releases could carry the association.
- **R3 Age robustness**: state metrics are heavily age-loaded (occ_global ~ age t=−7.3). Refit with a 3-knot age spline and as three age bands (5–9 / 10–13 / 14–22); require same-sign in all bands; report the attention×age interaction explicitly (it is also proposal P3 below).
- **R4 Dictionary-independence**: rebuild the dictionary on 10 bootstrap frame resamples → require corr(life_fpn variants) ≥0.95 with the locked dictionary and stable effect sign; plus the already-run k=6/7/8 and 7-net-k=5 replications; plus my literal-rule rebuild (centroid |r|=1.000).
- **R5 Influence/n_frames**: leave-25-out t-range; winsorize attention at 5/95%; add n_frames as covariate (life has a mild length bias; range 232–355, only 1 subject <250 after the MIN_T filter).
- Plus the FL dual-scheme agreement check above (cheap, pre-lock).

Design note: define dvars_mean as the mean over the runs analyzed (run-1 for single-run analyses; mean of both for run-averaged metrics) and spike_frac from the SAME fd_rest series as FD-trait so the motion block is internally coherent.

## 4. HIDDEN-DEEP SIGNAL PROPOSALS — with pre-tests (I ran them; most are already falsified)

I pre-tested every candidate object the brief suggested (and two of my own) on the fresh both-run labels/frames, using the parent's dictionary and the old fake covariates for comparability (n=747 complete cases; identical covariates for every row of the table, fl_perm 1000 perms seed 3). ICCs are my own correct-formula two-way ICCs across the 931 both-run subjects.

| Object | ICC (2 runs) | attention t (run-1 / run-avg) | Verdict |
|---|---|---|---|
| life_fpn (hard dwell, parent dict s2) | 0.146 | −2.37 / **−2.48 (p=.012)** | the effect lives here |
| nlong8 = # FPN bouts ≥8 TR | 0.220 | −1.90 / **−2.33 (p=.020)** | real, slightly less |
| p90 of FPN dwell durations | 0.128 | −1.99 / **−2.32 (p=.021)** | real, slightly less |
| nlong5 (# bouts ≥5 TR) | 0.280 | −1.13 / −1.60 | weak |
| nruns (# FPN bouts) | 0.255 | −0.29 / −0.84 | dead |
| exit rate from FPN (N_F·−N_FF)/(T−1) | 0.249 | −0.48 / −1.00 | dead (≈ occ/life, cancels) |
| entry rate into FPN | 0.253 | −0.47 / −0.91 | dead |
| **transition asymmetry exit−entry** | **0.017** | −0.10 | **mathematically degenerate** (closed trajectory: exits=entries ± boundaries) |
| FPN→global rate f2g | 0.170 | −0.19 / −1.38 | weak |
| global→FPN rate g2f | 0.117 | −0.21 / −0.26 | dead |
| hysteresis FPN-axis dwell (enter> P75, exit< P50, global thresholds) | **0.361** | −0.37 / −0.55 | reliable but carries NO effect |
| mean FPN-axis projection mean(V1·ĉ_FPN) | 0.400 | −0.40 / −0.39 | reliable but NO effect (global-coherence-loaded: a(t) median +0.336) |
| burstiness of FPN dwells | 0.016 | +0.89 | dead |
| fitted VAR(1) on 5-dim centroid-projection trajectory: ρ(A) | **0.056** | +1.00 / +2.10 (p=.031) | **unreliable** — ICC 0.06 kills it (and the old PCA-7 version was NYU+/Penn−) |
| same: MC_FPN (paper-6 Eq 10 on |A|/(1+λmax)−I) | 0.013 | +0.99 / +1.54 | dead |
| conditional handover P(global | FPN bout ≥5) | 0.214 (n=360) | −1.82 (p=.065) run-avg | exploratory only, underpowered |
| gcoh_mean / gcoh_sd (reference) | 0.598 / 0.558 | (known: p≈.05/…, motion latent) | reference |

**Reading of the table** (this is the round's most important negative knowledge): the attention association is **specific to the sustained-dwell duration of the anti-global/FPN state** — not to occupancy (occ tested, t=−2.32 old), not to how often you enter/exit (rates ≈ occ/dwell cancel), not to how close the trajectory sits to the FPN centroid on average (the axis projection is a gcoh twin), not to the fitted operator's spectral radius (unreliable), and not to transition asymmetry (degenerate). Also: mean|λ1| (v1frac) run1↔run2 r=0.535 — the global-coherence scalars are the most reliable and the least clinically specific, which is why they keep producing motion-latent artifacts.

### Ranked proposals (effect size × identifiability × compute cost)

**P1 — Sustained-dwell family, run-averaged, with distribution-shape members (highest identifiability, zero new compute).**
(a) Definition: per subject, per run r: the FPN-state dwell sequence D_r = {run lengths of state F in labels_r}; metrics m ∈ {life = mean(D), nlong8 = #{d≥8}, p90 = quantile.90(D)}; subject value = mean over the 2 runs (710 frames total). Locked family: {life_fpn, nlong8, p90, occ_global, gcoh_mean} with Freedman-Lane max-|t| (5 tests, replaces gcoh_sd with nlong8 or keep 6-test family — do NOT grow the family silently; fix it before running).
(b) Why: attention deficit is mechanistically a SUSTAINED-attention deficit (ability to hold task-set engagement over tens of seconds); mean dwell is dominated by high-frequency short bouts (Poisson noise floor), while nlong8/p90 measure the sustained-engagement tail — the exact behavioral construct. The pre-test shows the tail members are independently significant (p=.020/.021) and MORE reliable than mean dwell (ICC 0.22 vs 0.146); run-averaging moves single-run ICC 0.146 → SB 0.255–0.35.
(c) Datasets: HBN run-1+run-2 (931 both-run subjects — frames and labels already on disk); ds000030 after preprocessing; adhd200/Penn anti-phase cohorts CANNOT host it (no FPN state) — do not try.
(d) Falsifiable prediction: with run-averaging at n=930, life_fpn partial ρ should sharpen from −0.167 toward ≈ −0.19/−0.22 (attenuation scaling √(0.255/0.146)≈1.32), t≈−5±1, familywise p<.005; nlong8 and p90 same sign, |t|≥2. Kill/demote rule: familywise p>.05 or sign instability across R1–R5.

**P2 — ds000030 4th-cohort test of the SAME dwell object + dimensional ASRS (the decisive external adjudication; medium compute, already underway).**
(a) Definition: identical object (P1 metrics) on ds000030 after the project's own no-GSR 0.02–0.1 pipeline + CC200 parcellation; categorical ADHD dx (42) vs CONTROL (127) and dimensional asrs_score (asrs.tsv, available for all 169 with BOLD on disk — verified).
(b) Why: it is the only path out of "HBN-only"; ds000030 is paper 6's own dataset; motion covariates are real there (own pipeline), removing the HBN provenance caveat entirely; the ASRS gives a dimensional gradient within a cleanly preprocessed adult cohort.
(c) Datasets: `data/ds0000_rest/` 169 rest BOLD (verified 42 ADHD + 127 CONTROL; the 50 SCHZ + 49 BIPOLAR have participants.tsv rows but **NO BOLD on disk** — diagnostic breadth beyond ADHD-vs-control is NOT currently testable; note as a limitation, do not oversell).
(d) Falsifiable prediction: ADHD < CONTROL on FPN dwell (same direction as attention); d≈0.4–0.5 expected → power ≈0.55–0.70 at n=169 (80% power needs d≥0.62); ASRS ~ dwell ρ≤−0.20 → t≈−2.6, power ≈0.7. Either outcome is informative: if ADHD-vs-control reproduces with real FD covariates, the HBN effect graduates to "two-cohort"; if it flips or vanishes, the HBN effect is capped at "HBN-in-dataset" regardless of the robustness checks.

**P3 — Developmental specification of the dwell effect (age interaction; cheap, pre-registered as a spec test, not a discovery test).**
(a) Definition: extend the locked model with attention×age (centered age) and a 3-band stratified version (5–9 / 10–13 / 14–22); HBN age 5–22 spans the FPN's protracted maturation.
(b) Why: FPN is the last network to mature (mid-20s); ADHD is formally a maturational-delay phenotype (Shaw 2007 cortical-delay ~2–3 y); a genuine control-network dwell mechanism should be strongest where the FPN is still immature, and the age-gradient of global-state occupancy (t=−7.3) is already the pipeline's strongest known-real effect, so the age axis is validated in this data.
(c) Datasets: HBN (both runs); ds000030 (21–50) as the adult complement: the dwell-attention association should weaken/absent in adulthood if the maturational story is right (between-cohort, non-pooled — never pool HBN-vendor with own-pipeline ds000030).
(d) Falsifiable prediction: interaction β(attention×age) > 0 (slope attenuates with age), |t|≈2–3 at n=930 (power ~0.5–0.7 for the interaction — declared exploratory); band estimates monotone in age.
Kill: flat interaction AND flat bands → the effect is age-uniform (still publishable as a spec result).

**Explicitly killed by my pre-tests (do not spend main-agent compute on these):** directed transition rates and TPM asymmetries (degenerate/null, ICC table above); continuous/softmax state-projection dwell and hysteresis dwell (ICC 0.36–0.40 but zero association — the axis is a global-coherence proxy; the reliability gain is real but buys nothing clinically); paper-6 control energy / AC/MC on a fitted reduced-order operator (ICC 0.00–0.06 across runs — the estimator is noise at 355 frames; if ever revisited, needs both-runs concatenated ≥700 frames and a variance-stabilized A estimate); dwell burstiness (ICC 0.016); λ1(t) spectral features (same motion-loaded family as gcoh); HBN movie scans (**verified: zero movie cc200 files on disk** — `data/hbn_t2/cc200` holds only rest run-1/run-2); Penn nback task dictionary (already falsified in findings_log #5); switch metrics (ICC 0.081).

## 5. UNDERUSED-DATA CENSUS

- `data/ds000030/participants.tsv` (272 rows: CONTROL 130, SCHZ 50, BIPOLAR 49, ADHD 43) + `phenotype/asrs.tsv` (asrs_score, n=272; 169 with BOLD) + `phenotype/adhd.tsv` (item-level adhd1–adhd11) + barratt/bart tsvs: dimensional ADHD severity for the 4th cohort; SCHZ/BIPOLAR rows are phenotype-only (no BOLD on disk — specificity claim limited).
- `data/ds000030_rest/`: 169 task-rest BOLD (42 ADHD + 127 CONTROL, ages 21–50) — the 4th cohort; plus `cohort_needed.csv` (download ledger).
- `data/ds000030/sub-7000x/`: 10 subjects with anat T1w + task-rest BOLD (ADHD, duplicated in ds000030_rest) — the only T1s; enough for nothing beyond QC demos.
- `data/ds005899/`: 61 participants, CSST task BOLD (7 subjects × 2 runs on disk) — demo-only for a task-state LEiDA pipeline check; no clinical use at n=7.
- `data/adhd200/adhd200_preprocessed_phenotypics.tsv` (974 rows: Handedness, DX, ADHD Measure, ADHD Index (581 valid), Inattentive (660), Hyper/Impulsive (660), Verbal/Performance/Full4 IQ (890), Med Status (632), QC_Athena/QC_NIAK): **pooled dimensional within-ADHD analysis across 7 sites (site fixed effects) is available and was never run** (only NYU's Inattentive was used); no SWAN/WDIQ columns exist on disk (checked manifest + phenotypics).
- `data/hbn_t2/attention.csv` (2,192 rows × bifactor attention/p_factor/internalizing/externalizing + release_number + full_pheno flag): release-adjusted robustness (R2 above); bifactor specificity already partly used.
- `data/hbn_t2/fd_rest/` (833): now validated as the all-rest subject-level FD + the basis of per-run FD recovery (§2).
- Analysis levels: 7-net dictionary (replicated the effect at k=5, failed at k=4 Dunn optimum), Schaefer 100/300 1mm atlases on disk (`data/atlas/schaefer_2018/`) + `cc200_to_schaefer100.npy`: **a cross-atlas replication of the FPN dwell effect (CC200 → Schaefer-100/300) has never been run** and would answer the adversarial review's "FPN label is pipeline-asserted" caveat.
- `data/pennlead/ptseries_xcpd/` (174 files: 42 run-02 ~380TP, 30 ses-2, run-01 adds + motion tsv): planned repeated-measures reliability item (5) — also the clean-group dx retest with ~2× power (adversarial review recommendation #1, still pending).
- HBN run-2 beyond ICC (943 run-2 series): pre-registered internal replication sample for the locked test (run-1 primary, run-2 confirmation, both-averaged primary estimator).

## 6. TOP-3 ACTIONS FOR THE MAIN AGENT (ranked)

1. **Run the locked test exactly as planned, with three upgrades**: run-averaged life_fpn as primary (ICC mandate, SB≈0.255); motion block = {FD-trait(fd_rest mean), spike_frac(fd_rest), dvars_mean(recomputed per analyzed run)} on the full n≈930, with the 314 provenance-clean + per-run-FD(census∪DVARS-assigned) versions as sensitivity; family fixed at {life_fpn, nlong8, p90, occ_global, gcoh_mean} (add the two tail members BEFORE unblinding, not after). Decision rule: familywise p<.05 → promote to confirmed-in-HBN pending ds000030.
2. **Finish ds000030 preprocessing and run the 4th-cohort dwell test (dx + ASRS)** with identical object definitions, real FD from own pipeline, and the 14-ROI FOV restriction on record (FPN intact). This is the single most decisive external adjudicator available offline.
3. **Before unblinding #1, spend 20 minutes on the three mechanical guards**: (a) recompute per-run DVARS means (lost in the meta overwrite); (b) run the FL dual-scheme agreement check (permute P⊥y residuals vs residualized target, 1000 perms); (c) add release dummies + age-spline to the locked design matrix and verify the FPN-state identity via the Schaefer-100 cross-map (`cc200_to_schaefer100.npy`) so the "FPN" name in the writeup is atlas-verified, not pipeline-asserted.
