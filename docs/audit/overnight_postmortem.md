# Overnight LEiDA program postmortem (exp_overnight_leida, R1–R130)

Written by the postmortem-analysis subagent (read-only except this file).
Sources: `exp_overnight_leida/` — README, STALL_STATE, OVERNIGHT_FINAL_v3/v4, V5–V8 deltas,
R14/R30 reports, all phase logs (01–25), 00_recovered verify logs, JSON result files.
Bottom line up front: **27 courses + 1 extension, 0 triples.** The program's own conclusion
(power memo R78) is a *bounded null*: cross-cohort LEiDA/NCT dynamics effects ≥ d≈0.2
(r≈0.15 ADHD-200, r≈0.10 HBN) are EXCLUDED; the NAS minimax search over the whole enumerated
metric family priced itself at search-perm p=.92 — worse than the average chance-max.
Every p<.05 nominal died under audit, cross-dataset test, or sign-coherence.

---

## What was tried (compact table)

| # | Family / idea | Dataset | Contrast | Result (effect / p) | Verdict |
|---|---|---|---|---|---|
| 1 | 7-net LEiDA occupancy, frozen k=5 (paper-7 faithful) | ADHD200 site-5 pilot → Penn rest (n=85) → HBN (n=80) | Inatt / ESWAN-inatt (inverted) / attention | s3: +0.22 (pilot n=59) → +0.12 *wrong-sign* p=.26 / +0.02 p=.86; s1 −0.16/−0.21 flips to +0.14 in HBN | DEAD (R1–R2) |
| 2 | Within-subject rest→nback occupancy/dwell/switch change | Penn (n=71 paired) | Δ vs ESWAN + dx | d_switch +0.21 p=.094 (inatt), +0.23 p=.065 (total); d_occ all n.s. | DEAD as triple; kept as whisper (R3) |
| 3 | Switching/entropy levels | ADHD200 / HBN / Penn | Inatt / attention / load-change | ADHD −0.11, HBN +0.10, Penn +0.21 — "switching-good" 3/3, all n.s. | DEAD; full-n HBN flips sign (R3, R6) |
| 4 | NCT transition energy (paper-6 exact: A=\|FC\|/(1+lmax)−I, B=I, T=1, Gramian) | all 3 rest | centroid targets, one-hot nets, load-change dE | E0–E4 all p>.11, signs incoherent; one-hot r=.03–.07 (global Gramian scale); dE4 −0.23 p=.066 Penn-only | DEAD; E later killed by r=.97 mean-FC audit (R4, R11) |
| 5 | ROI-level LEiDA states (k=6/8, per-dataset discovery) | ADHD200 190p, HBN 200p, Penn 400p | Inatt / attention / ESWAN | ADHD pilot k8 s6 −0.43 (n=29, 26 states) → full-n all \|r\|<0.06; HBN s4 VIS −0.22 p=.034 nominal (Holm ~.20); Penn ARI .30 unstable; Yeo profiles \|v\|<0.05 → matching ill-posed | DEAD as triple (R5–R10) |
| 6 | Preregistered VIS-state spec (spec_v01: occ, dwell, E_enter_VIS) | ADHD n=336, HBN n=833, Penn n=85 | Inatt / attention / ESWAN + p_factor | VIS-occ 3/3 same sign (worse): −0.007 p=.88 / −0.040 p=.25 / −0.144 p=.18 — 0/3 pass; dwell all \|r\|<0.06; E_enter_VIS HBN +0.084 p=.013 but **r=.97 with mean FC** | NO CLAIM per rule (R11–R12) |
| 7 | Slow-5 / slow-3 bandpass LEiDA | all 3 (TR-aware FFT) | switching + VIS-occ | slow-5: +0.02/−0.12/−0.04; slow-3 VIS: −0.05/+0.15/−0.04; signs incoherent | DEAD (R7, R18) |
| 8 | Specificity C2/C4 + DX1/DX3 on dynamics | Penn (FD≤.5), ADHD200 | primary-ADHD vs TD, PRO-noADHD vs TD, subtypes | all \|diff\|<0.04, p>.12; ADHD subtype p>.12 | DEAD — dynamics have no categorical ADHD-specificity (R8) |
| 9 | Task-defined nback states (k=5 on Penn NBACK) | Penn n=86 | ESWAN + dx + stay vs d′ | occ s3 +0.18 p=.10; **dx switch +0.022 raw p=.031** (FD-resid p=.035, split-half 98–99%, median p=.088); **stay-s3 +0.266 p=.014** (FD-clean, split-half 89%, C2/C4/dx null); stay→d′ only +0.15 | Penn-only lead A, no triple possible (R13, R26–27, R36) |
| 10 | VIS-burden extremes (top/bottom 25%) | ADHD-Index 112/110, Penn 23/22, HBN 167/167+167/168 | extreme-group diffs | +0.027 p=.046 / +0.030 p=.096 / +0.004 p=.68 / **−0.014 p=.19 OPPOSITE (HBN externalizing)** | DEAD, sign-incoherent (R28–29) |
| 11 | Limbic modal controllability (paper-6 Eq.10 literal) | ADHD200 dx (n=722 within-site), Penn, HBN | dx / ESWAN-hyp / externalizing | ADHD dx −0.0014 p=.010 → −0.00098 p=.073 resid (site-4 flip; 2/5 sites neg; split-half sign 98%, median p=.082); Penn dimensional +0.27 p=.014 but **Penn dx +0.0049 OPPOSITE**; HBN +0.19 p=.002 = age confound (HI 9.6y vs LO 11.1y, partial → +0.10 n.s.) | LEAD B, triple-failed, precedented (Henry 2022) → deprioritized (R15–R16, R40) |
| 12 | Subnet (FPN/DMN/LIM) avg/modal + A-sweep | ADHD200, Penn | dx | FPN avg +9e−5 p=.047 / modal p=.033 nominal, Penn null p=.31–.99; A-sweep p=.024–.036 but subnet-LIM sign OPPOSITE to 7-net LIM | construction-dependent → closed (R17–18) |
| 13 | HBN extreme groups modal | HBN | HI vs LO externalizing | raw p=.007; HI age 9.6y vs LO 11.1y — pure age confound | DEAD (R18) |
| 14 | Backlog: Markov entropy-rate (transition operator) | Penn rest | ESWAN | r=−0.164 p=.15, below prespecified bar \|r\|>0.2 | not expanded (R18) |
| 15 | Backlog: ADHD-Index / p_factor on surviving whispers | n=541 / n=833 | VIS-occ, switch | +0.044 p=.30 / +0.022 p=.54 | DEAD (R19) |
| 16 | NAS minimax over 36 LEiDA-NCT configs (replicability objective) | ADHD + Penn screening, HBN waived | all × search-perm | champion MIN +0.046 vs null-mean +0.076 / q95 +0.106 → **search-perm p=.92** | DEFINITIVE NULL for the whole enumerated family (R21–23) |
| 17 | Kuramoto-PLV (modal coupling) | ADHD200 | LIM/FPN vs Hyper | −0.007 p=.90 / −0.013 p=.83 | gate-failed, closed (R24) |
| 18 | Paper-7 fidelity check (k sweep 3–20 + Dunn) | method audit | — | Dunn → k=5 = overnight's k; all paper-7 metrics covered | FIDELITY CONFIRMED — null is not from wrong k (R25) |
| 19 | Learned dynamics: K-Kuramoto fit (per-subject) | ADHD 7-net | 1-step R² gate | **R²=0.008** — phase-velocity not fittable → STOP | gate-failed; SLDS gated behind it (R74) |
| 20 | SLDS-2 switching linear model | spec only | occupancy/switch vs symptoms, incremental-validity gate Δr≥0.05 vs static-FC | never run (R95 queued; stalled) | DESIGNED, NOT EXECUTED |
| 21 | PAC slow-5-phase × slow-3-amp (Tort-MI) | Penn rest | ESWAN + dx | dim +0.05/+0.12, dx ≈0; family min-p=.293 | CLOSED (course 26) |
| 22 | dPLI directed phase-lag, 21 pairs, Morlet | Penn rest + nback | ESWAN Holm + dx | best rest r=.194 p=.080; nback r=−.193 p=.071; Holm 1.0 (2 impl. bugs caught pre-read) | CLOSED (course 27 + ext) |
| 23 | ds005899 (4th dataset, pediatric CSST task, n=61) | behavior F1 + Gate P + pilot n=6 | GoRT/IIRV/SSRT + CSST switch/stay | GoRT +37ms p=.011 ALIVE; IIRV/SSRT marginal; Hilbert pilot freeze×IIRV r=−.45 directional; F2 pilot D1 +0.005 p=.40, stay-IIRV +0.40 p=.19, stay-SSRT +0.05 p=.87 | battery designed (F2-Discovery v2, Morlet-locked) — **GATED on ~35GB pull, never run at n=61** |
| 24 | ds000030 adult rest (chunk-1+2, 10 ADHD processed) | battery S prespec | switch dx + LIM-modal (generalization frame) | never tested | GATED on 2.6GB pull |
| 25 | Harmonization (16_harmonization) | all 3 → pooled Schaefer-100 via atlas-overlap weights | VIS-occ in common space | pooled dict ARI .98, test **REJECTED wrong-sign p=.037** — kills the 3/3 VIS whisper; voxel-level harmonization impossible (no BOLD on disk) | whisper C declared dead (R37–40) |
| 26 | LEiDA vs IQ (cognition positive control) | n=655 pooled | VIS-occ / switch vs IQ | +0.060 p=.117 / +0.043 p=.275 — dynamics family poorer than statics (static IQ r≈.208) | null even for cognition (R77–78) |
| 27 | HBN internalizing / p_factor sweep | HBN | VIS-occ/switch | +0.052 p=.144 / +0.029 p=.41; externalizing age-confounded | null (R78) |

Provenance sub-track (00_recovered): the *pre-overnight* certified dispersion signal itself was
audited — 6 holes closed (byte-exact rebuilds max-diff 0.0), item-3 reproduced
(+0.319/VR 2.96/p=.0004 site-median), 3-block jackknife survives (p=.0005–.0015), and the
**null hierarchy** was established: pooled-frozen (~.0005) < within-site-frozen (~.002) <
refit (~.04). The certified number used the weakest null; honest p≈.04. This context matters:
the program had already learned that its own strongest "positive" result was fragile.

---

## Failure taxonomy

**(a) Genuine null — no signal exists under that pipeline (the bulk).**
7-net occupancy/dwell/switch/entropy (rest, all bands), NCT centroid + one-hot energy,
dPLI, PAC, Kuramoto-PLV, ROI full-n, specificity (C2/C4, DX1/DX3), ADHD-Index/p_factor/
internalizing backlogs, LEiDA-vs-IQ. Capped by the power memo (MDEs d≈0.2 cross-cohort,
r≈0.10 HBN, r≈0.15 ADHD dim, r≈0.30 Penn dim) and priced definitively by NAS search-perm
p=.92. "Hidden deep" = d<0.2, or task-cohort-only (MDE ~0.7), or non-first-moment/non-dictionary.

**(b) Pipeline bug / fidelity gap vs the papers.**
- Paper-7 fidelity per se is CLEAN (k=5 by Dunn confirmed; metrics covered) — the null is
  not a wrong-k artifact. But a real fidelity gap remains: paper 7 uses DPABI preprocessing
  with **0.02–0.1 Hz bandpass**; the overnight's primary R1–R6/R13 object was **broadband
  Hilbert phase** on vendor-preprocessed parcel series. 23_methods/morlet_hilbert.json:
  Hilbert-broadband vs Morlet-0.05 V1 agreement **0.44** (≈chance for 7-dim), dynamics scale
  10× apart (|dV1| 2.55 vs 0.27), subject-rank r=0.33 — two different biological objects.
  All Morlet-locked batteries (which would close this gap) are byte-gated and unrun.
- Bugs caught during the night: 2 dPLI implementation bugs (pre-read), mopar 6-param motion
  integration failed 3× silently (replaced by weaker FD-lite regression), NaN-`.mean()` bug in
  jackknife, SSRT **units bug** (+21ms p=.15), duplicate-file bug in F2 runner, variable-T
  pointer bug in ADHD ROI, ds005899 two scanner-gain scales (~15 vs ~180 DVARS units).
- HBN TR was **assumed 0.8s, not verified** (flagged in R7) — data-hygiene gap in every HBN
  frequency number.
- E_enter_VIS ≡ mean-FC **r=.97** — a metric that relabels its own confound (design flaw, not
  a coding bug). Refit-null spec (23_methods): frozen-perm p is anti-conservative for any
  fit-on-controls pipeline (receipts: frozen .023→refit .075; .0025→.040; placebo passes
  vacuously 0/20 under frozen perms but 16–19/20 through-pipeline).

**(c) Confound not controlled / discovered late.**
- Age: HBN limbic-modal HI-vs-LO raw p=.007 is pure age (9.6y vs 11.1y); HBN externalizing
  contradicts attention direction.
- Motion/FD: LIM-modal dx p=.010→.073 after FD+age+sex (30% attenuation); ADHD200
  FD-med imbalance (0.82/0.71); FD-switch corr within group never audited (flagged R13).
- Site: ADHD200 site-4 adult flip in LIM-modal; only 2/5 sites negative; pooled perms
  showed site-structure sensitivity in the static audit (+0.25 pooled vs +0.17 site-median).
- FC-strength: E_enter_VIS and (per audit) the energy family generally are inverse-FC-strength.
- Scanner gain (ds005899) and stochastic registration (label agreement 0.51–0.63) in the 4th dataset.

**(d) Statistical power / multiple comparisons.**
Penn n=85 dimensional MDE r≈0.30 — every Penn rest whisper (−0.14 VIS, −0.16 entropy-rate,
−0.23 dE4) sits at or below its own detection edge. Task cohorts MDE ≈0.7 (3× worse).
Nominals repeatedly die at Holm: HBN s4 −0.22 p=.034 → ~.20 over 6 tests; nback dx-switch
.031 → ~.19; dPLI .08 → Holm 1.0 over 21–28 pairs. The program's response (search-perm,
Holm, refit) was disciplined — the small effects were real-looking but unconfirmable at the
available n. Power is NOT the explanation for the headline null (bounded null excludes
d≥0.2); it IS the explanation for why the whisper shelf can't be promoted.

**(e) Contrast choice wrong / phenotype mapping incoherent.**
Three cohorts, three partially incommensurable clinical targets: ADHD-200 Inatt (higher=worse),
Penn ESWAN (inverted, higher=better), HBN attention (higher=better) — plus HBN externalizing,
which has the OPPOSITE sign to attention (−0.09, males +0.11 vs females −0.11). Every triple
required the same sign under three different scales; dimensional-without-categorical
dissociations repeated (stay-s3, LIM-modal: dimensional p=.014/p=.010 but dx null/opposite).
Penn dx opposite sign to ADHD200 dx for the same metric (LIM-modal) — the categorical label
itself behaves incoherently across cohorts. The one contrast family that was ever clean
(dispersion C2-alive/C4-null) belonged to the static program, not dynamics.

**(f) Preprocessing too aggressive.**
Largely NOT observed — if anything the overnight was too *permissive*: broadband (unfiltered)
phase in the core courses, FD-lite (weaker than 6-param) in ds005899, no GSR regression
anywhere (no artifact-chasing). Slow-band retrictions (slow-5, slow-3) were tested and null,
so "we over-filtered" is not a live explanation. The preprocessing worry runs the other way:
preprocessing is *less* standardized than paper 7 (no bandpass in the core object; HBN TR
assumed; parcel series of unknown provenance from CC200 .1D/CSV dumps).

---

## Near misses (gold for the next round — actual numbers)

1. **Penn nback stay-s3: r=+0.266, p=.014 (n=86)** — FD-clean (r=−0.00), split-half sign 89%,
   but categorically null (C2 −0.014 p=.73; C4 +0.044 p=.24; dx −0.023 p=.42 — largest diff in
   the NON-ADHD C4 group) and behaviorally ungrounded (stay→d′ only +0.15 vs d′→inatt +0.24).
2. **Penn nback dx-switch: +0.022 raw p=.031 / FD-resid p=.035**, split-half sign 98–99%,
   median p=.088 — but Holm ~.19 over 7 tests, single cohort.
3. **Penn within-subject d_switch vs ESWAN-total: +0.23 p=.065** (inatt +0.21 p=.094), n=71.
4. **Penn load-change energy dE4: −0.23 p=.066** — Penn-only, no rest analogue anywhere.
5. **HBN switch vs attention: −0.066 p=.064 at n=833** — the ONLY sub-.10 rest number at full
   power, but its sign is opposite to the ADHD-200 pilot convention (more-switching=worse-attention
   in HBN vs more-switching=less-inattention in ADHD pilot).
6. **VIS-occ whisper family — 5 same-sign reads:** ADHD-Index extremes +0.027 p=.046; Penn
   extremes +0.030 p=.096; Penn rest −0.144 p=.18; ADHD-Index dim +0.044 p=.30; p_factor
   +0.022 p=.54; HBN internalizing +0.052 p=.144. Killed by: HBN attention extremes flat
   (+0.004 p=.68), HBN externalizing OPPOSITE (−0.014 p=.19), and the harmonized pooled test
   REJECTING wrong-sign p=.037.
7. **Limbic-modal:** ADHD200 dx −0.0014 p=.010 raw → p=.073 residual; split-half sign 98%,
   median half2 p=.082, 36% of halves p<.05; Penn dimensional +0.27 p=.014 (but dx +0.0049
   opposite p=.083). Best-characterized whisper; precedented (Henry 2022) so novelty-barred.
8. **HBN ROI s4 VIS occurrence: −0.22 p=.034 nominal (n=100, Holm ~.20)** — never replicated
   (ADHD ROI all |r|<0.06; no profile match; harmonized retest contradicted).
9. **dPLI:** Penn rest pair-13 r=+.194 p=.080; nback pair-15 r=−.193 p=.071 (21 pairs, Holm 1.0).
10. **Penn rest entropy-rate: −0.164 p=.15** (prespecified bar |r|>0.2 not met — not expanded).
11. **ds005899 pilots:** Hilbert freeze×IIRV r=−0.45 (n=6, directional only); Morlet F2 pilot
    D2stay-IIRV +0.40 p=.19, D1 switch +0.005 p=.40 (n≈6-ish pilot battery rows).
12. **Subnet FPN modal/avg ADHD dx: +4.3e−5 p=.033 / +9.3e−5 p=.047** — nominal, Penn null,
    A-sweep construction-dependent → classified not-real.
13. **ADHD200 site-5 slow-5 pilot −0.17 (n=39)** — evaporated at full n (+0.017 p=.75).

Pattern across all near-misses: **PennLEAD is where everything whispers** (n≈85–87, single
site, TR=0.8, ESWAN inverted) — 6 of the 13 entries are Penn-only. Nothing whispers in ≥2
datasets with the same sign except VIS-occ (which harmonization killed).

---

## Contradictions between experiments

1. **LIM-modal sign flips by construction AND by dataset:** 7-net LIM-modal ADHD dx −0.0014
   (ADHD LOWER) vs subnet LIM-modal +0.00010 (ADHD HIGHER, p=.033 nominal) — same construct,
   different construction, opposite sign (R17–18). And Penn dx +0.0049 opposite to ADHD200 dx.
2. **VIS-occ: broadband whisper (3/3 "more VIS=worse", e.g. Penn −0.144) vs harmonized pooled
   Schaefer-100 test REJECTED wrong-sign p=.037** (R39–40) — the only cross-parcellation-clean
   test contradicts the per-dataset whisper. Also slow-3 Penn VIS-occ +0.15 vs broadband Penn
   VIS-occ −0.14: **band flips the sign within the same dataset.**
3. **Switching direction flips pilot→full within HBN:** pilot +0.10 (more switching=better
   attention, R3) vs full-n −0.07 (more switching=worse, p=.064, R6) — and opposite to the
   ADHD-200 pilot direction (−0.11, more switching=less inattention).
4. **7-net s1 occupancy:** ADHD −0.16 / Penn −0.21 agree; HBN +0.14 flips (R2).
5. **s3 occupancy:** ADHD pilot +0.22 (more s3=worse) vs Penn +0.12 under ESWAN-inverted
   (more s3=better) — wrong-sign by construction of the contrast (R2).
6. **Phase estimator:** Hilbert-broadband vs Morlet-0.05 V1 agreement 0.44, |dV1| 2.55 vs 0.27,
   rank r=0.33 (R113) — the program's core object changes identity with the estimator.
7. **Penn 5-net switch +0.022 p=.031 (7-net dictionary) vs 5-net switch null p=.31 sign-flipped
   (R62)** — dictionary resolution flips the task-switch result; killed F2-as-replication.
8. **ds005899 freeze×IIRV −0.45 (Hilbert pilot) vs stay-IIRV +0.40 (Morlet F2 pilot)** —
   related constructs, opposite signs, both n=6 pilots. Treat both as uninterpretable.
9. **LEiDA dynamics vs IQ (r≈+0.04–0.06, null) vs static FC vs IQ (r≈.208)** — the dynamic
   family carries LESS cognitive variance than the static family it was meant to supersede.
10. **NAS champion (slow-5 occ resid, MIN +0.046) underperforms the average null champion
    (+0.076)** — the best bilateral config is worse than typical label-noise.

Contradictions 1–3 and 7 are the load-bearing ones: each pairs a surviving whisper against a
cleaner test that rejects it, which is the classic signature of small-n dictionary noise, not
of a suppressed signal.

---

## What to avoid (dead ends, one line each)

- Re-running ANY 7-net rest dynamic metric (occ/dwell/switch/entropy, any band): NAS search-perm
  p=.92 + bounded null — the family is priced empty.
- One-hot / centroid control energy (E): E≡mean-FC r=.97; it is inverse-FC-strength, adds nothing.
- E_enter_VIS, Kuramoto-PLV, PAC, dPLI as LEiDA-style clinical metrics: all closed with receipts.
- Broadband Hilbert phase as the primary object: r=.44 agreement with the Morlet object the
  batteries are locked to; the core courses tested an object paper 7 doesn't use.
- ROI-state cross-parcellation triples via Yeo-profile matching: ill-posed by construction
  (|v|<0.05 washout, Penn ARI .30); harmonized-pooled transport already contradicted its one hit.
- Limbic-modal as a novel claim: precedented (Henry 2022), age/FD-confounded, sign unstable
  across construction and cohort.
- HBN as a confirmation cohort for attention contrasts: age-confounded, externalizing sign-opposite.
- Pilot r's at n<100 as effects: site-5 s3 +0.22, slow-5 −0.17, ROI k8 −0.43 all evaporated.
- ds005899 F2 as a *replication* of Penn (reframed to discovery in R62; limbic 0/26, VIS warp
  lottery r~0.15, stochastic registration, MDE≈0.7).
- Task cohorts without behavioral anchors (the d′ lesson) or with n<100 (MDE 0.7).

---

## Hypotheses for why everything failed

**H1 — Resolution/feature-class loss: the clinical signal (if any) is not in first-moment
phase-state statistics at 7-net resolution; compression to 7 nets destroyed it, and the only
common-space escape (harmonized transport) broke the very effect it was meant to rescue.**
Evidence: E≡FC r=.97 (7-net control structure adds ~nothing beyond mean FC); LEiDA-vs-IQ null
(+0.04/+0.06) while static-FC IQ r≈.208 — the dynamic family is strictly poorer than statics;
ROI-level states were MORE stable (ARI .99 vs .79) and yielded the strongest nominals (HBN s4
−0.22, k8 −0.43 pilot), all of which then died on cross-parcellation ill-posedness rather than
on replication failure *within* a parcellation.
Cheap falsifiable check: **within-dataset full-n ROI test in HBN alone** (same 200 parcels,
k=6, n=300+ with attention): if s4 VIS-occ shrinks to |r|<0.08 it was dictionary noise; if it
holds near −0.2, resolution was the blocker and the next step is a single-atlas multi-cohort
pull (voxel or common atlas), not more 7-net work.

**H2 — The triple bar is miscalibrated for a heterogeneous syndrome: requiring same-sign
p<.05 in all 3 datasets, under inverted scales, site structure, and sign-flip variance, has
near-zero pass rate even for known-real effects of this size.**
Evidence: HBN externalizing vs attention sign-opposite; LIM-modal 2/5 sites negative with a
site-4 flip; the certified static effect itself needed pooled vs within-site vs refit null
hierarchy (+0.25 pooled vs +0.17 site-median, p .0005→.04) to be honest; NAS showed even the
best bilateral config underperforms the null-mean. If a true d≈0.15–0.2 effect existed with
the observed between-cohort sign variance, it would look exactly like this night.
Cheap falsifiable check: **calibration run — push a known-real effect (the certified static
dispersion d=0.34, or static mean-FC vs attention) through the identical triple bar.** If even
that fails same-sign-p<.05-in-all-3, the bar itself (not the biology) is the failure mode, and
the acceptance rule should be redesigned (e.g. 2-of-3 with sign-locked third, or meta-analytic
fixed-effect p across cohorts).

**H3 — Rest-defined state dictionaries are the wrong object for clinical contrasts; task-loaded
states are where the structured variance lives.**
Evidence: every rest-state metric is null across ~20 courses; the ONLY dimensional p<.015
numbers in the whole night are task-derived (nback stay-s3 +0.266 p=.014) or off-dictionary
(LIM-modal Penn +0.27); nback dx-switch p=.031 with 98–99% split-half sign; ds005899 CSST
pilots directional (stay-IIRV +0.40); and the Morlet/Hilbert split (r=.44) suggests rest
broadband "states" are dominated by non-oscillatory phase jumps with no symptom mapping.
Cheap falsifiable check: **state-dictionary transfer test inside PennLEAD** (data on disk):
compare predictive power of nback-defined state features vs rest-defined state features for
ESWAN on the SAME subjects (split-half). If task dictionary ≫ rest dictionary, the next byte
spent should be task data (ds005899 F2, ds000030 stop), not rest reanalysis.

**H4 — Confound residue (motion, site, age) is exactly the size of the surviving whispers;
the program's own refit receipts prove that this pipeline's p-values inflate by ~+0.07 bias
units, which is enough to fabricate every p=.01–.05 nominal seen.**
Evidence: LIM-modal dx .010→.073 under FD+age+sex; refit spec receipts (frozen .023→.075,
.0025→.040, placebo passes vacuously 0/20 frozen but 16–19/20 through-pipeline); HBN
extreme-group p=.007 fully explained by age (9.6 vs 11.1y); ADHD200 FD-med imbalance.
Cheap falsifiable check: run the **placebo-through-pipeline assay (10 label shuffles) on the
two strongest whispers** (nback stay-s3, LIM-modal raw) — if ≥2/10 placebos yield p<.05,
every whisper number in the program is uninterpretable regardless of its nominal p.

**H5 — The LEiDA/phase machinery is the wrong *dynamics model* for parcel BOLD at these TRs:
the phase field is not a low-dimensional oscillator system here, and no fitted-dynamics
alternative was ever actually run.**
Evidence: K-Kuramoto 1-step R²=0.008 (gate-failed — phase velocity has no predictable
structure); Hilbert-broadband |dV1| 2.55 (raw jump-dominated) vs Morlet 0.27; SLDS-2 with its
incremental-validity gate was designed but NEVER executed (R95, stalled); the whole program
tested fixed formulas, so "the model class is wrong" has never been tested against a fitted
alternative with a validation gate.
Cheap falsifiable check: **run SLDS-2 (2-state, EM, 7-net) on ADHD-200 T-cohort only** (CPU
hours, no new bytes) against its own prespecified gate: if SLDS features beat static-FC by
Δr≥0.05 on Inatt, the phase formulation was the problem; if not, the dynamics family is
confirmed empty at the model level too, and the null map becomes final for this data.

---

## Session-state residue (for whoever continues)

- Stalled since R92 (STALL_STATE.md): 27 courses + 1 extension, 0 triples; goal left ACTIVE.
  Unstall triggers are ALL user-side bytes: `pull_bolds.sh` (~35GB → F2 at n=61),
  `pull_rest.sh` (~2.6GB → adult battery), stop-signal pull, GPU box, fresh task cohort.
- Everything else is banked: batteries F2-Discovery v2 (Morlet-locked, search-perm familywise)
  and S (stop-signal, generalization frame) are fully prespecified with validated runners;
  refit-null spec (23_methods) is binding; NAS-grade multiplicity discipline is in place.
- The honest next-round priorities implied by this postmortem: (1) H2 calibration check,
  (2) H1 HBN-ROI full-n check, (3) H3 Penn dictionary-transfer check — all cheap, all on-disk —
  before spending any new bytes.
