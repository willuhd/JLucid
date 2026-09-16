# Pilot 7/7b/7c/7d — Direction 5: FUSION AND INTERACTION, Gate-0 previews

Protocol: identical to pilots 1–6 (reliability ONLY; **no symptom/IQ target ever touched**;
candidate lists pre-committed in script docstrings BEFORE running; every point reported, dead
branches included). Split-half = first vs second half of each run, Pearson, SB = 2r/(1+r);
n=871 with T>=60; locked cache geometry (f=0.05 Hz, 5 cycles, cap 60s, TR=2.0, L=29 TRs);
band variants pilot-2-verbatim (3 cycles, cap 90s, KMeans k=5 n_init=10 rs=0, rng(0) 40k-frame
subsample — fully deterministic). "siteres" = site-demeaned halves.

Environment note: `mp.Pool(fork)` HANGS in this DSH sandbox (pilot6's 25s parallel run was
pre-sandbox); all passes re-run serial in-process — measured 0.56–0.74 s/subject, 450–487 s
wall for the full pilot7 pass (all four formulas). Parallel timings below are extrapolated
from pilot6's measured 4-core speedup (~12×); serial numbers are the guaranteed bound.

## Headline table (pilot7/7b/7c/7d_d5_results.json)

| candidate | d | raw SB | siteres | verdict |
|---|---|---|---|---|
| **AxA5_D5A** occ×age, RAW product (locked dict) | 5 | **0.792** | 0.700 | ALIVE but see FLAG-1 |
| **CxA5_D5A** occ×age, cohort-centered (locked dict) | 5 | 0.290 (2/5 ≥.30: s0 .375, s3 .367) | 0.292 | **DEAD by the letter** |
| **CxA_D5A-fused** centered occ×age on FUSED 3-band occ {0.05,0.07,0.09} | 5 | **0.463** (5/5, .37–.56) | **0.456** | **ALIVE (the proposal)** |
| **dFC5×28_D5B** state-conditional FC, hard gate (n_s≥8) | 140 | 0.013–0.081 per state, **0/140 ≥.30** | 0.016–0.073 | **DEAD** |
| dFCsoft_D5B soft gate (w=<v1,c_s>², all windows) | 140 | 0.030–0.129, 0/140 | 0.023–0.114 | **DEAD** |
| pc1 rescue (hard / soft; per-state & 112-dim) | 10 | ≤0.191 / ≤0.164 | ≤0.246 | **DEAD** (no common latent shift) |
| **CT14_D5C-tmpl** occ-weighted TEMPLATE per-state φ/μ | 14 | 0.404 (14/14) | 0.391 | ALIVE but see FLAG-2 (= occ re-mix) |
| **CS14_D5C-subj** occ-weighted OWN-window per-state φ/μ | 14 | **0.578** (14/14, .53–.64) | **0.315** (.24–.40) | **ALIVE (the proposal)** |
| **T10_D5D** template-state transition energies E_ij=(x_j−x_i)ᵀW⁻¹(x_j−x_i) | 10 | **0.843** (10/10, .81–.87) | **0.544** | **ALIVE (best in program)** |
| **E5_D5D** template-state entry energies x_jᵀW⁻¹x_j | 5 | **0.845** (5/5) | **0.551** | **ALIVE** |
| anchors: occ5 locked dict | 5 | 0.358 [.25–.47] | 0.349 | reproduces exp/36 Fact A & pilot6 |
| anchors: FC28 whole-half | 28 | 0.628 (28/28) | 0.599 | repro |

Fused-3-band occ itself: SB 0.505 siteres 0.474 — exact pilot-2 reproduction (per-state
.41/.44/.57/.54/.57). Band occs alone: 0.431–0.446.

## Noise math (ANOVA σ²_W/σ²_T predicts measured SB to ~2 decimals, as in pilot 6)

- **D5-B dies by window-count arithmetic, not by fit quality**: per-state usable windows per
  half (n_s≥8 both halves) = 74%/20%/46%/48%/62% of subjects for states 0–4; median n_s≈10–26.
  Fisher-z from n≈10–26 windows has within-σ ≈ 1/√(n−3) ≈ 0.21–0.44, vs between-subject
  σ(dFC)≈0.1–0.2 → half-r ≈ 0.02–0.08 = measured. The soft gate raises n_eff to 0.46–0.50 of T
  but the weights are then so diffuse that the contrast collapses toward overall FC
  (R²[dFCsoft|FC28]=0.08–0.13) and the residual is noise. PC1 of 28 blocks (SB ≤0.19) shows the
  28 per-block shifts share NO reliable common factor to aggregate — the family is closed at
  these run lengths (T=55–232), by measurement, not by argument.
- **CS14/T10E5 raw-vs-siteres inflation mechanism**: raw SB 0.578/0.843 vs siteres 0.315/0.544.
  Site-T differences (55–257 frames) change n_s, LedoitWolf shrinkage, and usability →
  systematic between-site feature shifts. The honest Gate-0 number is the siteres one; the
  Gate-2 r_sc (site-centered, 3-site T-cohort) operates between the two.
- **ANOVA**: CS14 pred r 0.588 vs measured 0.578; T10 0.704 vs 0.843 (SB formula
  over-predicts at high r); E5 0.707 vs 0.845; CxA-centered 0.171 vs 0.290.
- **Why T10/E5 beat everything**: the ONLY subject-level random object in the feature is W⁻¹
  (from the subject's full-run precision, ~172 windows — exp/36 Fact C: static
  precision/φ/μ are the reliable architecture blocks); the 15 targets x_s are noise-free
  population constants (pooled-cohort template state patterns, unit-norm, |x| 0.10–0.20).
  exp/31's C15 died because its Gramian came from the DOMINANT STATE's own ~26 windows.

## Redundancy (feature-only, no targets; R² given blocks, per pilot6 convention)

| feature | vs occ5(locked) | vs occFUSED | vs FC28 | vs occ5+FC28 | vs static598 (FC28+prec+φ+μ) |
|---|---|---|---|---|---|
| CxA_D5A-fused (centered interaction) | — | **0.026** | 0.044* | — | — |
| CS14_D5C-subj | 0.078 | 0.106 | 0.223 | 0.272 | **0.901** |
| T10E5_D5D (15) | 0.012 | 0.023 | 0.161 | 0.165 | **0.913** |
| CT14_D5C-tmpl | ≡1 (by construction) | — | 0.397 | — | — |
| occ5 anchor | — | — | 0.321 | — | — |

*locked-basis centered variant. D5-A′ is nearly orthogonal to its own occupancy basis
(R²=0.026) — the product term is a genuinely new axis, not occ re-expressed. CS14/T10E5 carry
~73%/84% new variance vs their two PARENTS (occ + FC28) but ~90% is re-expressed static
precision/φ/μ architecture → they inherit exp/36 Fact C's target profile: IQ-positive-control
expected, ADHD-symptom oracle expected ≈0 (the 10–16% residual is where any new symptom
signal would have to live).

## FLAGS (audit items — travel to consolidator)

1. **FLAG-1 (D5-A parameterization)**: the RAW occ×age product measures SB 0.792, but that is
   inflated by the trivially-reliable `mean_occ(s) × age` main-effect term (occ level is a
   stable subject property × exact age). The honest interaction is the COHORT-CENTERED product
   (occ − cohort mean) × age_z. On the locked dict it measures 0.290 → DEAD by the letter; on
   the fused 3-band basis 0.463 → ALIVE. Both parameterizations and both bases are recorded;
   per prereg rule 4 they are SEPARATE variants (dead sibling does not enter Gate 2 — no
   family inflation).
2. **FLAG-2 (C-T is an identity)**: occ-weighted TEMPLATE controllability = occ5 @ [VΦ; VΜ]
   with fixed population (5×14) weights — a fixed linear reparameterization of occupancy
   (R²|occ5 ≡ 1). Its SB 0.404 > occ's 0.358 only demonstrates the house law (fixed
   aggregation over states concentrates reliable variance). NOT proposed as a variant.
3. **FLAG-3 (siteres floor)**: CS14's honest reliability is 0.315 (raw 0.578). It clears the
   0.30 bar but not the exp/36 ρ≥0.35 ceiling bar at its floor; T10/E5 (0.544/0.551) clear it
   comfortably. Attenuation: max observable r = true_r·√(ρ·0.85).
4. **FLAG-4 (mp fork)**: `mp.Pool` fork hangs under this sandbox; serial wall times are the
   guaranteed bounds (all < 11 min/variant). In the normal environment pilot6's 4-core pool
   ran comparable passes in 25 s wall — compute estimates in the proposal use the serial
   bound.

## Files
- `run_pilot7_d5_fusion.py` / `pilot7_d5_results.json` / `pilot7_d5_raw.npz` — all four D5
  formulas + anchors (serial, 450–487 s; raw per-subject halves cached for re-analysis).
- `run_pilot7b_d5_softfc.py` / `pilot7b_d5_results.json` — soft-gate D5-B + first redundancy block.
- `run_pilot7c_d5_agefusion.py` / `pilot7c_d5_results.json` — fused 3-band occ basis + D5-A′
  (72 s wall).
- `run_pilot7d_d5_pc1_redundancy.py` / `pilot7d_d5_results.json` — PC1 rescue attempts (dead)
  + redundancy vs occ bases.

Gate-0 verdicts recorded; Gate 1/2 remain for the formal sieve per `prereg.md`.
