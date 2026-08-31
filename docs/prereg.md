# exp/37 — LEiDA Family Sieve: PREREGISTRATION (SCREEN protocol)

**Written BEFORE any screening fit. This is a SCREEN, not a confirmatory experiment.**

## Purpose
Close (or advance) the paper-6/7 LEiDA/controllability family formula-by-formula at ~5 min/variant instead of ~1 h/variant, using the exp/36 ceiling analysis. Discovery workflow (z-ai/glm-5.3-free, tokenrouter; 6 direction theorists + consolidator) proposes variants; this prereg governs how they are tested.

## The three gates (escalating cost; a variant must pass each to reach the next)
- **Gate 0 — reliability (≈2 min/variant)**: split-half Spearman-Brown ρ of the subject-level feature vector (first-half vs second-half of each run, n=871 with T≥60; for vector features: mean per-component ρ). **ρ < 0.30 → variant is DEAD** (attenuation law: with target reliability ≈0.85, ρ=0.30 caps observable r at 0.50×true; below that nothing detectable is possible). Recorded, never fitted further.
- **Gate 1 — oracle (≈2 min/variant)**: in-sample ridge (α grid {0.1,1,10,100,1000,3000,1e4,3e4,1e5,3e5}) on the T-cohort (n=336, sites 3/5/6) target Inattentive AND Hyper; site-centered r. **max oracle |r| < 0.15 → DEAD** (nothing to generalize). Survivors: report both targets' oracle r.
- **Gate 2 — honest battery (only for survivors, preregistered cell list before running)**: 3-site LOSO site-centered r_sc, ridge α inner-fold mode fixed under perms, **familywise max-stat null over ALL variants that reached Gate 2 in this screen (200 perms, permute within site)**. p<0.05 AND |r_sc|>0.15 AND ΔR²>0 vs age/sex/maxmotion AND both-sex same-sign AND all-site same-sign.

## Multiplicity pricing (anti-hacking rules)
1. The Gate-2 familywise null includes EVERY variant that entered Gate 2 — no dropping dead siblings from the correction.
2. A Gate-2 survivor is a **SCREEN POSITIVE**, not a claim. To become a claim it needs: (a) a confirmation run on data not touched by the screen — the PennLEAD cohort (n=87, ESWAN total as the symptom proxy; sign-consistency + Δr>0 threshold preregistered per exp/27 convention) or a site-holdout sub-split of ADHD-200 not used in screening, AND (b) a fresh 4-agent adversarial audit as in exp/28.
3. All Gate 0/1/2 numbers for every variant (dead or alive) are recorded in `results/sieve_table.json` — published nulls included.
4. No variant may be re-parameterized after seeing its gate result (one variant definition = one gate triple; re-entries are logged as NEW variants and inflate the Gate-2 family).

## Variant inventory (frozen at workflow return + my enumerated list; both recorded before Gate 1)
Base enumeration (mine, pre-workflow): FREQ grid {0.04, 0.06, 0.08, 0.10} × KERNEL {30, 45, 90}s; co-leadership matrix summaries; 2nd-eigenvector (λ2, V2) stats; HMM (sticky, 5 states) on V1; state-conditional FC (occ-weighted); transition-count normalization variants. Workflow variants appended verbatim with their proposer's reliability rationale.

## Data constraints (given to all theorists)
CC200: 190 ROI, T=55–232 (site-dependent), TR 1.5–2.5s, bandpassed 0.01–0.1Hz. T-cohort n=336 (sites 3/5/6, ADHD 157/HC 179, Inatt+Hyper T-scores). ADHD-Index n=514 (sites 1/3/5). IQ n=795 (6 sites). PennLEAD n=87 rest (514 ROI, T=156, TR=0.8, ESWAN inatt/hyper/total). Locked exp/21 k=5 dictionary + V1 wavelet cache exists (FREQ=0.05, KERNEL=60s); new frequencies/kernels require recomputation (~10 min/variant, acceptable).

## Stopping rule
Sieve ends when the variant inventory is exhausted (each with a recorded gate triple). If zero survivors: family closed with a quantitative ceiling statement. If survivors: Gate-2 battery, then the claim path above. Either way the trail is complete.
