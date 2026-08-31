# Pilot 6/6b (+ completed 5) — Direction 3: LEADERSHIP STRUCTURE, Gate-0 previews

Protocol: identical to pilots 1–4 (reliability ONLY; **no target ever touched**; candidate
lists pre-committed in the script docstrings BEFORE running; every point reported, dead
branches included). All numbers below are fully deterministic (cached V1 / closed-form
wavelet+V1; no k-means anywhere in A–E survivors), so the formal sieve Gate-0 re-run
reproduces them exactly.

Split-half = first vs second half of each run, Pearson, SB = 2r/(1+r); n=871 with T>=60;
locked cache geometry (f=0.05Hz, 5 cycles, cap 60s, TR=2.0, L=29TR). "siteres" = site-demeaned halves.

## Headline table (pilot6_d3_results.json, pilot6b_d3_results.json)

| candidate | d | mean SB | siteres | verdict |
|---|---|---|---|---|
| **Lnet28** — Yeo7 unordered block means of L=(1/T)Σ l_t l_tᵀ | 28 | **0.428** (all ≥0.32) | 0.399 | ALIVE |
| **Lstrnet7** — network means of node strength s_i=rowmean(L) | 7 | **0.479** (all ≥0.45) | 0.458 | ALIVE |
| Lrow190 — per-node strength | 190 | 0.381 | 0.360 | alive, dim price (not proposed) |
| **Mnet7** — network means of first moment m=(1/T)Σ l_t | 7 | **0.400** (all ≥0.34) | 0.382 | ALIVE |
| Mrow190 — per-node m_i | 190 | 0.308 | 0.291 | borderline (not proposed) |
| V2net28 — block means of L2=(1/T)Σ v2_t v2_tᵀ | 28 | 0.156 (1/28 ≥0.30) | 0.134 | **DEAD** |
| V2abs7 — net means of (1/T)Σ|v2_t| | 7 | 0.203 (3/7 ≥0.30) | 0.188 | **DEAD** |
| **ratio_mn** — mean_t λ2_t/λ1_t | 1 | **0.500** | 0.428 | ALIVE |
| **bim_frac** — frac_t 1[λ2_t/λ1_t > 0.5] | 1 | **0.480** | 0.447 | ALIVE |
| lam1_mn — mean_t λ1_t/N | 1 | 0.532 | 0.470 | see Fact-B flag |
| pers_R — pooled mean run length | 1 | 0.044 | 0.022 | **DEAD** (σ²_W=13.6, σ²_T≈0) |
| pers_phi — lag-1 label persistence | 1 | 0.062 | 0.032 | **DEAD** |
| dwell5 — per-state mean run length | 5 | 0.200 | 0.158 | **DEAD** |
| A_k — mean_t ⟨l_t, l_{t+k}⟩, k=1/4/8/16 | 4 | 0.23/0.15/0.16/0.17 | 0.12–0.20 | **DEAD** |
| **FUSED 2-band Lstr7** (0.05 c5K60 + 0.08 c3nat, z-fused) | 7 | **0.560** (all ≥0.53) | 0.534 | ALIVE (best) |

Anchor: locked-dict occ5 mean SB 0.358 per state [0.25–0.47] — reproduces exp/36 Fact A (0.32–0.51).
Measured dwell on locked dict = 9.49 TR; state changes/half = 9.40.

## Noise math (verified: ANOVA pred r = σ²_T/(σ²_T+σ²_W) matches observed to 3 decimals)

- Occupancy dies by binomial-on-epochs: σ²_W ≈ p(1−p)/N_eff = 0.16/10 ≈ 0.018 (measured
  0.018–0.021) vs between-subject σ²_T ≈ 0.004–0.011 → occ SB 0.32–0.51 = Fact A.
- L-strength comps: σ²_W ≈ 1.1e-8, σ²_T ≈ 5.1e-9 → half-r 0.31 → SB 0.48. Same ~10
  epochs/half, but every window contributes (no conditioning on rare events) and 26–32
  nodes are averaged per network → SNR 0.45 vs occupancy's 0.28.
- bim_frac: σ²_W = 0.0045 → ~36 effective windows/half — the λ2/λ1 ratio fluctuates
  FASTER than leader identity (dwell 9.5TR ≈ 10 epochs), so it gets ~3–4× more
  independent samples.
- Fusion: disjoint spectral support → σ²_W,fused ≈ (σ²_W1+σ²_W2)/4 → measured 0.479/0.517 → 0.560
  (the pilot-1 occupancy-fusion result 0.36→0.48 transfers to L).

## Fact-B flag (audit item — must travel to the consolidator)

exp/36 ANALYSIS.md states lam1 split-half r = −0.06 citing `reliability_lam1.py`; that
script (and reliability_occ.py, oracle_blocks.py, sieve_batch1.py) do NOT exist in
exp/36_leida_ceiling/ (only ANALYSIS.md), so Fact B is non-auditable. Three independent
reimplementations at the exact cache geometry give lam1_mn split-half SB 0.52–0.53:
pilot4 (c5K60 methodA 0.531), pilot5 (wavelet-halves 0.519; Hilbert 0.684; sliding-window-FC
0.787; only degenerate odd/even splits give ~1.0), pilot6 (0.532). My λ pass reproduces
exp/28's F_STATIC[:,598] at corr = 1.0000. Practical conclusion of Fact B stands via the
oracle table (lam1_3 IQ oracle 0.128 / CV 0.072 — weak content); its stated reliability
number does not stand. λ-RATIO features (D3-C) are not λ1's level (shared window cancels
the global scale) but Gate 0 should be re-run formally on them.

## FC redundancy (feature-only check, no targets)

R² given static FC28 (F_STATIC cols 0:28): Lnet28 0.519, Lstr7 0.542, Mnet7 ≈0.45,
lam1_mn 0.325, ratio_mn 0.309, bim_frac 0.276. L-family ≈ half static-FC variance;
ratio/bimod are the most FC-orthogonal quantities in this batch.

## Files

- `run_pilot6_d3.py` / `pilot6_d3_results.json` — A (Lnet28/Lstr7/Lrow190), B (Mnet7),
  C (ratio_mn/bim_frac/lam1_mn), dead branches (V2, dwell, persistence, A_k), occ anchor.
- `run_pilot6b_d3_fusion.py` / `pilot6b_d3_results.json` — D (fused 2-band Lstr7) + single bands.
- `run_pilot5.py` / `pilot5_results.json` — Fact-B reconciliation (was pre-committed, unrun).
- Timings (4 cores): pilot6 wall 25 s; pilot6b wall 24 s; λ pass alone 2 s; band V1 passes 3 s each.
