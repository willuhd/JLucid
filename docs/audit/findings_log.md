# Findings log — rounds 1–3 (exp_leida_xverify)

Session goal: find a REAL clinical LEiDA signal in adhd200 + Penn (cross-dataset verification),
staying in the paper-6 (controllability) / paper-7 (LEiDA) family. Everything below was run in
this session's exp folder with verified data hygiene (TRs verified, ESWAN direction verified
from data, perm p-values with label-blind pipelines, Bonferroni within families).

## Infrastructure built
- adhd200: 1395 runs / 965 subjects cached (190 consistent CC200 ROIs). Athena TCs are NOT
  band-passed (43–67% power >0.11 Hz) and are GS-transformed (static FC mean r≈+0.01,
  bipartite leading mode) → "anti-phase regime".
- pennlead = Penn LEAD (OpenNeuro ds007116, CC0 — no DUC needed; see notes/pennlead_ev_provenance.md).
  104 rest + 104 nback series, TR=0.8 (Cifti header), 156 named Schaefer parcels, anti-phase
  regime. NOTE: initial extraction had a transpose bug (parcels↔time) — fixed; all penn
  numbers in this log are post-fix.
- HBN (hbnt2, CPAC CC200, n=833 manifest / 790 extracted): no-GSR + 0.01–0.1 bandpass →
  INTACT global regime (FC mean +0.12..+0.21, per-frame coherence up to 0.53). Static-FC age
  control r=−0.20 (known-real effect present).
- Paper-7-exact pipeline (bandpass 0.02–0.1 per site TR, Hilbert, dPC, leading eigvec largest
  |λ|, majority-negative, k-means Dunn sweep, occurrence/lifetime/switch) implemented and
  validated: V1-variance fraction >0.5 on 100% of frames in all datasets; HBN states reproduce
  the canonical paper-7 repertoire (global/VIS/DMN/SMN-VAN/FPN); strong canonical age effects
  in HBN (global occ t=−7.3, DMN occ t=+7.0, all Bonf-surviving).

## Results (all with age/sex/FD covariates, 5000-perm p, Bonferroni within family)
1. Rest state dictionaries (occ/life/switch, k=3/5/8): NULL in NYU (best life_2 p=.057
   uncorrected; life_4 p=.012 uncorrected but partial-Spearman null), NULL in Penn. Occupancy
   split-half reliability r≈0.8 (metric is stable, just carries no dx/inatt variance).
2. Dictionary-free scalars (gcoh, gcoh_sd = EiDA spectral radius / spectral metastability;
   novelty note: no human clinical use exists, no LEiDA-ADHD paper at all):
   - NYU dx: gcoh p=.0052 (ρ=+.149), gcoh_sd p=.0016 (ρ=+.147); survives FD<0.5 (p=.0022/.0058),
     FD-matched pairs; subtype-consistent (both dx1/dx3 positive).
   - HBN attention: gcoh_sd p=.0018 (ρ=+.105); gcoh p=.052.
   - Penn: dx flat at rest (t=−0.4..−1.6); CLEAN-group dx ~ gcoh_sd t=−2.66 p=.0086 —
     SIGN-OPPOSITE to NYU. gcoh_sd×FD coupling: NYU +0.49, Penn +0.36 (both motion-sensitive).
   → VERDICT: sign contradiction in the same regime (NYU vs Penn both anti-phase) ⇒ the scalar
   family is preprocessing/non-invariant-latent dependent, NOT a verifiable cross-dataset signal.
   (First-LEiDA-ADHD claim therefore not made.)
3. Paper-6-exact controllability (A=|FC|/(1+λmax)−I, c=1; AC Eq.9; MC Eq.10; Yeo network means):
   - NYU dx: MC_Vis p=.0010 (Bonf .014, d≈+0.44), MC_Limbic p=.0014 (Bonf .0196), MC_SalVentAttn
     p=.0016 (Bonf .0224), AC_Cont p=.0020. Robust: FD<0.5 (t=2.48/2.30/2.03, d≈0.4),
     medication-insensitive (within-ADHD med effect ≈0), leave-25-out t∈[3.11,3.29], both
     subtypes positive, FD<median t=3.40 vs FD>median t=0.82 (motion-attenuation pattern).
   - Cross-site adhd200: KKI −, NeuroIMAGE 0, Peking_1 0 (AC_Cont +.15 p=.29), Pittsburgh
     nominal p=.0005–.005 BUT only 4 ADHD subjects (leverage artifact — not a replication).
     Pooled 4-site fixed effect p≈0.38–0.49 (sign heterogeneity).
   - Penn: DEAD FLAT (all 14 MC/AC |t|<0.5, |ρ|<0.08). HBN: MC_Limbic-p_factor +.076 p=.023
     uncorrected only.
   - Within-ADHD dimensional (Inattentive): null (|ρ|<0.1) — the effect is categorical-only.
   - Mediation: MC_Vis shares a latent with gcoh (ρ=.69)/gcoh_sd (ρ=.60); controlling gcoh the
     dx effect attenuates t 3.21→1.85 — one latent "phase-organization level" drives the family.
   → VERDICT: strong within-NYU effect (best candidate of the session) but NOT cross-dataset.
4. HBN FPN-state lifetime vs attention: p_perm=.0032, Bonf(11)=.035, ρ=−.122 (n=790); split-half
   sign-stable 10/10; leave-25-out stable. Not transferable: anti-phase datasets have no
   canonical FPN state; a dictionary-free "FPN-access" template metric gives incoherent
   cross-dataset signs (NYU sd+, Penn mean−, HBN posfrac− p=.053).
5. Task (n-back) dictionaries in Penn (k=5 Dunn): occ_0/life_0 vs inatt ρ≈−.19, p≈.06–.08
   uncorrected — and the H3 transfer test FALSIFIED: task dictionary ≈ rest dictionary on the
   same 92 subjects (both best |ρ|=0.194). The overnight's task advantage does not survive the
   paper-7-exact ROI narrow-band object.
6. Kuramoto generative fit (paper 7 Eqs 9–17; machinery validated on synthetic data, recovers
   τ/α/K at R²=0.47): real NYU data R² mean 0.013, max 0.029 — below the weak-coupling
   identifiability floor (~0.05). The phase field is NOT a Kuramoto-coupled oscillator system
   at ROI narrow-band either → postmortem H5 now closed at ROI level.

## Why the NYU family fails cross-dataset (mathematical explanation, lesson 6-2)
{gcoh, gcoh_sd, MC_net, AC_net} are all functions of one latent L = "level of phase
organization / FC spectral concentration". L is strongly measurement-dependent:
- L correlates with motion within-td (r≈+0.49 NYU, +0.36 Penn) and with the vendor pipeline
  (GS-handling differs: Athena vs XCP-D vs CPAC);
- dx is associated with L in NYU (+) but NOT in Penn (0 or −), and other adhd200 sites are 0/−;
- a latent that is not measurement-invariant cannot carry a cross-dataset clinical claim.
The remaining adjudicator would be a 4th cohort with OUR OWN preprocessing (ds000030, paper 6's
own dataset, ADHD n≈43 + controls) — heavy (raw BIDS, needs a preprocessing pass).

## Status vs the goal
- Genuine cross-dataset verification (same-sign in adhd200 AND Penn): NOT achieved by any
  metric family so far.
- Two within-dataset Bonferroni-surviving effects exist (NYU dx-MC family; HBN attention-life_4),
  both surviving manual robustness checks (influence, FD, medication, split-half) but neither
  replicates across cohorts.
- Not yet exhausted: (a) adversarial manual review of the two hits (running); (b) ds000030 as a
  self-preprocessed 4th cohort — RULED OUT: only 10/267 rest BOLD files on disk; ds005899 has
  only 14 bold files — both unusable without downloads (disallowed); (c) TPM/dwell-exponent
  micro-variants (low prior; the 7-net switch/entropy family was already NAS-priced null);
  (d) Penn LEAD rest run-2 / XCP-D derivatives would double Penn rest data (download = user
  decision, currently disallowed).

## Round-4 additions — adversarial review + real-FD re-test (DECISIVE)
12. Adversarial manual review (notes/adversarial_review_hits.md) verdicts:
    - **NYU dx-MC family = ARTIFACT**: MC_Vis is r=.98 with the global |FC| spectral radius;
      all 7 network MCs are 0.93–0.98 copies of one motion-loaded scalar; effect ABSENT raw
      (t=1.93 p=.054), exactly zero in the cleanest-motion quintile (t=0.04), concentrated in
      one FD stratum anchored by 33 subjects with an imputed FD placeholder (0.532); within-ADHD
      symptom gradient genuinely null.
    - **HBN attention ~ life_4 = REAL-in-dataset** (perm p=.0015, split-half 100% sign
      consistent over 200 halves, uniform across strata, attention-specific) BUT my HBN motion
      covariates were fake (dvars_z double-counted; spike_frac ≡ 0).
    - **Penn dx_adhd_i is PRO/CHR-diluted** (dx=1 = 22 ADHD + 33 PRO/CHR; dx=0 = 27 TD + 22
      PRO/CHR). Clean ADHD(22) vs TD(27): MC_Vis d=−0.62 t=−2.16, CI [−1.20,−0.04] — excludes
      NYU's +0.40 → "Penn flat" was a diluted-contrast artifact; Penn is underpowered
      (~0.32) and opposite-signed.
13. Real-FD HBN re-test (FD_power.1D, provenance-ok subset n=314/833 — 513 subjects have the
    WRONG fd file picked per fd_provenance_census.csv):
    - attention ~ life_4: t=−2.87, p_perm=.0046 (Bonf-11=.051), ρ=−.167 (sp_p=.003);
    - FD-matched extreme attention groups: mean life_4 diff = −1.57 frames, t=−3.47 (n=79 pairs);
    - FD<0.2 clean subset (n=105): ρ=−.205, sp_p=.036;
    - the fake-covariate "age effect on global state" largely vanishes under real FD
      (t=−1.37) — the age-global-synchrony link is substantially motion-mediated.
    → HBN FPN-lifetime-attention is the one genuine clinical LEiDA effect this session;
      it is HBN-only (the anti-phase adhd200/Penn regimes cannot host the FPN state;
      the fpn-access transfer metric was incoherent cross-dataset).
14. HBN k-sweep robustness (200-ROI dictionary, k=6/7/8): FPN-lifetime attention association
    replicates at every k (life_fpn p=.0068/.0114, ρ≈−.10); 7-net dictionary k=5 also
    replicates (p=.0090, ρ=−.103); absent at the 7-net k=4 Dunn optimum (state merge).
15. Downloads authorized by the user (2026-09-04): Penn LEAD XCP-D rest run-02 (+ ses-2) via
    ds006779 GitHub raw; ds000030 R1.0.5 task-rest BOLD for 43 ADHD + 130 CONTROL via
    anonymous S3 (per-file uncompressed prefix). No web login needed.

## Round-3 additions
8. Fitted-dynamics controllability (SSM/VAR(1) on the leading-eigenvector trajectory, top-7
   label-blind PCA; A fitted from data, not FC-derived; features ρ(A), Gramian trace, dyn range):
   NYU dx ~ ρ(A) t=+1.75 p=.074 (nominal, + direction, consistent with the L latent);
   Penn dx ~ ρ(A) t=−0.69 (ρ=−.128) — anti-sign again; HBN attention/p_factor null.
   The same NYU(+)/Penn(−) pattern in a sixth metric family.
9. FPN-access dictionary-free transfer of the HBN life_4 hit: NYU fpn_sd dx + (p=.013), Penn
   fpn_mean dx − (p=.048), HBN attention fpn_posfrac − (p=.053) — incoherent; the HBN
   dictionary lifetime effect does not transfer to a template-projection metric.
10. Manual influence analysis (lesson 6-2): NYU dx-MC_Vis survives leave-25-out (t 3.11–3.29),
    both subtypes, FD<0.5, medication; effect concentrated in the low-motion half (t=3.40 low
    vs 0.82 high motion). HBN attention-life_4 survives leave-25-out and is split-half
    sign-stable (10/10 halves).
11. On-disk dataset census: adhd200/pennlead/hbnt2 fully used; ds000030 = 10/267 rest files,
    ds005899 = 14 bold files (both partial, unusable without downloads).
