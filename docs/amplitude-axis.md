# The Amplitude Axis of the Wavelet Transform for ADHD rs-fMRI: A Methods Research Report

**Scope:** ADHD-200 (TR=2.0s, T=110–460, 190-parcel CC200) and PennLEAD (TR=0.8s, T=156, 400 Schaefer + subcortical), Morlet instantaneous phase currently computed at 0.04–0.1 Hz for LEiDA-style analysis. This report evaluates the **amplitude** axis of the same transform: precedents, band definitions, reliability, and concrete feature definitions.

---

## 0. Executive verdict (read this first)

**The amplitude axis has substantially more published ADHD support than the phase axis — but "more" is not "strong," and you should calibrate expectations accordingly.**

- **Pro:** ALFF/fALFF is the single most-published rs-fMRI index in pediatric ADHD (14 ALFF + 3 fALFF + 2 dALFF studies in the 2025 ALE meta-analysis alone), with two independent ALE meta-analyses converging on *decreased left frontal* activity (L middle frontal BA 9, superior frontal, medial frontal) and *increased occipital* (lingual/cuneus/middle occipital) activity. The ALFF measure was literally invented in an ADHD study (Zang et al. 2007). Two 2024–2025 papers apply frequency-resolved amplitude/FC analysis to ADHD-200 itself with AUCs of 0.72–0.96 (with important caveats, see §2).
- **Con:** (1) Nearly all positive amplitude studies are small, single-site, and moderately corrected; meta-analytic jackknife replicability is moderate (9/17 iterations for the child frontal cluster). (2) The multisite ADHD-200 pooled ADHD-vs-TC problem has historically been brutal: the ADHD-200 competition's landmark prediction model achieved 94% specificity but only **21% sensitivity** on the held-out test set (Eloyan et al., cited in Chen et al. 2024, BMC Psychiatry). (3) **There is not a single published ADHD study using amplitude-envelope correlation (AEC/aEPC-style) networks in fMRI** — the entire AEC/aEPC literature is MEG/EEG. (4) At T<200 TRs, amplitude-*variability* features are statistically fragile, and no fMRI study has directly compared amplitude vs. phase reliability at short scan lengths.
- **Net:** swapping phase for amplitude does not rescue a null program by itself; the best-grounded bets are (a) slow-5/slow-4 fALFF in frontal/OFC/precuneus parcels for ADHD-200 long-scan subjects, and (b) a narrowband band-passed FC arm (the strongest recent frequency-axis ADHD result, BMC Psychiatry 2025) as a positive control that also enables a formal amplitude-vs-phase decomposition of the connectivity estimate.

---

## 1. Q1 — AEC / aEPC / BLP: standard implementations and fMRI translation

### 1.1 The canonical AEC/aEPC family (MEG origin)

Four implementations define the standard; all share the structure *band-pass → (source-space) signal → envelope → correlation*:

1. **Brookes et al. 2011 (NeuroImage 56:1082–1104)** — two variants:
   - **AEC (Averaged Envelope Correlation):** band-pass filter; Hilbert transform → analytic signal; envelope = |analytic signal|; split envelope into *n* segments of length Δ (typically ~10 s); Pearson *r* between seed and target envelope **within each segment**; average the *n* correlation values.
   - **CAE (Correlation of Averaged Envelopes):** average the envelope within each Δ window first, then correlate the two downsampled envelope time series (more robust to phase jitter in the envelope; similar to de Pasquale 2009).
2. **Hipp et al. 2012 (Nature Neuroscience 15:884–890, doi:10.1038/nn.3101)** — *(note: the paper you're thinking of as "Hipp & Hawellek 2011" is this 2012 paper; Hawellek is second author)*: power-envelope correlation on **pairwise-orthogonalized** source signals, envelopes low-pass filtered below **0.5 Hz** before correlation; symmetrize the resulting matrix by averaging with its transpose. Correlation was strongest for alpha–beta carriers (8–32 Hz), frequency-specific hubs.
3. **Brookes et al. 2012 (NeuroImage 63:910–920)** — multivariate regression framework that removes linear leakage between seed and all targets simultaneously.
4. **Colclough et al. 2015 (NeuroImage, doi:10.1016/j.neuroimage.2015.03.071)** — **symmetric multivariate orthogonalization**: the current gold standard; removes the shared subspace from all ROIs at once (SVD-based) and fixes the order-dependence of pairwise orthogonalization.

The **Wens et al. 2014** line (Brain Topogr 27:620–634) uses seed-based **aEPC** (amplitude envelope power correlation) with minimum-norm sources, static orthogonalization, envelope low-pass at 1 Hz; the Brussels group's connectome-level extension is all-to-all aEPC. A precise published operational definition of the leakage-corrected variant (**AEC-c**) that you can implement verbatim (from the Alzheimer's MEG literature):

> "The corrected amplitude envelope correlation (AEC-c) overcomes the effects of spatial leakage by using pair-wise orthogonalisation prior to the AEC estimation for each pair of time series... The correction is performed by orthogonalization separately for each pair of time series... in two directions by means of linear regression, meaning time-series X is regressed out from time-series Y and time-series Y is regressed out from time-series X, and the AEC values (Pearson's correlation between the orthogonalized envelopes) for both directions are averaged." (Alzheimer's Res Ther 2022, doi:10.1186/s13195-022-00970-4)

**Implementable formula (pairwise-symmetric AEC-c, per pair i,j):**

```
x_i, x_j : band-passed parcel time series (band B)
1) x_j|i = x_j − (⟨x_i,x_j⟩/⟨x_i,x_i⟩)·x_i        # remove i's linear component from j
   e_i = |H(x_i)|,  e_j|i = |H(x_j|j... i)|        # Hilbert envelopes
   AEC(1) = r(e_i, e_j|i)                          # Pearson over time
2) swap roles of i and j → AEC(2)
3) AEC-c(i,j) = ½[AEC(1)+AEC(2)]                    # symmetric
   optionally remap to [0,1]: (AEC-c+1)/2
```
where `⟨a,b⟩` is the inner product over time and `H(·)` the Hilbert/analytic transform of the narrowband signal.

The **technical review** covering all of this: O'Neill, Barratt, Hunt, Tewarie & Brookes 2015, "Measuring electrophysiological connectivity by power envelope correlation: a technical review on MEG methods," Phys Med Biol 60(21):R271, https://iopscience.iop.org/article/10.1088/0031-9155/60/21/R271

### 1.2 Band-limited power (BLP)

BLP = squared band-passed signal (or Hilbert envelope²) of a fast carrier, whose own slow fluctuations (~0.01–0.4 Hz) carry the connectivity. The definitive fMRI-relevant demonstration is **Hacker et al. 2017** ("Frequency-specific electrophysiologic correlates of resting state fMRI networks," NeuroImage; ECoG): ECoG filtered into log-spaced carrier bins, squared, then the *envelope* band-pass filtered again into log-spaced modulation bins; envelope correlations in theta and gamma (but not intermediate frequencies) matched BOLD RSN topography. PDF: https://www.research.unipd.it/retrieve/e14fb26a-8693-3de1-e053-1705fe0ac030/frequency%20specific%20HACKER%20NEUROIMAGE%202017%20%201-s2.0-S1053811917300629.pdf

### 1.3 What happens when you transplant this to fMRI — the critical asymmetry

In MEG, the carrier is 8–30 Hz neural activity and the envelope modulates at <0.5 Hz. **In fMRI at TR=2s the "signal" in 0.01–0.1 Hz already IS the envelope** of underlying neural activity. Therefore:

- **Correlating band-passed BOLD parcels directly** (the standard 0.01–0.08 Pearson FC; what BMC Psychiatry 2025 did per slow band) is already an envelope-correlation-like operation — it mixes amplitude co-fluctuation and phase coherence.
- **Taking the Hilbert/Morlet envelope of narrowband BOLD** (IA of the 0.04–0.1 Hz component) and correlating envelopes measures *second-order* coupling — modulation-of-modulation at ≤0.01–0.03 Hz, which requires very long scans (≥300–500 TRs) to estimate. This is the single most important feasibility constraint on importing AEC to your data.
- The one published fMRI study that directly computes Hilbert **instantaneous amplitude (IA)** RSNs (vs instantaneous phase, IP) is **Mital, Sao & Biswal 2023** ("Impact of Amplitude and Phase of fMRI time series for Functional Connectivity Analysis," Magnetic Resonance Imaging 102:26–37, doi:10.1016/j.mri.2023.04.002; HCP, TR=0.72s, n=100): IA-based seed RSNs have **comparable cross-session consistency to IP-based RSNs** (IA better for motor, IP better for fronto-parietal), and **fusing IA+IP improves session-to-session similarity 3–20%** (DMN, motor; in 0.01–0.04, 0.04–0.07, 0.07–0.1 Hz, slow-5, slow-4). This is the closest published template for your amplitude axis.

### 1.4 Pitfalls: leakage, common signal, orthogonalization

| Pitfall | In MEG | Your fMRI analog | Required correction |
|---|---|---|---|
| **Common-signal leakage** | Source-space field spread inflates zero-lag AEC (Brookes 2011, 2012; Colclough 2016: "the impact of magnetic field spread... is profound... can artificially inflate measures of consistency") | (a) aliased respiration/cardiac; (b) shared global BOLD signal; (c) spatial smoothing + partial-volume in parcel averaging | aCompCor/physiological denoising; global-signal regression **or** global-signal orthogonalization (GSOr, Jo et al. 2010, NeuroImage — the direct fMRI homolog of orthogonalization); avoid over-smoothing |
| **Order-dependence of pairwise orthogonalization** | Asymmetric matrices | same | Symmetric version (Colclough 2015; or average the two directions as in §1.1) |
| **Orthogonalization removes true zero-lag signal** | Discussed in Front Psychiatry 2020 MEG-SZ reliability study: metrics robust to leakage "eliminate zero-phase correlations... invasive measures have demonstrated zero-phase correlations across broad regions... not all zero-phase correlations are related to artifact" | Same tradeoff: GSReg removes real global amplitude covariance | Report both with and without GSOr/GSReg; interpret direction of change |
| **Envelope temporal support vs scan length** | Envelope LP at 0.5–1 Hz needs minutes | Envelope of a 0.04 Hz BOLD component fluctuates at ~0.005–0.02 Hz → ≥300–500 TRs | Only compute envelope-coupling features on long-scan subjects (ADHD-200 ≥300 TRs) |
| **Wavelet edge effects** | n/a (MEG fast) | Morlet temporal σ_t = n_cycles/(2πf₀); support ≈ ±3σ_t (see §4.3) | Discard edge samples; constrain f₀ × n_cycles to scan length |
| **ALFF inflated by aliased physiological line noise** | n/a | Cardiac 0.9–1.3 Hz aliases to 0.0–0.2 Hz at TR=2s AND to 0.05–0.25 Hz at TR=0.8s; child respiration 0.25–0.4 Hz aliases to 0.05–0.2 Hz at TR=2s | Use **fALFF** (band ratio) rather than raw ALFF — this is exactly why Zou et al. 2008 introduced it; report motion (mean FD) as covariate |

**Frequencies used at each TR (in the published fMRI literature):**
- TR≈2s: 0.01–0.08 Hz (classic), or slow-5 (0.01–0.027) / slow-4 (0.027–0.073) splits (Zuo et al. 2010; the entire ADHD frequency-band literature). 0.04–0.07 Hz narrowband for phase synchrony (Glerean et al. 2012, NeuroImage — this is also the band your 0.04–0.1 Hz choice extends). Slow-3 usable but aliased-physiology-contaminated (see §4).
- TR≈0.6–0.8s (HCP/PennLEAD-class): 0.01–0.1 Hz plus higher bands become accessible; Mital 2023 used 0.01–0.04/0.04–0.07/0.07–0.1 and slow-3/slow-2 on HCP TR=0.72s; fast-fMRI work (Lin, ISMRM 2013, 10 Hz sampling) tracked Morlet band-limited power envelopes at 0.1–4 Hz; the HCP-class multiband literature shows RSNs exist >0.1 Hz (up to 0.4+ Hz; Lee et al. 2013; Boubela et al. 2013). Figueroa et al. 2020 found LEiDA **phase** reliability is *maximized* when including >0.1 Hz components at TR=0.72s — a hint that your 0.04–0.1 Hz band may be reliability-limited on the phase side, which fast TR partially fixes.

---

## 2. Q2 — ADHD amplitude literature: ALFF/fALFF/dALFF, effect sizes, regions

### 2.1 The two papers you named, verified in detail

**BMC Psychiatry 2025** — "Frequency-specific alterations in low-frequency functional connectivity in children with ADHD," BMC Psychiatry, doi:10.1186/s12888-025-07586-6, https://link.springer.com/article/10.1186/s12888-025-07586-6 (published 2025-11-21).
- n = 40 ADHD, 45 HC children. Bands: slow-3 (0.073–0.198), slow-4 (0.027–0.073), slow-5 (0.010–0.027), per Buzsáki & Draguhn 2004 / prior ADHD work.
- **Method note: this is NOT amplitude.** They band-pass filtered into each slow band, extracted AAL ROI time series, computed **Pearson FC** (Fisher-z), t-tests, then linear SVM (70/30 split, permutation-tested).
- Findings: slow-3: increased FC right precentral gyrus; slow-4 & slow-5: increased FC right inferior frontal orbital region; Slow-4/5 patterns identical → combined. **AUC 0.7550 (slow-3), 0.7830 (combined slow-4/5)**; accuracy 79–85%.
- Caveats: single-site n=85, t-test feature preselection + 70/30 split (optimistic), and their OFC finding is *opposite in direction* to Chen et al. 2024's decreased OFC fALFF — they discuss this discrepancy explicitly ("reduced fALFF may index diminished local amplitude, whereas increased FC reflects stronger network coupling; these patterns are not mutually exclusive... the importance of integrating both connectivity and amplitude-based measures in future frequency-resolved ADHD research").

**Frontiers in Human Neuroscience 2024** — Chen, Jiao, Zhu, Wang, Zhao, "Frequency-specific static and dynamic neural activity indices in children with different ADHD subtypes: a resting-state fMRI study," Front Hum Neurosci 18:1412572, doi:10.3389/fnhum.2024.1412572.
- **This IS ADHD-200**: ADHD-C n=25, ADHD-I n=26, TD n=28 from the ADHD-200 Consortium. Static + dynamic fALFF and ReHo in slow-5 and slow-4.
- Key results: subtype×frequency interaction in orbitofrontal (OFC), precuneus (PCUN), superior temporal (STG), angular (ANG) gyri. **ADHD-C and ADHD-I both show decreased static AND dynamic fALFF of OFC in slow-5.** Differences "more prominent in slow-5 than slow-4." Dynamic ReHo SD increased in STG/PCUN/ANG (ADHD-C). All slow-5 ROIs correlated with ADHD index.
- Classification (5 ROI features, logistic regression): slow-5 AUC **0.956** (ADHD-C vs TD, acc 90.6%), **0.907** (ADHD-I vs TD, acc 88.9%), **0.782** (ADHD-C vs ADHD-I); slow-4 AUCs 0.695–0.774.
- Caveats: n=79 total, ROI features selected from the same interaction analysis (circularity/optimism), in-sample ROC.

### 2.2 Meta-analytic convergence (the most replicated amplitude findings)

Two 2025 ALE meta-analyses + one SDM meta:

1. **Eur Child Adolesc Psychiatry 2025** (doi:10.1007/s00787-025-02906-3): 28 rs-fMRI studies, 1019 ADHD / 943 HC (methods: ReHo 14, ALFF 14, fALFF 3, dALFF 2). **Children with ADHD: decreased spontaneous activity in left middle frontal gyrus (BA 6, 9), superior frontal (BA 6), medial frontal (BA 6), precentral (BA 9); no increased regions.** Adolescents: increased paracentral/postcentral/medial frontal (sensorimotor); decreased cerebellum (tonsil, uvula, declive, anterior lobe) + superior/medial frontal (BA 9, 10). No overlap between child and adolescent clusters. Jackknife: child frontal clusters replicated in 9/17 iterations (moderate).
2. **World J Psychiatry 2025** (15 studies, 468 ADHD/466 HC adolescents, https://www.wjgnet.com/2220-3206/full/v15/i4/102215.htm): **increased right + left lingual gyrus (BA 18) and right cuneus (BA 23); decreased left medial frontal gyrus (BA 9) and left precuneus (BA 31)**; jackknife robust (11/13, 11/14).
3. **SDM meta-analysis, Front Psychiatry 2023** (doi:10.3389/fpsyt.2022.1070142; 36 fMRI studies): overactivation in **left middle occipital gyrus** (BA 18/19; SDM-Z 1.41, 347 voxels, replicated 34/36), right insula, right precuneus, cerebellum; hypoactivation bilateral superior temporal, left middle/inferior frontal, right pre/postcentral.

**Consensus amplitude map for feature targeting (pediatric ADHD):**
- **Decreased amplitude:** left middle/inferior frontal (BA 9/6), medial/superior frontal, left precuneus, cerebellum (adolescents), OFC (ADHD-200 slow-5, Chen 2024).
- **Increased amplitude:** occipital — lingual (BA 18), cuneus (BA 23), middle occipital; sensorimotor (adolescents: paracentral, postcentral).
- **Effect sizes:** rarely reported as Cohen's d; typical peaks: SDM-Z 1.2–1.4; Lou et al. 2021 dALFF peaks t=4.43 (R MOG), t=−4.68 (L MFG); AUCs 0.72–0.96 single-site (in-sample-optimistic), 0.75–0.78 in the best recent small-sample study.

### 2.3 Other amplitude-adjacent ADHD papers you asked about ("amplitude-envelope ADHD papers I'm missing")

**Honest answer: I found zero fMRI ADHD studies computing amplitude-envelope (Hilbert/Morlet envelope) features or envelope-correlation networks.** The closest existing literature is:

- **Lou et al. 2021** (Front Neurosci 15:731596, doi:10.3389/fnins.2021.731596): 50 ADHD / 28 HC children; **dynamic ALFF (dALFF) variability** (sliding-window ALFF, SD across windows, 32-TR Hamming windows, step 4 TR): ADHD showed **increased dALFF variability in right middle occipital gyrus and decreased in left middle frontal gyrus** (GRF-corrected), plus decreased voxel-wise concordance in left MFG correlating with WCST errors. Not ADHD-200 (GE Signa HDX 3T). *This is the direct precedent for your "amplitude variance" feature — see §5.*
- **Hong & Hwang 2022** ("Resting-State Brain Variability in Youth With ADHD," Front Psychiatry 13:918700, doi:10.3389/fpsyt.2022.918700): **ADHD-200** (Peking + NYU sites, TR=2s; 105 ADHD / 140 TD, ages 7–17). Greater number of significant BOLD signal changes and higher-order polynomial associations in DMN BOLD variability in ADHD, emerging **after the first 140 s**, coinciding with decreased DMN FC.
- **Nomi et al. 2018** (Front Hum Neurosci 12:431? per citation; doi:10.3389/fnhum.2018.00090): 40 ADHD / 30 TD children 7–12y; BOLD variability (MSSD): **no categorical ADHD-vs-TD difference**; dimensional: higher MSSD in dorsal/ventral MPFC correlated with ADHD-index and inattention severity (rho≈0.50). *Important honesty check: the categorical effect was null.*
- **Zang et al. 2007** (Brain Dev 29:83–91) — the original ALFF paper, performed in children with ADHD (13 vs 12).
- **fNIRS** ALFF ADHD (BMC Psychiatry 2024, doi:10.1186/s12888-024-06350-6): 34 ADHD-C / 52 ADHD-I / 24 HC children; increased ALFF in temporal/VPC channels (note: direction here differs from fMRI frontal findings — modality and coverage differences).
- **Complexity (amplitude-adjacent):** ABCD pre-adolescents (PMC10473793): reduced multiscale entropy in FPN regions in ADHD.
- **Frequency-band concordance ADHD work** (same group, Front Neurosci 2023, doi:10.3389/fnins.2023.1196290): ADHD-C/I/TD (same n=79 ADHD-200 subsample); voxel-wise concordance differences in MCC and SMA in **slow-5** only.

### 2.4 Practical note for your pipeline

The **ADHD-200 preprocessed release ships whole-brain fALFF and ReHo maps** (Athena pipeline), so parcel-level fALFF baselines are effectively free to compute against: Bellec et al., "The Neuro Bureau ADHD-200 Preprocessed Repository," doi:10.1101/037044, https://doi.org/10.1101/037044 (includes CC200/CC400 parcellations and regional time series).

---

## 3. Q3 — Test-retest reliability: amplitude vs phase

### 3.1 Direct evidence (MEG — the only head-to-head comparisons that exist)

- **Colclough et al. 2016**, "How reliable are MEG resting-state connectivity metrics?" (NeuroImage; HCP test-retest; 12 metrics): *"We find poor test-retest reliability in phase- or coherence-based metrics such as the phase lag index or the imaginary part of coherency. The most consistent methods for stationary connectivity estimation over all of our tests are simple amplitude envelope correlation and partial correlation measures"* — but only after leakage correction (uncorrected metrics look more "reliable" because the leakage artifact itself is highly repeatable). https://www.sciencedirect.com/science/article/pii/S1053811916301914
- **Garcés et al. 2016** (Brain Connect, doi:10.1089/brain.2015.0416): PLV showed high reliability but plausibly leakage-inflated; leakage-corrected envelope correlation (lc-ecor) reliable in beta. https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2020.551952 (the SZ reliability study) discusses the same tradeoff: metrics that "retain zero-phase signal" (coh, PLV) are more repeatable, possibly spuriously.
- **MEG in psychosis** (Front Psychiatry 2020, 10.3389/fpsyt.2020.551952): ICCs poor-to-good (0.26–0.66) for phase metrics (PLV/PLI/wPLI2), lower in patients than controls.
- **AD MEG** (Alzheimer's Res Ther 2022, doi:10.1186/s13195-022-00970-4): AEC-c was **the most reproducible metric** across two independent cohorts, correlated with disease severity, and was not confounded by band power; effect sizes ~−0.3 (alpha) / −0.4 (beta). PLI less sensitive.
- **Graph-measure reproducibility** (2022, PubMed 35019189, https://pubmed.ncbi.nlm.nih.gov/35019189/): *"The greatest reliability... was obtained when using amplitude metrics"* (AEC and leakage-corrected AEC, across 6 bands, source and sensor space).
- **OPM-MEG test-retest** (Life/PMC12007552 + bioRxiv 2022.12.21.521184): within-subject AEC connectome consistency 0.56–0.78 (theta/beta/alpha) from 600-s runs; ~60% within-subject at 5 min (Colclough); 72% at 560 s with head-cast (Liuzzi et al. 2017).

**Verdict from MEG:** amplitude-envelope correlation ≥ phase metrics for test-retest reliability, *provided leakage correction is applied* (otherwise the reliability is artifact).

### 3.2 fMRI evidence (amplitude measures; no direct phase comparison)

- **ALFF ICC ≈ 0.5–0.8; fALFF ≈ 0.5; improves with scan length.** Anesthesia study (Front Neurosci 2022, doi:10.3389/fnins.2022.937172; 15-min scans, 9 patients): awake mean ICC — **ALFF 0.81, fALFF 0.51, FC 0.65, ReHo 0.84**; reliability increases monotonically with scan length (1→15 min), and reliability of FC is the most scan-length-sensitive.
- **Zuo & Xing 2014** (systematic review of rs-fMRI reliability, cited therein): ALFF largely reliable within/between sessions.
- **High-amplitude ≠ reliable:** PLOS ONE 2015 (doi:10.1371/journal.pone.0128117): regions with the *highest* ALFF magnitude (PCC, mPFC, thalamus, primary visual/motor) had the *poorest* reliability; minimum ~156 s to compute ALFF at 0.01–0.08 Hz; short-TR (0.4s) data improved reliability notably. **Directly relevant warning for you: the DMN regions where ADHD amplitude effects concentrate are among the least reliable ALFF regions.**
- **Inter-scanner** (Front Neuroinform 2018, doi:10.3389/fninf.2018.00054): PerAF (percent amplitude of fluctuation) had the best inter-scanner reliability of ALFF/ReHo/DC; inter-scanner much worse than intra-scanner — relevant for multisite ADHD-200 (8 sites!).
- **Multiband/TR effects on ALFF reliability** (Antwerp, 24 older adults × 3 visits; repository.uantwerpen.be/docstore/d:irua:16068): ALFF ICC 46–81 (moderate to almost perfect); higher for MB protocols (TR 0.32s) than single-band (TR 2s) in cortex; single-band better subcortically; ALFF reliability > seed-based FC reliability.
- **FC scan-length dependence** (Birn et al. 2013, PMC4104183): reliability keeps improving to 12–16 min; +20% ICC for 12 vs 6 min — i.e., at 5–8 min (ADHD-200's typical 220–460 TRs = 3.7–15 min), FC-class measures are still scan-length-limited.

### 3.3 The specific "amplitude beats phase at T<200 TRs" question — honest gap

**No published fMRI study directly compares amplitude-envelope vs phase-feature reliability at short scan lengths.** The inference chain supporting it is indirect:
1. MEG head-to-heads favor AEC over PLI/PLV/imcoh (Colclough 2016; Garcés 2016; AD 2022);
2. fMRI ALFF has acceptable ICC at ≥5–8 min but degrades below ~150 s, and fALFF < ALFF for reliability;
3. the one fMRI IA-vs-IP comparison (Mital 2023) found them **roughly equal** (IA better for motor RSNs, IP better for fronto-parietal), with fusion best of all;
4. the only short-scan phase evidence is Figueroa 2020 (ghost attractors): LEiDA within-subject reliability *improved* when >0.1 Hz components were included — implying sub-0.1-Hz-only phase features (your band) are reliability-limited at any TR, and at TR=2s you cannot extend upward (Nyquist 0.25, aliasing above ~0.1).

**Reasonable prior: yes, amplitude > phase for single-subject reliability in your setting — but it is an extrapolation from MEG, not an established fMRI fact, and the amplitude advantage is confined to *mean* amplitude (ALFF-class), not *variability* of amplitude (dALFF-class), for which no reliability data exist at all below ~200 TRs.**

---

## 4. Q4 — Band definitions and TR-specific usability

### 4.1 Canonical definitions (Buzsáki & Draguhn 2004, Science 304:1926–1933, as applied to BOLD by Zuo et al. 2010, "The oscillating brain: complex and reliable," PubMed 19782143)

| Band | Definition | Cycle length | Gray-matter relevance |
|---|---|---|---|
| slow-5 | 0.010–0.027 Hz | 37–100 s | Strongest in cortical/DMN structures (mPFC, PCC) |
| slow-4 | 0.027–0.073 Hz | 14–37 s | Most robust in basal ganglia/subcortical (Zuo 2010); reliable across scans |
| slow-3 | 0.073–0.198 Hz | 5–14 s | Aliased physiological territory at TR=2s |
| slow-2 | 0.198–0.25 Hz | 4–5 s | WM/physiological; rarely used |

### 4.2 What is usable at each TR

**ADHD-200, TR=2.0s (Nyquist 0.25 Hz):**
- **slow-5 and slow-4: fully usable and recommended.** These are exactly the two bands the entire ADHD frequency-band literature uses, and they are the two with documented gray-matter specificity and test-retest stability (Zuo 2010: "slow-5 and slow-4 appear as stable parameters across scans with preferential spatial patterns").
- **slow-3 (0.073–0.198): below Nyquist, hence *nominally* usable, but it is the aliasing landing zone.** At TR=2s, sampling rate 0.5 Hz: child respiration 0.25–0.4 Hz folds to 0.1–0.25 Hz and 0.4→0.1, 0.45→0.05 Hz (i.e., into upper slow-4!); cardiac 0.9–1.3 Hz folds to 0.0–0.2 Hz. So slow-3 power in ADHD-200 is a superposition of neural signal and aliased respiration, and your current upper edge 0.1 Hz already includes aliased-cardiac contributions. Mitigations: fALFF-style normalization, aCompCor, mean-FD covariates; ADHD-200 mostly lacks RETROICOR-able physiological recordings. **Notable counterpoint:** BMC Psychiatry 2025 still found the *largest* single-band FC effect in slow-3 (precentral; AUC 0.755) — so the band is contaminated but not empty.
- **slow-2 (0.198–0.25): unusable for features** (respiratory alias core, near-Nyquist attenuation).

**PennLEAD, TR=0.8s (Nyquist 0.625 Hz):**
- **slow-4 and slow-3 fully usable**; respiration (0.2–0.33 Hz) is *directly sampled, not aliased* — it can be band-stopped or regressed if respiratory traces exist (check your acquisition; if not, it is an unmodelled confound sitting inside slow-2/upper slow-3, and fALFF normalization is again advisable).
- **Cardiac still aliases into the target band at TR=0.8s:** f_s = 1.25 Hz; cardiac 1.0–1.3 Hz folds to |f−1.25| = 0.05–0.25 Hz — i.e., **exactly your slow-3/slow-4 range.** This is an under-appreciated point: shortening TR to 0.8s fixes respiration aliasing but NOT cardiac aliasing (you need TR≲0.4s to beat ~1 Hz cardiac; HCP-class TR=0.72 still has 1.25-fold issues resolved the same way). Prefer ratio-based (fALFF) amplitude features for both datasets.
- **slow-5 is effectively unusable in PennLEAD** — not because of Nyquist, but because of **record length and wavelet support** (next section).

### 4.3 The wavelet temporal-support constraint (critical, and specific to your Morlet implementation)

For a Morlet wavelet with n cycles at center frequency f₀, the temporal standard deviation is σ_t = n/(2πf₀); the effective support is ≈ ±3σ_t (a 99% energy window).

| Scan | Duration | f₀ (n=5 cycles) | σ_t | Support (±3σ) | Verdict |
|---|---|---|---|---|---|
| PennLEAD T=156 @ 0.8s | 124.8 s | 0.02 Hz | 39.8 s | ±119 s (238 s) | **Edges dominate; slow-5 out of reach** |
| PennLEAD T=156 | 124.8 s | 0.04 Hz | 19.9 s | ±60 s (120 s) | Marginal — barely one support window |
| PennLEAD T=156 | 0.07 Hz | 0.07 Hz | 11.4 s | ±34 s (68 s) | Fine |
| ADHD-200 T=110 @ 2s | 220 s | 0.015 Hz | 53 s | ±159 s (318 s) | Marginal; slow-5 only with n≤3 |
| ADHD-200 T=110 | 220 s | 0.04 Hz | 19.9 s | ±60 s (120 s) | Fine |
| ADHD-200 T=460 @ 2s | 920 s | 0.012 Hz | 66 s | ±199 s (398 s) | Fine |

Additionally, for **slow-5 specifically in PennLEAD:** at 0.01 Hz the cycle is 100 s; a 124.8-s run contains ~1.25 cycles of the slowest slow-5 component. **No amplitude, phase, or envelope estimate at 0.01 Hz is meaningful in PennLEAD. Restrict PennLEAD amplitude analysis to slow-4 (0.027–0.073, i.e., ~3–9 cycles) and slow-3 (~9–25 cycles).** Your existing 0.04–0.1 Hz Morlet band is actually well-placed for PennLEAD (spans the slow-4/slow-3 boundary where wavelet support is adequate); for ADHD-200 you can afford to split into canonical slow-5 and slow-4 bands, with slow-5 restricted to subjects with ≥250–300 TRs (≥500–600 s).

### 4.4 Band recommendations, concretely

- **ADHD-200:** primary = slow-5 + slow-4 (0.01–0.073 Hz), computed as two separate features or a 0.01–0.073 combined band (matching the PLOS dementia work and BMC 2025's combined slow-4/5 arm, their best classifier at AUC 0.783). Secondary/exploratory = slow-3 (0.073–0.198) with explicit aliasing caveats. Your current 0.04–0.1 Hz band straddles the slow-4/slow-3 boundary; for amplitude work prefer canonical splits so results map onto the meta-analytic literature.
- **PennLEAD:** slow-4 (0.027–0.073) primary; slow-3 (0.073–0.198) secondary — this is the dataset where slow-3 is uniquely clean (non-aliased respiration). Do not attempt slow-5.

---

## 5. Q5 — Amplitude variance (std of instantaneous amplitude): precedents

**Yes — this feature family exists in the literature, and "dALFF variability" is its dominant instantiation.**

### 5.1 dALFF variability (closest precedent)

Dynamic ALFF = sliding-window ALFF (window ~32–50 TR; step 4–5 TR), with **variability = SD of windowed ALFF across time** (then voxel-normalized, z-scored). Papers:
- **Lou et al. 2021** (ADHD; see §2.3): increased dALFF variability right MOG, decreased left MFG.
- **EOS schizophrenia** (Front Neurosci 2020, doi:10.3389/fnins.2020.00901; 78 patients/90 HC): decreased dALFF variability in bilateral precuneus, right supramarginal/postcentral; increased right MTG, correlated with negative symptoms.
- **Schizophrenia framework paper** ("Characterizing Dynamic Amplitude of Low-Frequency Fluctuation and its relationship with dynamic functional connectivity," PMC5860934, 151 SZ/163 HC): dALFF clusters into 6 recurring "dALFF states"; dALFF–dFC correlations altered in SZ; windowed-ALFF mean correlates r>0.9 with static ALFF.
- **GAD** (Cui 2020) and **MDD** (Li 2019) — both cited within the retrieved ADHD/EOS papers as showing dALFF variability differences; the GAD paper is described as dALFF contributing *more than static ALFF* to group separation.
- Windows are typically Hamming, 32 TR (64 s) with 4 TR step (Lou 2021) or 50 TR with 60–80% overlap (EOS 2020); results reported stable across 30–80 TR windows.

**Relation to your feature:** std of Morlet instantaneous amplitude A(t) over time is the *windowless, better-resolved* version of dALFF variability restricted to your wavelet band. Two formal differences to note: (a) dALFF-variability uses band 0.01–0.08 (FFT) while yours would be the Morlet band (e.g., 0.027–0.073 for slow-4); (b) the sliding window (≥32 TR ≈ 64 s) low-passes the variability estimate — your A(t)-std captures faster envelope fluctuations (down to the band's own timescale), which is *more* information but has no reliability data behind it. Both are amplitude-variance features; dALFF variability is the published operationalization, your A(t)-std is the wavelet-native one.

### 5.2 BOLD signal variability (SD / MSSD of the broadband signal) — related but distinct

- **Garrett et al. 2010+; Nomi et al. 2017 J Neurosci 37:5539** (lifespan, 6–85y, two TRs: MSSD formula `MSSD = mean_t [x(t+1) − x(t)]²` after z-scoring each voxel's time series; MSSD preferred over SD because it is robust to autocorrelation and TR-dependent temporal smoothing).
- **Nomi et al. 2018** (ADHD, dimensional mPFC effects only; no categorical effect — see §2.3).
- **Zhang et al. 2016 (temporal variance)** as synthesized in the pediatric BOLD-SV review (Life 2023, 13(7):1587, https://www.mdpi.com/2075-1729/13/7/1587): children with ADHD showed increased BOLD SV in dorsal-attention/sensory regions and decreased in subcortical regions; increased variability in medial frontal (ASD) vs posterior cingulate (ADHD).
- Key distinction: these are **broadband** variance features (no band isolation); your A(t)-std is band-limited and therefore closer to dALFF variability than to BOLD SD/MSSD.

### 5.3 Verdict for Q5

Precedent exists and is specifically ADHD-positive (Lou 2021), but (a) n was 50/28, single site; (b) the categorical result in the closest pediatric BOLD-variability study (Nomi 2018) was **null** — only dimensional symptom correlations emerged; (c) no test-retest data for variability features at short T. Treat amplitude variance as a hypothesis-generating secondary feature, computed windowlessly from your existing A(t), and only interpret it for subjects where T ≥ ~250 TRs.

---

## 6. Q6 — Combining phase AND amplitude from wavelets for clinical classification

**Honest answer: no paper I found combines wavelet-derived instantaneous amplitude and phase features from the same transform for clinical classification in fMRI.** The three closest precedents:

1. **Mital, Sao & Biswal 2023** (Magnetic Resonance Imaging 102:26–37, doi:10.1016/j.mri.2023.04.002): Hilbert IA + IP (and IF) of narrowband BOLD, seed-based RSNs in healthy HCP. **Fusing IA- and IP-derived maps improved test-retest similarity 3–20%** over IP alone (DMN, motor; bands 0.01–0.1 Hz + slow bands). Not clinical, but this is your exact methodological template — and it shows the amplitude representation carries *partially independent, additive* information, not redundant information.
2. **Castro et al. 2014** (NeuroImage; PMC3946896): multiple-kernel-learning classification of schizophrenia from complex-valued fMRI (task AOD), combining **magnitude and phase** ICA features: +5% classification accuracy when phase data included vs magnitude-only. https://pmc.ncbi.nlm.nih.gov/articles/PMC3946896/
3. **EEG (not fMRI): Li, Wang, Li, Zhao 2023** (Cognitive Neurodynamics, doi:10.1007/s11571-023-10041-5): P-MSWC fusion feature (synchrosqueezed wavelet coherence — amplitude-inclusive — + phase-locking value) for MDD classification with CNN; fusion explicitly motivated by "phase synchronization relies only on the phase difference... and does not consider amplitude."

**Implication for you:** there is a genuine methodological gap/first-mover opportunity: a Morlet amplitude+phase joint feature set (e.g., [mean A, std A, A-envelope edges] ⊕ [LEiDA eigenvector states]) for ADHD classification has no direct precedent, but every component has separate support. A cheap, principled version is available to you immediately: the **narrowband correlation decomposition.** For band-passed signals, the Pearson correlation (what BMC 2025 computed per slow band) factorizes as

  r_ij(narrowband) ≈ (amplitude-coupling term) × (⟨cos Δφ⟩ phase-coherence term)

formally: for zero-mean narrowband analytic signals z_i = A_i e^{iθ_i}, the real correlation r(x_i, x_j) = ⟨A_i A_j cos(θ_i−θ_j)⟩ / (σ_i σ_j) — i.e., the narrowband FC you can already compute decomposes exactly into a **phase-locking factor** ⟨cos Δθ⟩ (your LEiDA ingredient) and an **amplitude co-fluctuation factor** ⟨A_i A_j⟩/⟨A_i²⟩^½⟨A_j²⟩^½ (the AEC ingredient). Testing ADHD effects in each factor separately (and their interaction) is (a) exactly implementable from your existing complex wavelet coefficients, (b) directly interpretable, and (c) unpublished in ADHD.

---

## 7. Ranked feature list — amplitude-axis features with best published support for ADHD at n≈100–500

Ranked by (published ADHD evidence × implementability at your TRs and T × expected robustness at multisite scale).

### Rank 1 — Slow-5/slow-4 fALFF ("wavelet ALFF") in frontal/OFC/precuneus/occipital parcels

**Definition (implementable from your Morlet machinery):** for parcel i, band B (e.g., slow-5: scales covering 0.010–0.027 Hz):
```
A_i(t) = |W_i(f,t)|  averaged over scales in B          # you already compute this
wALFF_i(B)   = mean_t [ A_i(t) ]                          # ≈ FFT-ALFF by Parseval
wfALFF_i(B)  = mean_t[A_i(t)] / mean_t[A_full-band(t)]    # fALFF analog (recommended; suppresses aliased physiology)
```
**Band:** ADHD-200: slow-5 (0.010–0.027) and slow-4 (0.027–0.073) separately + a 0.01–0.073 combined arm. PennLEAD: slow-4 only.
**Target regions/direction:** left middle/inferior frontal (BA 9/6) ↓, OFC ↓ (Chen 2024, ADHD-200), left precuneus ↓, medial/superior frontal ↓; lingual/cuneus/middle-occipital ↑ (three meta-analyses).
**Published support:** Zang 2007 (ALFF born in ADHD); Chen 2024 (ADHD-200, slow-5 fALFF OFC decrease, subtype-vs-TD AUC 0.907–0.956); Eur Child Adolesc Psychiatry 2025 ALE (children: left frontal ↓, 28 studies); WJP 2025 ALE (adolescents: L MFG ↓, L PCUN ↓, lingual/cuneus ↑); SDM Front Psychiatry 2023 (L MOG ↑, 34/36 jackknife).
**Caveats:** moderate meta-analytic replicability (9/17 child frontal); high-ALFF-magnitude DMN regions are the least test-retest-reliable (PLOS ONE 2015); fALFF preferred over raw ALFF; covary age/sex/site/mean-FD; restrict slow-5 to subjects ≥250–300 TRs.
**Expected multisite performance:** the honest ceiling is the BMC-2025 range (AUC ~0.75–0.80), not the single-site 0.9+ figures; ADHD-vs-TD will likely be weak while subtype contrasts (C/I vs TD) are likelier to replicate (Chen 2024's subtype AUCs exceeded the pooled contrast).

### Rank 2 — Amplitude variability: std/CV of instantaneous amplitude (windowless dALFF-analog)

**Definition:**
```
CV_i(B) = std_t[A_i(t)] / mean_t[A_i(t)]        # band-limited amplitude variability, no windowing
# or, for literature comparability on long scans (T≥300):
dALFFvar_i(B) = SD over windows w of [ wALFF_i(w,B) ], window 32–50 TR, step 4–5 TR
```
**Band:** slow-4 (and slow-5 for long ADHD-200 subjects); slow-3 in PennLEAD.
**Target regions/direction:** right MOG ↑, left MFG ↓ (Lou 2021); mPFC dimensional ↑ with symptom severity (Nomi 2018); DMN instability after 140 s (Hong 2022, ADHD-200).
**Published support:** Lou 2021 (ADHD, GRF-corrected); Chen 2024 (dynamic indices correlated with ADHD index in all slow-5 ROIs); Hong & Hwang 2022 (ADHD-200); EOS SZ and GAD analogs.
**Caveats:** no reliability data at T<200 TRs; the only categorical pediatric BOLD-variability test (Nomi 2018) was null — dimensional/symptom-severity effects are its stronger mode; windowed versions need ≥10 windows (≥400 TRs at 32-TR windows) — use the windowless CV for short subjects.
**Why rank 2:** it is the *only* amplitude feature with a specific published ADHD group difference at the whole-brain corrected level besides fALFF itself, and it is orthogonal to mean amplitude (captures dynamics the mean discards).

### Rank 3 — Narrowband band-passed FC (slow-4/5 combined + slow-3), as positive control and decomposition substrate

**Definition:** band-pass each parcel time series to B; Pearson r_ij; Fisher-z; edges → SVM/logistic (BMC 2025 protocol). Then decompose each edge (from your complex wavelet):
```
r_ij | B  ≈  ⟨A_i A_j cos(θ_i−θ_j)⟩ / (σ_i σ_j)          # exact for zero-mean analytic narrowband signals
phase factor    : ⟨cos(θ_i−θ_j)⟩                          # your LEiDA ingredient
amplitude factor: ⟨A_i A_j⟩ / (⟨A_i²⟩⟨A_j²⟩)^½           # AEC ingredient
```
**Band:** 0.01–0.073 (combined slow-4/5) and 0.073–0.198 (slow-3).
**Published support:** BMC Psychiatry 2025 — combined slow-4/5 AUC 0.783, slow-3 AUC 0.755 (right precentral ↑, right inferior frontal-orbital ↑); this is the frequency axis's strongest recent ADHD result.
**Caveats:** n=85 single site, t-test feature preselection (optimistic); this is NOT a pure amplitude feature — it is the mixed benchmark your amplitude features must beat or explain; the decomposition above is unpublished in ADHD (see Q6) and is the cheapest genuinely novel contribution available to you.
**Purpose in your program:** (a) validates that your preprocessing can detect a published effect size at all, before interpreting nulls in new features; (b) the phase-vs-amplitude factorization tells you *which component* of the published narrowband effect carries the ADHD signal.

### Rank 4 — Slow-3 amplitude features (PennLEAD only)

**Definition:** wALFF/wfALFF/CV at 0.073–0.198 Hz (9–25 cycles per 125-s run; wavelet support adequate).
**Published support:** honest gap — no published slow-3 *amplitude* ADHD study exists; the only slow-3 ADHD evidence is FC-based (BMC 2025). The rationale is opportunity: PennLEAD is the dataset where slow-3 is uncontaminated by respiratory aliasing, it is the band with the best wavelet support at T=156, and it is where fast-TR fMRI literature shows extra signal (Figueroa 2020: LEiDA reliability maximized including >0.1 Hz; Mital 2023 included 0.07–0.1 Hz bands; fast-fMRI RSNs above 0.1 Hz).
**Caveats:** cardiac still aliases to 0.05–0.25 Hz at TR=0.8s → prefer fALFF normalization; check whether PennLEAD acquired respiratory traces (if yes, RETROICOR/RVT regression is worth doing for this band specifically).

### Rank 5 — AEC / aEPC envelope-correlation networks (exploratory; lowest expected yield)

**Definition:** pairwise-symmetric leakage-corrected AEC per §1.1, on band-passed parcel signals, band slow-4 (0.027–0.073) only:
```
per pair (i,j): orthogonalize both directions, envelope = |Hilbert|, r, average the two directions
fMRI-specific preprocessing: aCompCor + GSOr (global-signal orthogonalization) BEFORE band-passing
restrict to subjects with ≥300 TRs; edges Fisher-z; predict with ridge/SVM
```
**Published support:** *zero ADHD fMRI studies* (honest gap); strong analog evidence that amplitude-envelope coupling is the most reliable and most clinically reproducible connectivity metric — in MEG (Colclough 2016: AEC most consistent, phase metrics worst; AD MEG 2022: AEC-c most reproducible, ES −0.3/−0.4, surviving power covariates; graph-measure reproducibility 2022: amplitude metrics best).
**Why rank 5 despite the MEG endorsement:** in fMRI at 0.027–0.073 Hz, the envelope-of-the-envelope fluctuates at ≤0.01–0.03 Hz, so AEC edges from T=110–460 TRs are estimated from a handful of envelope cycles — the feature is mathematically defined but statistically starved. Only worth attempting in your longest ADHD-200 subjects, and only framed as exploratory.

### Recommended battery summary

| Dataset | Primary | Secondary | Skip |
|---|---|---|---|
| ADHD-200 (all subjects) | R1 fALFF slow-4; R3 slow-4/5 narrowband FC (+ decomposition) | R2 CV slow-4 | AEC edges on short subjects; slow-2 |
| ADHD-200 (T≥300) | R1 fALFF slow-5; R2 windowed dALFF-var slow-5 | R5 AEC slow-4 (exploratory) | — |
| PennLEAD | R1 fALFF slow-4; R2 CV slow-4 | R4 slow-3 fALFF/CV | slow-5 (record too short); any slow-5 envelope feature |

---

## 8. Final honest calibration

1. **The amplitude axis is better supported than the phase axis for ADHD — but the support is modest, heterogeneous in direction, and largely single-site.** If 12 preregistered phase screens were null, amplitude features at multisite scale should be expected to produce *smaller* effects than the single-site literature reports: plan for AUC 0.6–0.75 rather than 0.9+, and prioritize subtype contrasts (ADHD-C vs TD, ADHD-I vs TD) over pooled ADHD-vs-TD, which the ADHD-200 multisite record (Eloyan et al.: 94% specificity / 21% sensitivity) suggests is the hardest contrast.
2. **The genuinely novel, low-cost contribution available from your existing machinery is the narrowband-FC decomposition** (§6): it requires no new data processing, directly interrogates why phase-only features failed, and has no ADHD precedent.
3. **If the amplitude axis also returns nulls across these five features at your n, that is a publishable, well-powered negative result** — the field's amplitude literature is almost entirely n<60 single-site, and a preregistered multisite test with the decomposition and positive-control arm would be informative either way.

---

## 9. Reference list (with URLs)

**AEC / aEPC / BLP methodology**
- Brookes MJ, Hale JR, Zumer JM, et al. (2011). Measuring functional connectivity using MEG: Methodology and comparison with fcMRI. NeuroImage 56(3):1082–1104. doi:10.1016/j.neuroimage.2011.02.054. https://pubmed.ncbi.nlm.nih.gov/21352925/ (AEC/CAE definitions)
- Brookes MJ, Woolrich MW, Luckhoo H, et al. (2011). Investigating the electrophysiological basis of resting state networks using MEG. PNAS 108(40):16783–16788. https://doi.org/10.1073/pnas.1112685108
- Hipp JF, Hawellek DJ, Corbetta M, Siegel M, Engel AK (2012). Large-scale cortical correlation structure of spontaneous oscillatory activity. Nat Neurosci 15(6):884–890. doi:10.1038/nn.3101. https://markussiegel.net/download/hipp_natn_2012.pdf (pairwise orthogonalization; envelope LP 0.5 Hz)
- Brookes MJ, Woolrich MW, Barnes GR (2012). Measuring functional connectivity in MEG: A multivariate approach insensitive to linear source leakage. NeuroImage 63(2):910–920. doi:10.1016/j.neuroimage.2012.03.048. https://doi.org/10.1016/j.neuroimage.2012.03.048
- Colclough GL, Brookes MJ, Smith SM, Woolrich MW (2015). A symmetric multivariate leakage correction for MEG connectomes. NeuroImage. doi:10.1016/j.neuroimage.2015.03.071. https://doi.org/10.1016/j.neuroimage.2015.03.071
- O'Neill GC, Barratt EL, Hunt BAE, Tewarie PK, Brookes MJ (2015). Measuring electrophysiological connectivity by power envelope correlation: a technical review on MEG methods. Phys Med Biol 60(21):R271. https://iopscience.iop.org/article/10.1088/0031-9155/60/21/R271
- Wens V, Bourguignon M, Goldman S, et al. (2014). Inter- and Intra-Subject Variability of Neuromagnetic Resting State Networks. Brain Topography 27(5):620–634. doi:10.1007/s10548-014-0364-8. https://doi.org/10.1007/s10548-014-0364-8 (aEPC seed-based; variability bounds)
- Hacker CD, et al. (2017). Frequency-specific electrophysiologic correlates of resting state fMRI networks. NeuroImage. https://www.research.unipd.it/retrieve/e14fb26a-8693-3de1-e053-1705fe0ac030/frequency%20specific%20HACKER%20NEUROIMAGE%202017%20%201-s2.0-S1053811917300629.pdf (ECoG BLP)
- Mital P, Sao AK, Biswal BB (2023). Impact of Amplitude and Phase of fMRI time series for Functional Connectivity Analysis. Magnetic Resonance Imaging 102:26–37. doi:10.1016/j.mri.2023.04.002. https://www.sciencedirect.com/science/article/abs/pii/S0730725X23000796 (IA vs IP in fMRI; IA+IP fusion)
- Deco G, et al. (2014). How delayed network interactions lead to structured amplitude envelopes. NeuroImage. https://www.sciencedirect.com/science/article/pii/S1053811913011968 (envelope theory + formulas)

**ADHD amplitude/frequency studies**
- Zang YF, et al. (2007). Altered baseline brain activity in children with ADHD revealed by resting-state fMRI (original ALFF). Brain Dev 29:83–91.
- Zou QH, et al. (2008). An improved approach to detection of ALFF for resting-state fMRI: fractional ALFF. J Neurosci Methods 172(1):137–141. doi:10.1016/j.jneumeth.2008.04.012
- Chen R, Jiao Y, Zhu JS, Wang XH, Zhao MT (2024). Frequency-specific static and dynamic neural activity indices in children with different ADHD subtypes. Front Hum Neurosci 18:1412572. doi:10.3389/fnhum.2024.1412572. https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2024.1412572/full (**ADHD-200; slow-5 fALFF OFC; AUCs**)
- (2025). Frequency-specific alterations in low-frequency functional connectivity in children with ADHD. BMC Psychiatry. doi:10.1186/s12888-025-07586-6. https://link.springer.com/article/10.1186/s12888-025-07586-6 (**slow-3/4/5; AUC 0.755/0.783**)
- Lou Y, et al. (2021). Altered Variability and Concordance of Dynamic Resting-State fMRI Indices in Patients With ADHD. Front Neurosci 15:731596. doi:10.3389/fnins.2021.731596. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.731596/full (**dALFF variability ADHD**)
- Nomi JS, Schettini E, Voorhies WI, Bolt T, Heller AS, Uddin LQ (2018). Resting-State Brain Signal Variability in Prefrontal Cortex Is Associated With ADHD Symptom Severity in Children. Front Hum Neurosci 12. doi:10.3389/fnhum.2018.00090. https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2018.00090/full
- Hong SB, Hwang SS (2022). Resting-State Brain Variability in Youth With ADHD. Front Psychiatry 13:918700. doi:10.3389/fpsyt.2022.918700. https://doi.org/10.3389/fpsyt.2022.918700 (**ADHD-200 DMN instability**)
- (2025). A resting-state fMRI ALE meta-analysis of differences in brain activity between children and adolescents with ADHD. Eur Child Adolesc Psychiatry. doi:10.1007/s00787-025-02906-3. https://link.springer.com/article/10.1007/s00787-025-02906-3 (**28 studies; child/adolescent divergence**)
- Shu YP, et al. (2025). Vulnerable brain regions in adolescent ADHD: An ALE meta-analysis. World J Psychiatry 15(4):102215. https://www.wjgnet.com/2220-3206/full/v15/i4/102215.htm (**lingual/cuneus ↑; L MFG/L PCUN ↓**)
- (2023). Meta-analysis of structural and functional alterations of brain in patients with ADHD. Front Psychiatry 14:1070142. doi:10.3389/fpsyt.2022.1070142. https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2022.1070142/full (SDM; L MOG overactivation)
- (2023). Frequency characteristics of temporal and spatial concordance among dynamic indices in inattentive and combined subtypes of ADHD. Front Neurosci 17:1196290. doi:10.3389/fnins.2023.1196290 (ADHD-200; slow-5 concordance)
- (2024). fNIRS ALFF/FC in ADHD subtypes. BMC Psychiatry. doi:10.1186/s12888-024-06350-6. https://link.springer.com/article/10.1186/s12888-024-06350-6
- Bellec P, et al. (2016). The Neuro Bureau ADHD-200 Preprocessed Repository. doi:10.1101/037044. https://doi.org/10.1101/037044 (**fALFF/ReHo maps + CC200 parcellations shipped**)

**Reliability**
- Colclough GL, et al. (2016). How reliable are MEG resting-state connectivity metrics? NeuroImage. https://www.sciencedirect.com/science/article/pii/S1053811916301914 (**AEC best; phase metrics worst**)
- Garcés P, Martín-Buro MC, Maestú F (2016). Quantifying the Test-Retest Reliability of MEG resting-state FC. Brain Connect. doi:10.1089/brain.2015.0416. https://doi.org/10.1089/brain.2015.0416
- (2020). Test-Retest Reliability of MEG resting-state FC over 1 hour/1 week in psychosis. Front Psychiatry 11:551952. doi:10.3389/fpsyt.2020.551952
- (2022). Sensitive and reproducible MEG resting-state metrics of FC in Alzheimer's disease. Alzheimer's Res Ther. doi:10.1186/s13195-022-00970-4. https://link.springer.com/article/10.1186/s13195-022-00970-4 (**AEC-c definition + most reproducible; ES −0.3/−0.4**)
- (2022). Reproducibility of graph measures from MEG FC metrics. PubMed 35019189. https://pubmed.ncbi.nlm.nih.gov/35019189/ (**amplitude metrics most reliable**)
- (2022). Test-retest reliability of the human connectome: OPM-MEG. PMC12007552. https://pmc.ncbi.nlm.nih.gov/articles/PMC12007552/ / bioRxiv 2022.12.21.521184
- (2022). The effect of general anesthesia on test-retest reliability of rs-fMRI metrics and optimization of scan length. Front Neurosci 16:937172. doi:10.3389/fnins.2022.937172. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.937172/full (**ALFF .81 / fALFF .51 / FC .65 / ReHo .84; scan-length curves**)
- (2015). Low-Frequency Fluctuations of the Resting Brain: high ALFF magnitude ↔ poor reliability. PLOS ONE. doi:10.1371/journal.pone.0128117. https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0128117 (**~156 s ALFF minimum; DMN worst**)
- (2018). Intra- and Inter-Scanner Reliability of Voxel-Wise Whole-Brain Analytic Metrics for rs-fMRI. Front Neuroinform 12:54. doi:10.3389/fninf.2018.00054. https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00054/full (**PerAF best inter-scanner; multisite warning**)
- Birn RM, et al. (2013). The effect of scan length on reliability of rs-fMRI FC. PMC4104183. https://pmc.ncbi.nlm.nih.gov/articles/PMC4104183/ (**ICC still rising to 12–16 min**)
- (2022). Comparing test-retest reliability of rs-fMRI metrics across single-band and multiband (Antwerp). https://repository.uantwerpen.be/docstore/d:irua:16068 (**ALFF ICC 46–81; MB>SB cortical; ALFF>FC**)
- Zuo XN, et al. (2010). The oscillating brain: complex and reliable. PubMed 19782143. https://pubmed.ncbi.nlm.nih.gov/19782143/ (**slow-4/5 definitions; slow-4 subcortical; ALFF reliability**)

**Bands / physiology / fast fMRI**
- Buzsáki G, Draguhn A (2004). Neuronal oscillations in the intact brain. Science 304:1926–1933.
- Cordes D, et al. (2001). Frequencies contributing to functional connectivity in "resting-state" data. AJNR 22(7):1326–1333. (respiration 0.1–0.3 Hz; cardiac 0.6–1.1 Hz)
- Figueroa CA, et al. (2020). Ghost Attractors in Spontaneous Brain Activity (LEiDA; reliability maximized >0.1 Hz). Front Syst Neurosci 14:20. doi:10.3389/fnsys.2020.00020. https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2020.00020/full
- (2022). Test-retest reliability of time-varying patterns (LEiDA, older adults, SB vs MB). Front Hum Neurosci. doi:10.3389/fnhum.2022.980280. https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2022.980280/full
- (2022). Comparing sliding window correlation and instantaneous phase coherence in dFC (PC vs SW; LEiDA). ISMRM 2022. https://isr.tecnico.ulisboa.pt/wp-content/uploads/2023/03/ISMRM-2022_papper.pdf
- (2013). Resting-state fMRI at 4 Hz: Morlet band-limited power envelopes, 0.1–4 Hz. ISMRM 2013. https://cds.ismrm.org/protected/13MProceedings/PDFfiles/0041.PDF
- (2017). Investigation of true high-frequency electrical substrates of RSNs (MB8, TR=200 ms, pICA). Front Neuroinform 11:74. doi:10.3389/fninf.2017.00074. https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2017.00074/full
- (2019). On the analysis of rapidly sampled fMRI data. PMC6984348. https://pmc.ncbi.nlm.nih.gov/articles/PMC6984348/
- Glerean S, et al. (2012). Functional MRI phase synchronization analysis (0.04–0.07 Hz narrowband). NeuroImage 60(2). [as described in Mital 2023]

**Variability / dALFF**
- (2020). Dynamic Alterations of ALFF in drug-naïve first-episode early-onset schizophrenia. Front Neurosci 14:901. doi:10.3389/fnins.2020.00901. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00901/full (**dALFF-variability protocol: 50-TR windows, SD, z-score**)
- (2018). Characterizing Dynamic ALFF and its relationship with dFC: application to schizophrenia. PMC5860934. https://pmc.ncbi.nlm.nih.gov/articles/PMC5860934/ (dALFF states; dALFF–dFC coupling)
- Nomi JS, et al. (2017). Moment-to-Moment BOLD Signal Variability Reflects Regional Changes in Neural Flexibility across the Lifespan. J Neurosci 37(22):5539–5548. doi:10.1523/JNEUROSCI.3408-16.2017. https://www.jneurosci.org/content/37/22/5539 (**MSSD formula**)
- (2023). The Role of BOLD Signal Variability in Pediatrics (review; ADHD/ASD/SZ). Life 13(7):1587. https://www.mdpi.com/2075-1729/13/7/1587

**Phase+amplitude combination**
- Castro E, et al. (2014). MKL classification of groups from complex-valued fMRI: application to schizophrenia (magnitude+phase, +5%). NeuroImage. https://pmc.ncbi.nlm.nih.gov/articles/PMC3946896/
- Li L, Wang X, Li J, Zhao Y (2023). An EEG-based marker of FC: detection of MDD (PLV + synchrosqueezed wavelet coherence fusion). Cognitive Neurodynamics. doi:10.1007/s11571-023-10041-5. https://doi.org/10.1007/s11571-023-10041-5
- Mital 2023 (above; IA+IP fusion, 3–20% consistency gain)
- Cabral J, et al. (2017). LEiDA original. Sci Rep 7:5135. https://github.com/juanitacabral/LEiDA ; Python: https://github.com/PSYMARKER/leida-python

---

*Report compiled from web searches via Exa (native web_search unavailable in this session: no API key configured). All quantitative claims above are traceable to the cited sources; where evidence is absent (ADHD envelope-correlation fMRI; fMRI amplitude-vs-phase reliability head-to-head; slow-3 amplitude in ADHD), this is stated explicitly.*
