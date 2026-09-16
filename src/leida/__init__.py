"""LEiDA/controllability sieve — ADHD dispersion analysis program.

Refactored from JLucid2 exp/37_leida (the experiment container) into real code.
Layout:
    src/leida/            analysis scripts (batch 1-15 + pilots + figures)
    src/leida/atlas/      CC200->Yeo7 parcel mapping (yeo_true_idx.json)
    src/controllability/   ADHD-200 CC200 loader (copied from JLucid2)
    src/pennlead/               PennLEAD loader (copied from JLucid2)
    results/     all outputs: feature caches, gate results, batch results,
                                figures, and input data assets (V1/V2 wavelet caches)
    docs/                       reports (report.md is the main finding writeup)

Entry points (see docs/report.md and docs/sieve-table.md for the full prereg trail):
    make_figures.py        regenerate all 6 figures + figure_stats.json from cached artifacts
    run_batch15_het.py     the certified dispersion analysis (discovery leg)
    build_features.py      rebuild feature caches from raw CC200 timeseries
"""
