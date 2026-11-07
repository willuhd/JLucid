# Reproduce

Environment: .venv/bin/python -c with requirements.txt (numpy, scipy, pandas, sklearn, statsmodels, torch, matplotlib, h5py). No GPU needed (stay field trains ~1 s/epoch on CPU).

1. Download data (docs/data-download.md) into data/.
2. o4 effect: .venv/bin/python src/o4_occupancy.py  (expect d=-0.302, perm p=0.001, SB=0.586, refit matched -0.299, literal -0.035, recovery 9/40)
3. Motion: .venv/bin/python src/essay_motion_stats.py  (expect dz=-0.172, 260 pairs)
4. Rank-2/gauge: .venv/bin/python src/rank2_math.py ; .venv/bin/python src/gauge_artifact.py  (expect sigma3/sigma1 ~5.6e-16, flip rates 32.3%/11.3%)
5. J-space: audit + figures via src/jspace/ and essay_figures logic in docs (expect Ck R2 lin 0.556/quad 0.766, coupling -0.322 -> +0.079 after occupancy control)

Figures: essay/figs/*.png are committed; regenerate via the figure producer in src/ (essay_figures).
