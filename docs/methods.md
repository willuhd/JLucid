# Methods (summary; full text in essay/paper.tex)

Data: ADHD-200 resting fMRI, CC200-190 (k=5 lens) and AAL-90 (k=3 J-space lens, Athena).
LEiDA: Hilbert phase -> iPA matrix C(t)=cos(phi_i-phi_j) (rank 2) -> leading eigenvector v1(t) (N-vector, Z2 sign gauge).
k=5 lens: max-component gauge, de-novo k-means seed 0, n=665 (309 ADHD/356 TDC), sites Peking pooled/KKI/NeuroIMAGE/NYU, >=120 TRs.
k=3 lens: majority-negative gauge, JLucid4 switching-linear stay field z<-z+A(p)z+b(p), latent 12, GRU window 7, N=307 (124/183).
Stats: FE meta d, within-site-code permutation (2000 draws), FWER (1000), split-half SB, motion matching on per-run max motion.
