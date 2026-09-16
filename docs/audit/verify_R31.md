# R31 verify: item-1 recovered builder is BYTE-EXACT
- Run: `.venv/bin/python exp_overnight_leida/00_recovered/item1_batch15_comb.py` (blocking, local venv).
- Output: motion-matched obs=+0.1713 p=0.0230 (n=98/401); full-unmed obs=+0.2507 p=0.0005 (n=192); VR=2.19.
- VERIFY line: max|regenerated comb - saved npz comb| = 0.0.
- All three match the sieve-table certified cells to 4 decimals. Provenance hole for the ADHD-200 combined artifact: CLOSED (transcribed+hash-verified; original JLucid2 paths mapped in file header).
- Confirmed construction: POOLED permutation (seed 41, 2000, (k+1)/2001) — the weakest of the three nulls; refit p≈0.04 stands as the honest number.
- Next: item-2 Penn audit_b transcription + verify vs batch15c_penn_dispz.npz (B3 section doubles as item-5 verification).
