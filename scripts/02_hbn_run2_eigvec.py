"""Extract per-frame leading eigenvectors for BOTH HBN rest runs.

Faithful to exp_leida_xverify/scripts/02b (the pipeline that produced the
existing k5 dictionary), with ONE change per the audit: the sign rule is the
LITERAL majority-count rule — flip if more than half the loadings are positive
(the old mean>0 rule differs when loadings are non-uniform).

Pipeline (identical to 02b):
  loadtxt(skiprows=2) -> per-parcel z -> butter bandpass 0.02-0.1 (TR by n_t)
  -> edge trim 10 -> Hilbert phase -> per-frame dPC = outer(c,c)+outer(s,s)
  -> leading eigenvector by |lambda| -> LITERAL majority sign rule
  -> store V (T, 200) float32 per run + per-frame lam.

Output: results/eigvec_hbn_both_runs.npz (V_<sid>_run<r>, lam_<sid>_run<r>),
        results/eigvec_hbn_both_runs_meta.csv.
"""
import glob
import os
import re
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from multiprocessing import Pool

ROOT = '/Volumes/thinkplus/Code/JLucid'
OUT = f'{ROOT}/exp_r6_lock/results'
BAND = (0.02, 0.1)
EDGE = 10

files_glob = {}


def leading_eigvec(tc, tr):
    n_tp, n_roi = tc.shape
    b, a = butter(4, [BAND[0] * 2 * tr, BAND[1] * 2 * tr], btype='bandpass')
    tc_f = filtfilt(b, a, tc, axis=0)
    if EDGE:
        tc_f = tc_f[EDGE:-EDGE]
    ph = np.angle(hilbert(tc_f, axis=0))
    c, s = np.cos(ph), np.sin(ph)
    T = c.shape[0]
    V = np.empty((T, n_roi), dtype=np.float32)
    lam = np.empty((T, 2), dtype=np.float32)
    for t in range(T):
        M = np.outer(c[t], c[t]) + np.outer(s[t], s[t])
        w, vec = np.linalg.eigh(M)
        k = int(np.argmax(np.abs(w)))
        v = vec[:, k].astype(np.float64)
        if (v > 0).sum() > len(v) / 2:   # LITERAL majority-count sign rule
            v = -v
        V[t] = v
        ab = np.abs(w).copy(); ab[k] = -np.inf
        lam[t, 0] = w[k]; lam[t, 1] = w[int(np.argmax(ab))]
    return V, lam, (np.abs(lam[:, 0]) / n_roi).mean()


def job(args):
    sid, r, f = args
    d = np.loadtxt(f, skiprows=2)
    n_t = d.shape[0]
    tr = 1.45 if n_t >= 420 else 0.8
    z = (d - d.mean(0)) / np.where(d.std(0) < 1e-8, 1.0, d.std(0))
    V, lam, v1f = leading_eigvec(z, tr)
    return sid, r, V, lam, v1f, n_t, tr


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = {}
    for f in sorted(glob.glob(f'{ROOT}/data/hbn_t2/cc200/*_scan_rest_run-*_cc200.csv')):
        b = os.path.basename(f)
        m = re.match(r'(sub-NDAR\w+)_ses-1_scan_rest_run-(\d+)_cc200\.csv', b)
        if m:
            runs.setdefault(m.group(1), {})[int(m.group(2))] = f
    jobs = [(s, r, f) for s, rr in sorted(runs.items()) for r, f in sorted(rr.items())]
    print(f'{len(set(runs))} subjects, {len(jobs)} runs', flush=True)
    t0 = time.time()
    Vd, metas = {}, []
    with Pool(8) as p:
        for i, (sid, r, V, lam, v1f, n_t, tr) in enumerate(p.imap_unordered(job, jobs, chunksize=4)):
            Vd[f'V_{sid}_run{r}'] = V
            Vd[f'lam_{sid}_run{r}'] = lam
            metas.append(dict(sid=sid, run=r, file=os.path.basename(f), n_t=n_t, tr=tr,
                              n_frames=V.shape[0], v1frac_mean=float(v1f)))
            if (i + 1) % 200 == 0:
                print(f'{i+1}/{len(jobs)} ({time.time()-t0:.0f}s)', flush=True)
    np.savez_compressed(f'{OUT}/eigvec_hbn_both_runs.npz', **Vd)
    pd.DataFrame(metas).to_csv(f'{OUT}/eigvec_hbn_both_runs_meta.csv', index=False)
    df = pd.DataFrame(metas)
    print(f'done: {len(metas)} runs, v1frac mean {df.v1frac_mean.mean():.3f}, {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
