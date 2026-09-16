"""Per-run FD recovery for HBN (auditor round-1 method).

Key correction vs 01_fd_provenance.py: the within-run FD<->DVARS relation is
NEGATIVE (CPAC Censor:SpikeRegression FD_J 0.5 flattens high-motion frames),
and FD must be edge-aligned to the DVARS proxy (FD[11:11+n]).

Method: dv = RMS frame-diff of the raw cc200 matrix over frames [10, T-10];
fd_seg = fd[11:11+len(dv)]; c_r = corr(fd_seg, dv_r).
Assignment: run-1 if |c1| > |c2| (margin recorded); run-2 if reversed.
Group-level validation: census-known rest_run-1 subjects must show c1 < c2
paired (auditor positive control t=-10.7).

Output: results/fd_recovery_v2.csv (sid, c1, c2, margin, verdict, fd_mean,
        spike_frac) + group-level validation print.
"""
import glob
import os
import re
import numpy as np
import pandas as pd

ROOT = '/Volumes/thinkplus/Code/JLucid'
OUT = f'{ROOT}/exp_r6_lock/results'


def dvars_proxy(arr, edge=10):
    d = np.abs(np.diff(arr[edge:-edge], axis=0))
    return np.sqrt((d ** 2).mean(1))


def load_run(f):
    return pd.read_csv(f, header=None, skiprows=2, sep='\t').values.astype(float)


def corr_aligned(fd, dv):
    n = len(dv)
    if len(fd) < 11 + n:
        return np.nan
    a = fd[11:11 + n]
    m = np.isfinite(a) & np.isfinite(dv)
    if m.sum() < 50:
        return np.nan
    return float(np.corrcoef(a[m], dv[m])[0, 1])


def main():
    runs = {}
    for f in glob.glob(f'{ROOT}/data/hbn_t2/cc200/*_scan_rest_run-*_cc200.csv'):
        m = re.match(r'(sub-NDAR\w+)_ses-1_scan_rest_run-(\d+)_cc200\.csv', os.path.basename(f))
        if m:
            runs.setdefault(m.group(1), {})[int(m.group(2))] = f

    fd_rest = {}
    for f in glob.glob(f'{ROOT}/data/hbn_t2/fd_rest/*/FD_power.1D'):
        sid = os.path.basename(os.path.dirname(f)).replace('_ses-1', '')
        fd_rest[sid] = np.loadtxt(f)

    rows = []
    subs = sorted(set(runs) & set(fd_rest))
    print(f'{len(subs)} subjects with both a run set and an fd_rest file', flush=True)
    for i, s in enumerate(subs):
        row = dict(sid=s)
        dv = {}
        for r, f in runs[s].items():
            try:
                dv[r] = dvars_proxy(load_run(f))
            except Exception:
                pass
        fd = fd_rest[s]
        row['fd_mean'] = float(np.nanmean(fd))
        row['spike_frac'] = float(np.mean(fd > 0.5))
        c1 = corr_aligned(fd, dv[1]) if 1 in dv else np.nan
        c2 = corr_aligned(fd, dv[2]) if 2 in dv else np.nan
        row['c1'], row['c2'] = c1, c2
        if np.isfinite(c1) and np.isfinite(c2):
            row['margin'] = abs(c1) - abs(c2)
            row['verdict'] = 'run1' if abs(c1) > abs(c2) else 'run2'
        else:
            row['margin'] = np.nan
            row['verdict'] = 'single_run_only' if (np.isfinite(c1) or np.isfinite(c2)) else 'none'
        rows.append(row)
        if (i + 1) % 150 == 0:
            print(f'{i+1}/{len(subs)}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/fd_recovery_v2.csv', index=False)
    print('\nverdicts:', df.verdict.value_counts().to_dict())

    # group-level validation with the census positive control
    cen = pd.read_csv(f'{ROOT}/data/hbn_t2/fd_provenance_census.csv')
    cen['s'] = cen.sid.str.replace('_ses-1', '', regex=False)
    known = set(cen[cen.picked == cen.should_be].s)
    ctl = df[df.sid.isin(known) & (df.verdict.isin(['run1', 'run2']))]
    both = ctl.dropna(subset=['c1', 'c2'])
    if len(both) > 30:
        d = both.c1 - both.c2
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        print(f'census-positive-control (n={len(both)}): mean(c1-c2)={d.mean():+.4f} '
              f'paired t={t:+.1f}  [auditor ref: -10.7; negative = run-1 structure confirmed]')
    test = df[~df.sid.isin(known) & (df.verdict.isin(['run1', 'run2']))].dropna(subset=['c1', 'c2'])
    if len(test) > 30:
        d = test.c1 - test.c2
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        print(f'test group (n={len(test)}): mean(c1-c2)={d.mean():+.4f} paired t={t:+.1f} '
              f'[auditor ref: -9.5]')
    r1 = df[df.verdict == 'run1']
    print(f'run-1 FD recoverable: {len(r1)} subjects; fd_mean med {r1.fd_mean.median():.3f}')


if __name__ == '__main__':
    main()
