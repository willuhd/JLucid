"""Adversarial audit of the adhd200 dispersion-C finding (dD=+.46, p=.0016).

Q1 Pittsburgh leverage: drop Pittsburgh (4 ADHD), recompute pooled test.
Q2 Motion-variance mechanism: (a) var(FD) by group — if ADHD motion is more
   VARIABLE, any motion-loaded block shows elevated D mechanically;
   (b) corr(D_C, FD); (c) FD-stratified D comparison; (d) motion-matched pairs.
Q3 Per-feature decomposition: which of the 14 AC/MC features drive the elevation
   (mean |z| by group per feature)?
Q4 Per-site VR table.
"""
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

d12 = import_module('12_dispersion')


def build():
    df = d12.load_adhd200()
    c_cols = [c for c in df.columns if c.startswith(('AC_', 'MC_'))]
    sub = df.dropna(subset=c_cols + ['age', 'sex_c']).copy()
    is_ctrl = (sub.dx == 0).values
    D, kept = d12.normative_D(sub[c_cols].values, sub.age.values, sub.sex_c.values, is_ctrl)
    sub['D'] = D
    return sub, c_cols


def main():
    sub, c_cols = build()
    print(f'n={len(sub)} sites={sorted(sub.site.unique())}')
    # Q1: drop Pittsburgh
    no_pitt = sub[sub.site != 'Pittsburgh'].copy()
    r = d12.perm_p_within_site(no_pitt.D.values, no_pitt.dx.values, no_pitt.site.values, seed=7)
    print(f'[Q1 drop Pittsburgh] dD={r["dD"]:+.4f} VR={r["VR"]:.2f} d={r["d"]:+.3f} p={r["p"]:.4f} (n={r["n_ad"]}/{r["n_ct"]})')
    # motion availability
    man = pd.read_csv(f'{d12.ROOT}/exp_leida_xverify/results/cache/adhd200_cc190_manifest.csv', dtype={'sid': str})
    print('manifest motion cols:', [c for c in man.columns if 'motion' in c.lower() or 'fd' in c.lower() or 'FD' in c])
    # FD per subject: NYU real FD from controllability csv; others: maxmotion if present
    fd = pd.Series(np.nan, index=sub.index)
    nyu = pd.read_csv(f'{d12.ROOT}/exp_leida_xverify/results/controllability_p6_nyu.csv', dtype={'sid': str}).set_index('sid')
    ix = sub[sub.site == 'NYU'].index.intersection(nyu.index)
    fd.loc[ix] = nyu.loc[ix, 'fd'].values
    mm = man.sort_values(['sid', 'run']).drop_duplicates('sid').set_index('sid')
    # check for a maxmotion-like column usable per subject
    for c in mm.columns:
        if 'motion' in c.lower():
            vals = pd.to_numeric(mm[c], errors='coerce')
            common = sub.index.intersection(vals.dropna().index)
            print(f'  manifest[{c}]: {len(common)} subjects overlap, by site:',
                  {s: int(((sub.site == s) & sub.index.isin(common)).sum()) for s in sorted(sub.site.unique())})
    sub['fd'] = fd
    # Q2a: FD variance by group (NYU only, where FD is real)
    nys = sub[sub.site == 'NYU'].copy()
    nys = nys[nys.fd.notna()]
    print(f'[Q2a NYU var(FD): ADHD={nys[nys.dx==1].fd.var():.4f} (n={(nys.dx==1).sum()}) '
          f'TD={nys[nys.dx==0].fd.var():.4f} (n={(nys.dx==0).sum()})]')
    # Q2b: corr(D, FD)
    ok = nys.fd.notna() & nys.D.notna()
    print(f'[Q2b NYU corr(D_C, FD)]: overall {nys[ok].D.corr(nys[ok].fd, method="spearman"):+.3f} | '
          f'ADHD {nys[ok & (nys.dx==1)].D.corr(nys[ok & (nys.dx==1)].fd, method="spearman"):+.3f} | '
          f'TD {nys[ok & (nys.dx==0)].D.corr(nys[ok & (nys.dx==0)].fd, method="spearman"):+.3f}')
    # Q2c: FD-quartile D means (NYU)
    nys['fdq'] = pd.qcut(nys.fd, 4, labels=False, duplicates='drop')
    print('[Q2c NYU D by FD quartile x dx]:')
    print(nys.groupby(['fdq', 'dx']).D.mean().round(3).to_string())
    # Q3: per-feature mean|z| by group (pooled, all sites)
    print('\n[Q3 per-feature drivers] (mean|z| ADHD vs TD, pooled z from the normative fit)')
    X = sub[c_cols].values.astype(float)
    age = sub.age.values; sex = sub.sex_c.values
    is_ctrl = (sub.dx == 0).values
    rows = []
    for k, c in enumerate(c_cols):
        okk = np.isfinite(X[:, k]) & np.isfinite(age) & np.isfinite(sex) & is_ctrl
        C = np.column_stack([np.ones(okk.sum()), age[okk], sex[okk]])
        b, *_ = np.linalg.lstsq(C, X[okk, k], rcond=None)
        r = X[okk, k] - C @ b
        sig = 1.4826 * np.median(np.abs(r - np.median(r)))
        if not np.isfinite(sig) or sig < 1e-8:
            continue
        z = np.clip((X[:, k] - (b[0] + b[1] * age + b[2] * sex)) / sig, -3, 3)
        okf = np.isfinite(z)
        ad = (sub.dx.values == 1) & okf
        ct = (sub.dx.values == 0) & okf
        rows.append((c, abs(z[ad]).mean(), abs(z[ct]).mean(), ad.sum(), ct.sum()))
    tab = pd.DataFrame(rows, columns=['feat', 'mabs_ADHD', 'mabs_TD', 'n_ad', 'n_ct'])
    tab['diff'] = tab.mabs_ADHD - tab.mabs_TD
    print(tab.sort_values('diff', ascending=False).round(3).to_string(index=False))
    # Q4: per-site VR
    print('\n[Q4 per-site VR (block C D)]')
    for s in sorted(sub.site.unique()):
        ss = sub[sub.site == s]
        if (ss.dx == 1).sum() >= 4:
            print(f'  {s}: VR={ss[ss.dx==1].D.var(ddof=1)/max(ss[ss.dx==0].D.var(ddof=1),1e-12):.2f} '
                  f'(n_ad={(ss.dx==1).sum()}, n_ct={(ss.dx==0).sum()})')


if __name__ == '__main__':
    main()
