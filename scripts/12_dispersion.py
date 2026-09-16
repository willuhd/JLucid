"""Dispersion-D test (second moment of control-normative deviations).

Faithful to the rough-drafts spec (project_rough_drafts.pdf Sec 2.4, Eqs 22-26)
applied to THIS project's validated cached features (not the overnight's
7-net tables, which are not on disk as artifacts):
- Control-only normative models per site (adhd200) / cohort (Penn, HBN-split):
  feature_k ~ age + sex, OLS on controls.
- Robust scale: sigma_k = 1.4826 * MAD(control residuals); skip degenerate.
- z_ik clipped to +-3; D_i = mean_k z^2 over block K.
- Stats: dD (mean ADHD - mean control), VR (variance ratio), Cohen's d,
  permutation (within-site adhd200; full shuffle Penn), 5000 perms, two-sided +1.

Cohorts (argv): adhd200 | penn | hbn
Blocks: C = controllability (AC/MC nets), G = global scalars (gcoh/gcoh_sd/lam),
        S = state metrics (occ/life/switch).
"""
import sys
import numpy as np
import pandas as pd

ROOT = '/Volumes/thinkplus/Code/JLucid'
N_PERM = 5000


def normative_D(X, age, sex, is_ctrl, groups=None, seed=0):
    """X: (n, p) features. Returns D (n,), per-feature sigma, n_skipped."""
    X = np.asarray(X, float)
    age = np.asarray(age, float); sex = np.asarray(sex, float)
    n, p = X.shape
    D = np.zeros(n)
    kept = 0
    Z = np.zeros_like(X)
    for k in range(p):
        ok = np.isfinite(X[:, k]) & np.isfinite(age) & np.isfinite(sex) & is_ctrl
        if ok.sum() < 10:
            continue
        C = np.column_stack([np.ones(ok.sum()), age[ok], sex[ok]])
        b, *_ = np.linalg.lstsq(C, X[ok, k], rcond=None)
        r = X[ok, k] - C @ b
        mad = np.median(np.abs(r - np.median(r)))
        sig = 1.4826 * mad
        if not np.isfinite(sig) or sig < 1e-8:
            continue
        pred = b[0] + b[1] * age + b[2] * sex
        z = (X[:, k] - pred) / sig
        z = np.clip(z, -3, 3)
        Z[:, k] = np.where(np.isfinite(X[:, k]), z, 0.0)
        kept += 1
    if kept == 0:
        return np.full(n, np.nan), 0
    D = (Z ** 2).mean(1)
    return D, kept


def perm_p_within_site(D, y, site, n_perm=N_PERM, seed=0):
    """Two-sided permutation p for mean(ADHD)-mean(control), shuffling within site."""
    D = np.asarray(D, float); y = np.asarray(y, float)
    ok = np.isfinite(D) & np.isfinite(y)
    D, y = D[ok], y[ok]
    site = np.asarray(site)[ok]
    obs = D[y == 1].mean() - D[y == 0].mean()
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        yp = y.copy()
        for s in np.unique(site):
            m = site == s
            yp[m] = rng.permutation(yp[m])
        d = D[yp == 1].mean() - D[yp == 0].mean()
        cnt += abs(d) >= abs(obs)
    p = (cnt + 1) / (n_perm + 1)
    ad = y == 1
    vr = D[ad].var(ddof=1) / max(D[~ad].var(ddof=1), 1e-12)
    sp = np.sqrt(((ad.sum() - 1) * D[ad].var(ddof=1) + ((~ad).sum() - 1) * D[~ad].var(ddof=1)) / (len(y) - 2))
    dco = (D[ad].mean() - D[~ad].mean()) / sp
    return dict(dD=float(obs), VR=float(vr), d=float(dco), p=float(p),
                n_ad=int(ad.sum()), n_ct=int((~ad).sum()))


def load_adhd200():
    a = pd.read_csv(f'{ROOT}/exp_leida_xverify/results/acmc_scalars_adhd_sites.csv', dtype={'sid': str})
    m = pd.read_csv(f'{ROOT}/exp_leida_xverify/results/cache/adhd200_cc190_manifest.csv', dtype={'sid': str})
    m['Inattentive'] = pd.to_numeric(m['Inattentive'], errors='coerce')
    mm = m.sort_values(['sid', 'run']).drop_duplicates('sid').set_index('sid')
    df = a.set_index('sid')
    for c in ['Age', 'Gender', 'DX', 'Med Status']:
        df[c] = mm[c]
    df['age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['sex_c'] = (pd.to_numeric(df['Gender'], errors='coerce') == 1).astype(float)
    df['dx'] = df['DX'].isin(['1', '3', '2']).astype(float)
    return df


def run_adhd200():
    df = load_adhd200()
    c_cols = [c for c in df.columns if c.startswith(('AC_', 'MC_'))]
    g_cols = [c for c in ['gcoh', 'gcoh_sd', 'lam_max', 'fc_mean'] if c in df.columns]
    print(f'adhd200: n={len(df)} sites={sorted(df.site.unique())} features C={len(c_cols)} G={len(g_cols)}')
    print('dx counts:', df.dx.value_counts(dropna=False).to_dict())
    for bname, cols in [('C', c_cols), ('G', g_cols)]:
        sub = df.dropna(subset=cols + ['age', 'sex_c']).copy()
        is_ctrl = (sub.dx == 0).values
        D, kept = normative_D(sub[cols].values, sub.age.values, sub.sex_c.values, is_ctrl)
        sub['D'] = D
        print(f'\n--- block {bname} ({kept} features kept) ---')
        r = perm_p_within_site(sub.D.values, sub.dx.values, sub.site.values)
        print(f'ALL ADHD vs TD: dD={r["dD"]:+.4f} VR={r["VR"]:.2f} d={r["d"]:+.3f} p={r["p"]:.4f} (n={r["n_ad"]}/{r["n_ct"]})')
        # per-site breakdown
        for s in sorted(sub.site.unique()):
            ss = sub[sub.site == s]
            if (ss.dx == 1).sum() >= 4 and (ss.dx == 0).sum() >= 10:
                d = ss[ss.dx == 1].D.mean() - ss[ss.dx == 0].D.mean()
                print(f'  {s}: dD={d:+.4f} (n_ad={(ss.dx==1).sum()}, n_ct={(ss.dx==0).sum()})')
        # medication sensitivity: exclude Med Status == '1'
        nomed = sub[sub['Med Status'] != '1'].copy()
        if (nomed.dx == 1).sum() > 30:
            r2 = perm_p_within_site(nomed.D.values, nomed.dx.values, nomed.site.values, seed=1)
            print(f'  excl Med==1: dD={r2["dD"]:+.4f} VR={r2["VR"]:.2f} d={r2["d"]:+.3f} p={r2["p"]:.4f} (n={r2["n_ad"]}/{r2["n_ct"]})')


def run_penn():
    import re
    # rest state metrics: assign rest frames to the rest-k5 dictionary
    d = np.load(f'{ROOT}/exp_leida_xverify/results/eigvec_penn_rest.npz')
    dk = np.load(f'{ROOT}/exp_leida_xverify/results/eigvec_penn_rest_k5.npz')
    C5 = dk['centroids'].astype(np.float64)
    sc = pd.read_csv(f'{ROOT}/exp_leida_xverify/results/eigvec_penn_scalars.csv', dtype={'sid': str})
    sc['s'] = sc.sid.str.replace('sub-', '', regex=False)
    sc = sc.set_index('s')
    cp = pd.read_csv(f'{ROOT}/exp_leida_xverify/results/controllability_p6_penn.csv', dtype={'sid': str})
    cp['s'] = cp.sid.str.replace('sub-', '', regex=False)
    cp = cp.set_index('s')
    co = pd.read_csv(f'{ROOT}/data/pennlead/pheno/cohort.csv', dtype={'participant_id': str})
    co['s'] = co.participant_id.str.replace('sub-', '', regex=False)
    co = co.set_index('s')
    nb = pd.read_csv(f'{ROOT}/exp_leida_xverify/results/eigvec_penn_nback_k5_subjects.csv')
    nb['s'] = nb.sid.astype(str).str.replace('sub-', '', regex=False)
    nb = nb.set_index('s')

    def assign_rest(sid):
        key = f'V_sub-{sid}'
        if key not in d:
            return None
        V = d[key].astype(np.float64)
        lab = ((V[:, None, :] - C5[None, :, :]) ** 2).sum(-1).argmin(1)
        out = {}
        for s_ in range(5):
            out[f'occ_{s_}'] = float((lab == s_).mean())
            is_s = (lab == s_).astype(int)
            runs, cur = [], 0
            for v in is_s:
                if v:
                    cur += 1
                elif cur:
                    runs.append(cur); cur = 0
            if cur:
                runs.append(cur)
            out[f'life_{s_}'] = float(np.mean(runs)) if runs else 0.0
        out['switch'] = float((lab[1:] != lab[:-1]).mean())
        return out

    rest_rows = []
    for s in sc.index:
        m = assign_rest(s)
        if m:
            m['s'] = s
            rest_rows.append(m)
    rest = pd.DataFrame(rest_rows).set_index('s')
    print(f'Penn rest state metrics: {len(rest)} subjects')

    S = [f'occ_{i}' for i in range(5)] + [f'life_{i}' for i in range(5)] + ['switch']
    G = ['gcoh_rest', 'gcoh_sd_rest']
    Cnets = [c for c in cp.columns if c.startswith(('AC_', 'MC_'))]
    base = pd.DataFrame(index=rest.index)
    base = base.join(rest[S]).join(sc[G]).join(cp[Cnets])
    base = base.join(co[['study_group', 'age', 'sex', 'dx_adhd']])
    base['sex_c'] = (base.sex == 'M').astype(float)
    base['age'] = pd.to_numeric(base.age, errors='coerce')

    # nback metrics
    nS = [f'occ_{i}' for i in range(5)] + [f'life_{i}' for i in range(5)] + ['switch']
    nG = ['gcoh_nb', 'gcoh_sd_nb']
    nb2 = nb[nS + nG].copy()
    nb2.columns = [c + '_nb' if not c.endswith('_nb') else c for c in nb2.columns]
    # rename switch_nb collision: nback file already has switch_nb
    base = base.join(nb2, how='left')
    print(f'with nback: {int(base[[c for c in base.columns if c.endswith("_nb")]].notna().any(axis=1).sum())} subjects')

    def test_block(df, cols, y, tag, seed=0):
        sub = df.dropna(subset=cols + ['age', 'sex_c']).copy()
        is_ctrl = (sub.study_group == 'TD/NC').values
        if is_ctrl.sum() < 10:
            print(f'{tag}: too few TD/NC controls ({is_ctrl.sum()}), skip');
            return None
        D, kept = normative_D(sub[cols].values, sub.age.values, sub.sex_c.values, is_ctrl)
        sub['D'] = D
        r = perm_p_within_site(sub.D.values, y[sub.index].values if isinstance(y, pd.Series) else y,
                               np.zeros(len(sub)), seed=seed)
        # full shuffle: single site -> perm_p_within_site with constant site == full shuffle
        print(f'{tag} ({kept} feats): dD={r["dD"]:+.4f} VR={r["VR"]:.2f} d={r["d"]:+.3f} p={r["p"]:.4f} '
              f'(n_ad={r["n_ad"]}, n_ct={r["n_ct"]})')
        return sub[['D']].assign(tag=tag)

    y_broad = (base.dx_adhd == 1).astype(float)
    y_clean = base.study_group.map({'ADHD': 1.0, 'TD/NC': 0.0})
    print('\n--- Penn REST (normative: TD/NC) ---')
    test_block(base, S, y_broad, 'rest-S broad')
    test_block(base, G, y_broad, 'rest-G broad')
    test_block(base, Cnets, y_broad, 'rest-C broad')
    cl = base[base.study_group.isin(['ADHD', 'TD/NC'])].copy()
    test_block(cl, S, (cl.study_group == 'ADHD').astype(float), 'rest-S clean')
    print('\n--- Penn NBACK (normative: TD/NC) ---')
    nS_nb = [c for c in base.columns if re.match(r'(occ|life)_[0-4]_nb$', c)] + (['switch_nb'] if 'switch_nb' in base.columns else [])
    nG_nb = [c for c in ['gcoh_nb', 'gcoh_sd_nb'] if c in base.columns]
    test_block(base, nS_nb, y_broad, 'nback-S broad')
    test_block(base, nG_nb, y_broad, 'nback-G broad')
    print('\n--- Penn rest->nback WITHIN-SUBJECT change ---')
    # D per run with run-specific normative models, then change between groups
    for bname, rcols, ncols in [('S', S, nS_nb), ('G', G, nG_nb)]:
        # rest D
        sub = base.dropna(subset=rcols + ['age', 'sex_c']).copy()
        is_ctrl = (sub.study_group == 'TD/NC').values
        Dr, _ = normative_D(sub[rcols].values, sub.age.values, sub.sex_c.values, is_ctrl)
        # nback D
        sub2 = base.dropna(subset=ncols + ['age', 'sex_c']).copy()
        is_ctrl2 = (sub2.study_group == 'TD/NC').values
        Dn, _ = normative_D(sub2[ncols].values, sub2.age.values, sub2.sex_c.values, is_ctrl2)
        common = sorted(set(sub.index) & set(sub2.index))
        chg = pd.Series(Dn[sub2.index.get_indexer(common)] if False else
                        pd.Series(Dn, index=sub2.index).loc[common].values - pd.Series(Dr, index=sub.index).loc[common].values,
                        index=common)
        sg = base.loc[common, 'study_group']
        keep = sg.isin(['ADHD', 'TD/NC'])
        if keep.sum() < 20:
            # fall back to broad groups
            dxb = base.loc[common, 'dx_adhd']
            keep = dxb.notna()
            yy = (dxb[keep] == 1).astype(float).values
            lab = 'broad'
        else:
            yy = (sg[keep] == 'ADHD').astype(float).values
            lab = 'clean'
        xx = chg[keep].values
        ok = np.isfinite(xx) & np.isfinite(yy)
        rng = np.random.default_rng(0)
        obs = xx[ok & (yy == 1)].mean() - xx[ok & (yy == 0)].mean()
        cnt = 0
        for _ in range(5000):
            yp = rng.permutation(yy[ok])
            dd = xx[ok][yp == 1].mean() - xx[ok][yp == 0].mean()
            cnt += abs(dd) >= abs(obs)
        print(f'change-{bname} ({lab}, n={int(ok.sum())}): dD_nb-rest ADHD-TD={obs:+.4f} p={(cnt+1)/5001:.4f}')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'adhd200'
    if which == 'adhd200':
        run_adhd200()
    elif which == 'penn':
        run_penn()
    else:
        print('cohort Penn/HBN handled in part 2')
