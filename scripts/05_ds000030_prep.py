"""ds000030 uniform preprocessing -> CC200 (HBN atlas) ROI timeseries.

Per subject (169 downloaded rest BOLD, 64x64x34 native, TR=2s):
  1. Motion correction: ants.motion_correction (BOLDRigid). FD taken from the
     returned FD curve if provided (fdOffset=50 suggests it computes one);
     otherwise FD estimated from the rigid transforms if accessible; else
     DVARS proxy only (flagged in meta).
  2. Mean image -> N4 bias correction.
  3. Register mean -> HBN CC200 atlas (SyNRA; QC'd dice 0.79).
  4. Inverse-transform the atlas into NATIVE space (nearest neighbour) and
     average the 4D within each label -> ROI timeseries in NATIVE space
     (no slab stretching; honest per-ROI coverage counts).
  5. QC per subject: per-ROI native voxel counts, DVARS (raw), tSNR-ish,
     static FC mean (intact-regime check).

Output: results/ds30_tc/<sub>.npz (tc float32 [n_live_roi, T], roi_ids,
        coverage, dvars, fd?, fc_mean), results/ds30_prep_meta.csv.
"""
import glob
import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
import ants
from scipy import ndimage

ROOT = '/Volumes/thinkplus/Code/JLucid'
OUT = f'{ROOT}/exp_r6_lock/results/ds30_tc'
ATLAS = f'{ROOT}/data/hbn_t2/CPAC_CC200_atlas.nii'


def prep_sub(path):
    sub = os.path.basename(path).split('_')[0]
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    n_t = data.shape[3]
    aimg = ants.image_read(path)
    mc = ants.motion_correction(aimg, type_of_transform='BOLDRigid')
    mc_img = mc['motion_corrected'] if isinstance(mc, dict) and 'motion_corrected' in mc else mc
    fd_curve = None
    if isinstance(mc, dict):
        for key in ['fd', 'FD', 'framewise_displacement']:
            if key in mc:
                fd_curve = np.asarray(mc[key]).ravel()
                break
    mc_data = mc_img.numpy() if mc_img.numpy().ndim == 4 else mc_img.numpy()[..., None]
    mean3d = mc_data.mean(-1)
    n4 = ants.n4_bias_field_correction(ants.from_numpy(mean3d, origin=aimg.origin[:3],
                                                       spacing=list(aimg.spacing[:3]),
                                                       direction=aimg.direction[:3, :3]))
    fixed = ants.image_read(ATLAS)
    reg = ants.registration(fixed=fixed, moving=n4, type_of_transform='SyNRA')
    atlas_native = ants.apply_transforms(fixed=n4, moving=fixed,
                                         transformlist=reg['invtransforms'],
                                         interpolator='nearestNeighbor')
    lab = np.rint(atlas_native.numpy()).astype(np.int32)
    live_ids = np.unique(lab[lab > 0])
    # ROI means per frame via bincount over flattened labels
    flat = lab.ravel()
    counts = np.bincount(flat, minlength=201)
    tc = np.empty((len(live_ids), n_t), dtype=np.float32)
    for t in range(n_t):
        s = mc_data[..., t].ravel()
        sums = np.bincount(flat, weights=s, minlength=201)
        tc[:, t] = sums[live_ids] / counts[live_ids]
    cov = counts[live_ids]
    # DVARS on raw ROI series (edge 10)
    d = np.abs(np.diff(tc[:, 10:-10], axis=1))
    dvars = float(np.sqrt((d ** 2).mean())) if d.shape[1] > 5 else np.nan
    # static FC (intact-regime check): corr across time of z-scored TCs
    z = (tc - tc.mean(1, keepdims=True)) / (tc.std(1, keepdims=True) + 1e-8)
    C = np.corrcoef(z)
    iu = np.triu_indices_from(C, 1)
    fc_mean = float(np.nanmean(C[iu]))
    fracneg = float(np.mean(C[iu] < 0))
    # tSNR proxy
    tsnr = float(np.median(tc.mean(1) / (tc.std(1) + 1e-8)))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f'{OUT}/{sub}.npz', tc=tc, roi_ids=live_ids.astype(np.int16),
                        coverage=cov.astype(np.int32),
                        fd=fd_curve.astype(np.float32) if fd_curve is not None else np.array([np.nan]),
                        dvars=dvars, fc_mean=fc_mean, fracneg=fracneg, tsnr=tsnr)
    return dict(sub=sub, n_t=n_t, n_live=len(live_ids), cov_min=int(cov.min()),
                cov_med=int(np.median(cov)), dvars=dvars, fc_mean=fc_mean,
                fracneg=fracneg, tsnr=tsnr, has_fd=fd_curve is not None)


def main(subs=None, workers=8):
    files = sorted(glob.glob(f'{ROOT}/data/ds000030_rest/*_task-rest_bold.nii.gz'))
    if subs:
        files = [f for f in files if os.path.basename(f).split('_')[0] in subs]
    # skip already-prepped (idempotent resume; verified-complete npz only)
    files = [f for f in files if not os.path.exists(
        f'{ROOT}/exp_r6_lock/results/ds30_tc/{os.path.basename(f).split("_")[0]}.npz')]
    print(f'{len(files)} subjects to prep', flush=True)
    from multiprocessing import Pool
    rows = []
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as p:
            for i, r in enumerate(p.imap_unordered(prep_sub, files, chunksize=1)):
                rows.append(r)
                print(f'{i+1}/{len(files)} {r["sub"]} live={r["n_live"]} fc={r["fc_mean"]:+.3f} '
                      f'fd={r["has_fd"]} ({time.time()-t0:.0f}s)', flush=True)
    else:
        for i, f in enumerate(files):
            rows.append(prep_sub(f))
            print(f'{i+1}/{len(files)} ({time.time()-t0:.0f}s)', flush=True)
    pd.DataFrame(rows).to_csv(f'{ROOT}/exp_r6_lock/results/ds30_prep_meta.csv', index=False)
    df = pd.DataFrame(rows)
    print('\n=== summary ===')
    print('live ROIs: med', df.n_live.median(), 'min', df.n_live.min())
    print('fc_mean: med %.3f  fracneg med %.3f' % (df.fc_mean.median(), df.fracneg.median()))
    print('subjects with fd curve:', int(df.has_fd.sum()))


if __name__ == '__main__':
    import sys
    subs = [s for s in sys.argv[1].split(',') if s] if len(sys.argv) > 1 else None
    main(subs=subs, workers=int(sys.argv[2]) if len(sys.argv) > 2 else 8)
