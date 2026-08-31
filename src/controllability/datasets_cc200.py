"""Real ADHD200 CC200 TC loader — reads Athena filtfix time courses."""
from pathlib import Path
import pandas as pd
import numpy as np
import glob

def load_cc200(data_dir="/Volumes/thinkplus/Code/JLucid/data/adhd200", qc_only=True, use_first_session=True):
    """Load CC200 TCs for all subjects passing QC.
    Returns:
        timeseries_list: list of (T, N) arrays, N~190
        labels: 0=TDC (DX=0), 1=ADHD (DX=1/2/3)
        meta: dict with sites, ages, subject_ids, etc.
    """
    ph_path = Path(data_dir) / "adhd200_preprocessed_phenotypics.tsv"
    df = pd.read_csv(ph_path, sep='\t')
    # Filter QC
    if qc_only:
        df = df[df['QC_Athena']==1]
    # Exclude pending DX
    df = df[df['DX']!='pending']
    df = df.dropna(subset=['DX'])
    # Map DX: 0=TDC, else ADHD
    df['label'] = df['DX'].apply(lambda x: 0 if str(x)=='0' else 1)
    # Build file index: subject_id -> file
    pattern = str(Path(data_dir) / "*" / "*" / "sfnwmrda*_cc200_TCs.1D")
    # Need recursive for Peking_*
    all_files = glob.glob(str(Path(data_dir) / "*" / "*" / "*.1D"))
    # Filter sfnwmrda only (filtered)
    sfn_files = [f for f in all_files if "sfnwmrda" in f and "cc200" in f]
    # Index by subject dir name int
    file_by_subj = {}
    for f in sfn_files:
        subj_dir = Path(f).parent.name  # e.g., 0026001
        try:
            subj_int = int(subj_dir)
        except:
            continue
        # keep first file per subject (sorted)
        if subj_int not in file_by_subj:
            file_by_subj[subj_int] = f
        else:
            # if multiple, keep first encountered but sorted alphabetically keeps earliest
            # we already sorted? ensure we keep earliest
            pass
    # Now match
    timeseries = []
    labels = []
    sites = []
    ages = []
    subject_ids = []
    skipped = 0
    for _, row in df.iterrows():
        sid = int(row['ScanDir ID'])
        fpath = file_by_subj.get(sid)
        if fpath is None:
            # try alternative: maybe file uses 7-digit padded version, but sid already int, so same
            # also try searching for sid in file path substring
            # Fallback: glob search for *sid*.1D
            candidates = [x for x in sfn_files if str(sid) in x]
            if candidates:
                fpath = sorted(candidates)[0]
            else:
                skipped += 1
                continue
        # Load TC file: header File, Sub-brick, then 190 Mean columns
        try:
            df_tc = pd.read_csv(fpath, sep='\t')
            tc_cols = [c for c in df_tc.columns if c.startswith('Mean_')]
            tc = df_tc[tc_cols].values.astype(np.float64)  # T x N
            # tc already filtered, but ensure no NaN
            tc = np.nan_to_num(tc, nan=0.0)
            # If T is very short (<30), skip
            if tc.shape[0] < 30:
                skipped += 1
                continue
            timeseries.append(tc)
            labels.append(int(row['label']))
            sites.append(int(row['Site']))
            ages.append(float(row['Age']) if not pd.isna(row['Age']) else np.nan)
            subject_ids.append(sid)
        except Exception as e:
            skipped += 1
            continue
    print(f"[cc200] loaded {len(timeseries)} subjects (skipped {skipped}), HC={sum(1 for l in labels if l==0)}, ADHD={sum(1 for l in labels if l==1)}")
    # Site counts
    uniq, cnts = np.unique(sites, return_counts=True)
    print(f"[cc200] site distribution: {dict(zip(uniq, cnts))}")
    return timeseries, np.array(labels, dtype=int), {"sites": np.array(sites), "ages": np.array(ages), "subject_ids": subject_ids, "n_rois": timeseries[0].shape[1] if timeseries else 0}

if __name__ == "__main__":
    ts, labs, meta = load_cc200()
    print("example tc shape", ts[0].shape if ts else "none")
    print("labels", np.unique(labs, return_counts=True))
