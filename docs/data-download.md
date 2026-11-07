# Data download

CC200 (no account): public FCP-INDI S3 mirror, data/Projects/ADHD200/RawData/<site>/<sid>/session_1/rest_1/rest.nii.gz (see scripts/download_adhd200_cc200.py), then scripts/parcellate.py (AFNI 3dROIstats + templates in data/adhd200/templates/).

Athena AAL (free NITRC account, no DUA): Athena ADHD200 preprocessed filtfix archive from NITRC (fcon_1000), extract to data/athena/main/ as *_aal_TCs.1D (see scripts/download_athena_aal.py for validation).

HBN fMRI (Fig.1/gauge, NOT EEG): the gauge test uses HBN resting-state movie fMRI, not EEG. Inputs already on disk in the lab: data/hbn_movie_fmri/<sid>__rest_run-1__roi_stats.csv (180 subjects), data/e123_hbn_labels.npz, data/pilot_out/leida_centroids_190.npy. HBN fMRI movie release is OpenNeuro public (no DUA for fMRI derivatives used here); HBN EEG is a separate DUA-gated modality and is NOT needed for this paper — do not conflate the two.

All downloads write only into gitignored data/.
