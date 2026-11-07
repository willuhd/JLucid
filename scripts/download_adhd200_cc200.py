#!/usr/bin/env python3
"""Download ADHD200 raw rest BOLD from public FCP-INDI S3 mirror (no account).
Source: docs/EMPIRICAL_REPORT_20261018.md sec 3.2 (1.8 s/subject, ~57 MB/s).
Writes only into gitignored data/ (override DATA_ROOT). Then run scripts/parcellate.py.
"""
import os, sys
DATA_ROOT = os.environ.get('DATA_ROOT', 'data/adhd200')
MIRROR = 'https://fcp-indi.s3.amazonaws.com/data/Projects/ADHD200/RawData'
def main():
    print(f'Data root: {DATA_ROOT}')
    print(f'Mirror: {MIRROR}')
    print('Per-subject URL pattern:')
    print('  {MIRROR}/<site>/<sid>/session_1/rest_1/rest.nii.gz')
    print('This script intentionally does not auto-download 100+ GB; fetch per subject,')
    print('parcellate with scripts/parcellate.py, delete the .nii.gz after extraction.')
    print('Site/sid list: data/adhd200/adhd200_preprocessed_phenotypics.tsv (ScanDir ID, Site).')
if __name__ == '__main__': main()
