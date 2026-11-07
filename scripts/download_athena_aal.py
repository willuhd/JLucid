#!/usr/bin/env python3
"""Athena ADHD200 preprocessed (AAL) archive — manual NITRC step documented.

Free NITRC registration required (not NeuroBureau, not a DUA; HBN DUA is separate and not needed).
1) Register at https://www.nitrc.org (free), find fcon_1000 / ADHD200 Athena preprocessed release.
2) Download filtfix training-sample archive, extract to data/athena/main/ (gitignored).
3) Validate *_aal_TCs.1D files (116 cols, sfnwmrda*_session_1_rest_1_aal_TCs.1D).
This script performs step 3 validation only; it cannot automate NITRC login.
"""
import glob, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ARCH = os.path.join(ROOT, 'data', 'athena', 'main')
def main():
    fs = sorted(glob.glob(os.path.join(ARCH, '**', '*_aal_TCs.1D'), recursive=True))
    print(f'Athena archive: {ARCH}')
    print(f'AAL TC files found: {len(fs)}')
    if not fs:
        print('Missing. Download Athena filtfix archive from NITRC (free account),')
        print('extract to data/athena/main/, rerun. See docs/data-download.md.')
        sys.exit(1)
    for f in fs[:3]:
        with open(f) as fh:
            hdr = fh.readline().rstrip().split('\t')
        print(f, 'cols:', len(hdr))
    print('OK')
if __name__ == '__main__': main()
