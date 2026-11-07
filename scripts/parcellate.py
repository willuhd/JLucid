#!/usr/bin/env python3
"""Parcellate raw rest .nii.gz to ROI time courses (AFNI 3dROIstats).
Adapted from data/adhd200/extract_tcs.sh (templates in data/adhd200/templates/).
Requires AFNI in PATH. Writes <restfile>_<atlas>_TCs.1D next to the input (gitignored data/).
"""
import os, subprocess, sys
TEMPLATES = {
    'aal': 'templates/aal_mask_pad.nii.gz',
    'ez': 'templates/ez_mask_pad.nii.gz',
    'ho': 'templates/ho_mask_pad.nii.gz',
    'tt': 'templates/tt_mask_pad.nii.gz',
    'cc200': 'templates/ADHD200_parcellate_200.nii.gz',
    'cc400': 'templates/ADHD200_parcellate_400.nii.gz',
}
def main():
    if len(sys.argv) < 2:
        print('usage: parcellate.py <restfile.nii.gz> [atlas ...]'); return
    rest = sys.argv[1]; atlases = sys.argv[2:] or list(TEMPLATES)
    base = os.path.dirname(os.path.abspath(rest))
    for a in atlases:
        tpl = os.path.join(base, TEMPLATES[a]) if os.path.exists(os.path.join(base, TEMPLATES[a])) else TEMPLATES[a]
        out = rest[:-len('.nii.gz')] + f'_{a}_TCs.1D'
        print(f'{rest} {tpl} --> {out}')
        subprocess.run(['3dROIstats', '-mask', tpl, rest], stdout=open(out, 'w'), check=True)
if __name__ == '__main__': main()
