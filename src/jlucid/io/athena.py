"""Readers for the Athena AAL time-course archive.

File layout (verified 2026-08-09):  per subject directory there are two
variants - ``sfnwmrda*`` (Athena-filtered: 0.009-0.08 Hz bandpass after 6-param
motion + WM/CSF regression and 6mm smoothing) and ``snwmrda*`` (motion-
corrected but unfiltered).  We use only the ``sfnwmrda`` variant for the
primary analysis (D1/D3).

The .1D files have 116 ROI columns whose header labels match the sorted unique
values of the bundled ``templates/aal_mask_pad.nii.gz`` (AFNI-style codes
2001..9170).  The first 90 columns are the AAL-90 cerebral ROIs in standard
Tzourio-Mazoyer order (verified against the SPM12 AAL atlas labels); columns
9001..9170 are cerebellum + vermis and are dropped (D11).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from ..config import (
    ATHENA_MAIN_DIR,
    EXPECTED_TC_COLS,
    N_ROI,
    REST_SESSION,
    SRC_DATA_DIR,
    TC_PREFIX,
)

TC_RE = re.compile(rf"^{TC_PREFIX}(\d+)_session_1_{REST_SESSION}_aal_TCs\.1D$")


def iter_tc_files(archive_dir: Path = ATHENA_MAIN_DIR) -> Iterator[tuple[str, str, Path]]:
    """Yield (site, raw_id, path) for every usable filtered TC file."""
    for site_dir in sorted(
        p for p in archive_dir.iterdir() if p.is_dir() and p.name != "templates"
    ):
        site = site_dir.name
        for tc_file in sorted(site_dir.rglob("*_aal_TCs.1D")):
            m = TC_RE.match(tc_file.name)
            if m:
                yield site, m.group(1), tc_file


def list_archive_subjects(archive_dir: Path = ATHENA_MAIN_DIR) -> pd.DataFrame:
    """Table of (site, raw_id, tc_path) for filtered rest_1 files (dedup)."""
    rows = []
    seen: set[tuple[str, str]] = set()
    for site, raw_id, path in iter_tc_files(archive_dir):
        if (site, raw_id) in seen:
            continue
        seen.add((site, raw_id))
        rows.append({"site": site, "raw_id": raw_id, "tc_path": str(path)})
    return pd.DataFrame(rows, columns=["site", "raw_id", "tc_path"])


def parse_tc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an Athena .1D time-course file.

    Returns (data (T, 116) float64, mask codes (116,)).  The first two columns
    are the AFNI File / Sub-brick identifiers and are skipped.
    """
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
    codes = np.array([int(h.replace("Mean_", "").strip()) for h in header if h.startswith("Mean_")])
    if len(codes) != EXPECTED_TC_COLS:
        raise ValueError(f"{path}: expected {EXPECTED_TC_COLS} ROI columns, found {len(codes)}")
    rows = []
    with open(path) as fh:
        next(fh)
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            rows.append([float(v) for v in parts[2 : 2 + EXPECTED_TC_COLS]])
    data = np.asarray(rows, dtype=np.float64)
    if data.shape[1] != EXPECTED_TC_COLS:
        raise ValueError(f"{path}: data has {data.shape[1]} columns, expected {EXPECTED_TC_COLS}")
    return data, codes


def cerebral_tc(data: np.ndarray, n_roi: int = N_ROI) -> np.ndarray:
    """Keep the AAL-90 cerebral columns (first 90 in mask-code order, D11)."""
    if data.shape[1] < n_roi:
        raise ValueError(f"expected >= {n_roi} columns, got {data.shape[1]}")
    return data[:, :n_roi]


def load_aal_labels() -> pd.DataFrame:
    return pd.read_csv(SRC_DATA_DIR / "aal90_labels.csv")


def load_network_membership() -> pd.DataFrame:
    return pd.read_csv(SRC_DATA_DIR / "aal_network_membership.csv")


def load_athena_phenotypic(path: Path) -> pd.DataFrame:
    """Read a per-site Athena phenotypic CSV (column variants handled)."""
    df = pd.read_csv(path)
    if "ScanDir ID" not in df.columns and "ScanDirID" in df.columns:
        df = df.rename(columns={"ScanDirID": "ScanDir ID"})
    if "QC_Athena" not in df.columns and "QC_Rest_1" in df.columns:
        df["QC_Athena"] = df["QC_Rest_1"]
    return df
