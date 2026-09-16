"""Consolidated ADHD-200 phenotype table + per-site motion CSVs."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from ..config import DX_GROUP, SITE_NAME


def normalize_id(value) -> str:
    """Zero-strip a scan ID (archive 0010001 <-> TSV 10001)."""
    return str(int(str(value).strip()))


def load_consolidated_phenotypics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"ScanDir ID": str})
    df["ScanDir ID"] = df["ScanDir ID"].str.strip()
    df["site_name"] = df["Site"].astype(str).map(SITE_NAME)
    df["dx_group"] = df["DX"].astype(str).map(DX_GROUP)
    return df


MOTION_RE = re.compile(r"^rp_(\d+)_session_\d+_rest_(\d+)\.1D$")


def load_motion_csvs(per_site_dir: Path) -> pd.DataFrame:
    """Parse all *_motion.csv files into a tidy table keyed by (site, raw_id).

    Columns kept: max displacement (mm), max rotation (deg), per-axis maxima.
    """
    rows = []
    for f in sorted(per_site_dir.glob("*_motion.csv")):
        site = f.name.replace("_motion.csv", "")
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                fn = (row.get("File") or "").strip()
                m = MOTION_RE.match(fn)
                if not m:
                    continue
                def _f(key):
                    try:
                        return float(row.get(key))
                    except (TypeError, ValueError):
                        return float("nan")
                rows.append(
                    {
                        "site": site,
                        "raw_id": m.group(1),
                        "rest_run": int(m.group(2)),
                        "max_motion_mm": _f("Max Motion (mm)"),
                        "max_rotation_deg": _f("Max Rotation (degree)"),
                        "max_x_mm": _f("Max X (mm)"),
                        "max_y_mm": _f("Max Y (mm)"),
                        "max_z_mm": _f("Max Z (mm)"),
                        "max_roll_deg": _f("Max Roll (degree)"),
                        "max_pitch_deg": _f("Max Pitch (degree)"),
                        "max_yaw_deg": _f("Max Yaw (degree)"),
                    }
                )
    df = pd.DataFrame(rows)
    if len(df):
        df["site"] = df["site"].astype(str)
        df["raw_id"] = df["raw_id"].astype(str)
        df["key"] = df["site"].map({"KKI": "3", "NeuroIMAGE": "4", "NYU": "5",
                                    "OHSU": "6", "Peking_1": "1", "Peking_2": "1",
                                    "Peking_3": "1", "Pittsburgh": "7", "WashU": "8"}
                                   ).astype(str) + "_" + df["raw_id"].apply(normalize_id)
    return df
