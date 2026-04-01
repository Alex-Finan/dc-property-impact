"""Confounder data acquisition: schools, crime, transit.

These variables are critical controls in the hedonic DiD model to avoid
omitted variable bias, particularly from the Silver Line Phase 2 opening.
"""

from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Virginia School Quality Profiles
# ---------------------------------------------------------------------------

VA_SCHOOL_QUALITY_URL = "https://schoolquality.virginia.gov/download-data"


def load_school_quality_data(filepath: Path) -> pd.DataFrame:
    """Load Virginia School Quality Profile data.

    Download from: https://schoolquality.virginia.gov/download-data
    Select: Loudoun County Public Schools, all years available.

    Args:
        filepath: Path to downloaded CSV/Excel file.

    Returns:
        DataFrame with school-level quality metrics.
    """
    if filepath.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


# ---------------------------------------------------------------------------
# Metro Silver Line Stations (critical confounder)
# ---------------------------------------------------------------------------

SILVER_LINE_PHASE2_STATIONS = {
    "Reston Town Center": {"lat": 38.9535, "lon": -77.3385, "opened": "2022-11-15"},
    "Herndon": {"lat": 38.9535, "lon": -77.3709, "opened": "2022-11-15"},
    "Innovation Center": {"lat": 38.9591, "lon": -77.4167, "opened": "2022-11-15"},
    "Dulles Airport": {"lat": 38.9559, "lon": -77.4462, "opened": "2022-11-15"},
    "Loudoun Gateway": {"lat": 38.9965, "lon": -77.4622, "opened": "2022-11-15"},
    "Ashburn": {"lat": 39.0053, "lon": -77.4912, "opened": "2022-11-15"},
}


def get_silver_line_stations() -> pd.DataFrame:
    """Return Silver Line Phase 2 station locations and opening dates.

    The two Loudoun County stations (Loudoun Gateway, Ashburn) are
    critical confounders for the DC property impact analysis.
    """
    records = []
    for name, info in SILVER_LINE_PHASE2_STATIONS.items():
        records.append({
            "station_name": name,
            "latitude": info["lat"],
            "longitude": info["lon"],
            "date_opened": pd.Timestamp(info["opened"]),
            "in_loudoun": name in ("Loudoun Gateway", "Ashburn"),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# FBI Crime Data (Uniform Crime Reporting)
# ---------------------------------------------------------------------------

def load_crime_data(filepath: Path) -> pd.DataFrame:
    """Load crime data for Loudoun County.

    Download from FBI Crime Data Explorer:
        https://cde.ucr.cjis.gov/
    Search for: Loudoun County Sheriff's Office

    Or from Virginia State Police annual reports:
        https://vsp.virginia.gov/sections-units-bureaus/bass/
        criminal-justice-information-services/uniform-crime-reporting/
    """
    if filepath.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


# ---------------------------------------------------------------------------
# Loudoun County Zoning Changes
# ---------------------------------------------------------------------------

ZONING_TIMELINE = [
    {
        "date": "2022-04-01",
        "event": "Board Data Center Discussion Series begins",
        "impact": "Signals potential regulatory changes",
    },
    {
        "date": "2024-07-01",
        "event": "Board approves two-phase regulatory reform",
        "impact": "Market anticipation of stricter zoning",
    },
    {
        "date": "2025-03-18",
        "event": "Phase 1 approved: Special Exception required for all new DCs",
        "impact": "Eliminates by-right DC development",
    },
    {
        "date": "2026-12-01",
        "event": "Phase 2 expected: New noise/setback/design standards",
        "impact": "Additional operational requirements",
    },
]


def get_zoning_timeline() -> pd.DataFrame:
    """Return key data center zoning regulatory events for Loudoun County."""
    df = pd.DataFrame(ZONING_TIMELINE)
    df["date"] = pd.to_datetime(df["date"])
    return df
