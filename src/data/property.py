"""Property transaction data acquisition and processing.

Data sources:
- Loudoun County Real Property Sales Reports (2013-present, free Excel)
- FHFA House Price Index at census tract level (free CSV)
- Zillow ZTRAX via ICPSR (academic access, requires DUA)
"""

from pathlib import Path

import pandas as pd
import requests

from src.utils.config import RAW_DIR


# ---------------------------------------------------------------------------
# Loudoun County Real Property Sales Reports
# ---------------------------------------------------------------------------

LOUDOUN_SALES_BASE_URL = "https://www.loudoun.gov/649/Public-Real-Estate-Reports"


def load_loudoun_sales(filepath: Path) -> pd.DataFrame:
    """Load a Loudoun County Real Property Sales Report (Excel).

    The user must manually download the Excel files from:
        https://www.loudoun.gov/649/Public-Real-Estate-Reports

    Args:
        filepath: Path to the downloaded .xlsx file.

    Returns:
        Cleaned DataFrame with standardized column names.
    """
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    date_cols = [c for c in df.columns if "date" in c or "sale" in c]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def combine_loudoun_sales(directory: Path) -> pd.DataFrame:
    """Combine multiple years of Loudoun sales reports from a directory.

    Args:
        directory: Path to directory containing .xlsx sales report files.

    Returns:
        Combined DataFrame across all years.
    """
    dfs = []
    for f in sorted(directory.glob("*.xlsx")):
        df = load_loudoun_sales(f)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No .xlsx files found in {directory}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates()
    return combined


# ---------------------------------------------------------------------------
# FHFA House Price Index (Census Tract Level)
# ---------------------------------------------------------------------------

FHFA_TRACT_URL = (
    "https://www.fhfa.gov/sites/default/files/2024-08/"
    "HPI_AT_BDL_tract.csv"
)


def fetch_fhfa_tract_hpi() -> pd.DataFrame:
    """Fetch the FHFA tract-level house price index.

    Returns:
        DataFrame with tract-level HPI data. Columns include:
        - tract: Census tract FIPS code
        - year: Year of observation
        - annual_change: Annual % change in HPI
        - hpi: House price index value
        - hpi_1990: HPI indexed to 1990 base
        - hpi_2000: HPI indexed to 2000 base
    """
    df = pd.read_csv(FHFA_TRACT_URL)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def filter_fhfa_to_county(df: pd.DataFrame, fips_county: str = "51107") -> pd.DataFrame:
    """Filter FHFA tract-level data to a specific county.

    The tract FIPS code starts with the state+county FIPS (e.g., '51107' for Loudoun).
    """
    tract_col = [c for c in df.columns if "tract" in c or "fips" in c][0]
    df[tract_col] = df[tract_col].astype(str).str.zfill(11)
    mask = df[tract_col].str.startswith(fips_county)
    return df[mask].copy()


# ---------------------------------------------------------------------------
# ZTRAX (Zillow Transaction and Assessment) via ICPSR
# ---------------------------------------------------------------------------

def load_ztrax_transactions(filepath: Path, fips_county: str = "51107") -> pd.DataFrame:
    """Load ZTRAX transaction data for a specific county.

    ZTRAX data must be obtained through ICPSR with a Data Use Agreement:
        https://www.icpsr.umich.edu/web/ICPSR/studies/39652

    Args:
        filepath: Path to the ZTRAX transaction file (CSV or parquet).
        fips_county: 5-digit state+county FIPS code.

    Returns:
        Filtered DataFrame for the specified county.
    """
    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath, low_memory=False)

    fips_col = [c for c in df.columns if "fips" in c.lower() or "county" in c.lower()][0]
    return df[df[fips_col].astype(str).str.startswith(fips_county)].copy()
