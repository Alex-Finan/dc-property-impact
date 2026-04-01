"""Census and demographic data acquisition.

Data sources:
- American Community Survey (ACS) 5-Year estimates via Census API
- TIGER/Line shapefiles for geographic boundaries
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from src.utils.config import EXTERNAL_DIR


# ---------------------------------------------------------------------------
# Census API (requires free API key from https://api.census.gov/data/key_signup.html)
# ---------------------------------------------------------------------------

CENSUS_API_BASE = "https://api.census.gov/data"

# Key ACS variables for hedonic model controls
ACS_VARIABLES = {
    "B19013_001E": "median_household_income",
    "B01003_001E": "total_population",
    "B25077_001E": "median_home_value",
    "B25064_001E": "median_gross_rent",
    "B25003_001E": "total_housing_units",
    "B25003_002E": "owner_occupied_units",
    "B15003_022E": "bachelors_degree",
    "B15003_023E": "masters_degree",
    "B15003_025E": "doctorate_degree",
    "B23025_005E": "unemployed",
    "B23025_002E": "labor_force",
    "B25035_001E": "median_year_built",
    "B25018_001E": "median_rooms",
}


def fetch_acs_tract_data(
    year: int,
    state_fips: str = "51",
    county_fips: str = "107",
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch ACS 5-Year estimates at the census tract level.

    Args:
        year: ACS year (e.g., 2022 for 2018-2022 5-year estimates).
        state_fips: 2-digit state FIPS code.
        county_fips: 3-digit county FIPS code.
        api_key: Census API key. Get one free at
                 https://api.census.gov/data/key_signup.html

    Returns:
        DataFrame with one row per tract and columns for each ACS variable.
    """
    variables = ",".join(ACS_VARIABLES.keys())
    url = (
        f"{CENSUS_API_BASE}/{year}/acs/acs5"
        f"?get=NAME,{variables}"
        f"&for=tract:*"
        f"&in=state:{state_fips}%20county:{county_fips}"
    )
    if api_key:
        url += f"&key={api_key}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns=ACS_VARIABLES)

    # Build full tract FIPS
    df["tract_fips"] = df["state"] + df["county"] + df["tract"]

    # Convert numeric columns
    for col in ACS_VARIABLES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived metrics
    if "owner_occupied_units" in df.columns and "total_housing_units" in df.columns:
        df["pct_owner_occupied"] = df["owner_occupied_units"] / df["total_housing_units"]

    if "unemployed" in df.columns and "labor_force" in df.columns:
        df["unemployment_rate"] = df["unemployed"] / df["labor_force"]

    edu_cols = ["bachelors_degree", "masters_degree", "doctorate_degree"]
    if all(c in df.columns for c in edu_cols):
        df["college_educated"] = df[edu_cols].sum(axis=1)
        df["pct_college"] = df["college_educated"] / df["total_population"]

    return df


# ---------------------------------------------------------------------------
# TIGER/Line Census Tract Boundaries
# ---------------------------------------------------------------------------

TIGER_TRACT_URL_TEMPLATE = (
    "https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/"
    "tl_{year}_{state_fips}_tract.zip"
)


def fetch_tiger_tract_boundaries(
    state_fips: str = "51",
    year: int = 2020,
) -> gpd.GeoDataFrame:
    """Fetch TIGER/Line census tract boundary shapefiles.

    Downloads the shapefile ZIP and reads it into a GeoDataFrame.

    Args:
        state_fips: 2-digit state FIPS.
        year: TIGER vintage year.

    Returns:
        GeoDataFrame with tract geometries in EPSG:4326.
    """
    url = TIGER_TRACT_URL_TEMPLATE.format(year=year, state_fips=state_fips)
    local_path = EXTERNAL_DIR / f"tiger_tracts_{state_fips}_{year}.zip"

    if not local_path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        local_path.write_bytes(response.content)

    gdf = gpd.read_file(f"zip://{local_path}")
    gdf = gdf.to_crs(epsg=4326)
    return gdf


def filter_tracts_to_county(
    gdf: gpd.GeoDataFrame,
    county_fips: str = "107",
) -> gpd.GeoDataFrame:
    """Filter tract boundaries to a specific county."""
    return gdf[gdf["COUNTYFP"] == county_fips].copy()
