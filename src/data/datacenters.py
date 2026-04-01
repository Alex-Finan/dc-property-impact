"""Data center location and timeline data acquisition.

Data sources:
- NVRC (Northern Virginia Regional Commission) Data Center Map
- Loudoun County Building Permits
- Cleanview.co US Data Center Project Tracker
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# NVRC Data Center Map (must request GIS data from NVRC)
# ---------------------------------------------------------------------------

def load_nvrc_datacenter_locations(filepath: Path) -> gpd.GeoDataFrame:
    """Load NVRC data center locations from a GIS file.

    The GIS data must be requested from NVRC (703-642-0700):
        https://www.novaregion.org/1598/Data-Centers

    Supports: .shp, .geojson, .gpkg formats.
    """
    gdf = gpd.read_file(filepath)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


# ---------------------------------------------------------------------------
# Loudoun County Building Permits
# ---------------------------------------------------------------------------

def load_building_permits(filepath: Path) -> pd.DataFrame:
    """Load Loudoun County building permit data (Excel).

    Download from: https://www.loudoun.gov/1164/Issued-Building-Permit-Reports

    Args:
        filepath: Path to the downloaded .xlsx permit file.

    Returns:
        DataFrame with permit data, filtered for data-center-related permits.
    """
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def filter_dc_permits(df: pd.DataFrame) -> pd.DataFrame:
    """Filter building permits to likely data center projects.

    Uses keyword matching on description/type fields. Data centers may appear
    under various names: 'data center', 'technology center', 'server farm',
    or high-amperage electrical permits without explicit DC labeling.
    """
    dc_keywords = [
        "data center", "data centre", "datacenter",
        "server farm", "server facility",
        "technology center", "tech center",
        "colocation", "colo facility",
    ]
    text_cols = [c for c in df.columns if any(k in c for k in ["desc", "type", "name", "use"])]

    if not text_cols:
        return df

    pattern = "|".join(dc_keywords)
    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        mask |= df[col].astype(str).str.lower().str.contains(pattern, na=False)

    return df[mask].copy()


# ---------------------------------------------------------------------------
# Manual DC Timeline Construction
# ---------------------------------------------------------------------------

def create_dc_timeline_template() -> pd.DataFrame:
    """Create a template for manually tracking DC announcement/construction dates.

    This is necessary because no single source provides complete timeline data.
    Users should populate this from news articles, permit records, and NVRC data.
    """
    columns = [
        "dc_id",
        "name",
        "operator",
        "address",
        "latitude",
        "longitude",
        "date_rumored",
        "date_announced",
        "date_permitted",
        "date_construction_start",
        "date_operational",
        "megawatts",
        "square_feet",
        "source_url",
        "notes",
    ]
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Cleanview.co Project Tracker (web scraping helper)
# ---------------------------------------------------------------------------

CLEANVIEW_URL = "https://cleanview.co/public/data-centers/us"


def fetch_cleanview_projects() -> pd.DataFrame:
    """Fetch US data center project data from Cleanview.co.

    Note: This fetches the public project list. Check Cleanview's terms
    of service before automated access. Manual download may be preferred.
    """
    response = requests.get(CLEANVIEW_URL, timeout=30)
    response.raise_for_status()

    # Cleanview serves a JS-rendered page; if table extraction fails,
    # fall back to manual CSV download from the site.
    try:
        tables = pd.read_html(response.text)
        if tables:
            df = tables[0]
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            return df
    except ValueError:
        pass

    raise RuntimeError(
        "Could not parse Cleanview page. Download manually from: "
        f"{CLEANVIEW_URL}"
    )
