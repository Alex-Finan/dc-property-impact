"""Load and verify all downloaded datasets.

Run this script to confirm data integrity and print a summary.
Usage: python -m src.data.load_all
"""

import csv
import json
import os
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW = DATA_DIR / "raw"
EXT = DATA_DIR / "external"


def check_geojson(path: Path) -> dict:
    """Load a GeoJSON file and return feature count and field names."""
    with open(path) as f:
        data = json.load(f)
    features = data.get("features", [])
    fields = list(features[0]["properties"].keys()) if features else []
    has_geometry = bool(features and features[0].get("geometry"))
    return {
        "features": len(features),
        "fields": fields,
        "has_geometry": has_geometry,
    }


def check_csv(path: Path, filter_prefix: str | None = None, col_idx: int = 0) -> dict:
    """Load a CSV and return row count, optionally filtering by prefix."""
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    total = len(rows)
    filtered = total
    if filter_prefix:
        filtered_rows = [r for r in rows if r[col_idx].zfill(11).startswith(filter_prefix)]
        filtered = len(filtered_rows)

    return {"headers": headers, "total_rows": total, "filtered_rows": filtered}


def check_json_array(path: Path) -> dict:
    """Load a Census API JSON response."""
    with open(path) as f:
        data = json.load(f)
    return {"headers": data[0], "rows": len(data) - 1}


def main():
    print("=" * 60)
    print("DC PROPERTY IMPACT — DATA INVENTORY")
    print("=" * 60)

    # --- FHFA Tract-Level HPI ---
    print("\n[1] FHFA Tract-Level House Price Index")
    info = check_csv(RAW / "fhfa_tract_hpi.csv", filter_prefix="51107")
    print(f"    Total rows: {info['total_rows']:,}")
    print(f"    Loudoun County rows: {info['filtered_rows']:,}")
    print(f"    Columns: {info['headers']}")

    # --- FHFA County-Level HPI ---
    print("\n[2] FHFA County-Level House Price Index")
    info = check_csv(RAW / "fhfa_county_hpi.csv")
    print(f"    Total rows: {info['total_rows']:,}")

    # --- Census ACS ---
    print("\n[3] Census ACS 5-Year Estimates (Loudoun County Tracts)")
    for year in [2018, 2019, 2020, 2021, 2022]:
        path = RAW / f"acs_loudoun_tracts_{year}.json"
        if path.exists():
            info = check_json_array(path)
            print(f"    {year}: {info['rows']} tracts")

    # --- ByRight DC Parcels ---
    print("\n[4] ByRight Data Center Parcels (Loudoun County)")
    info = check_geojson(RAW / "byright_dc_parcels_full.geojson")
    print(f"    Features: {info['features']}")
    print(f"    Has geometry: {info['has_geometry']}")
    print(f"    Fields: {info['fields'][:10]}...")

    # --- Loudoun Parcels ---
    print("\n[5] Loudoun County Parcels (20k sample)")
    info = check_geojson(RAW / "loudoun_parcels_20k.geojson")
    print(f"    Features: {info['features']}")
    print(f"    Has geometry: {info['has_geometry']}")

    # --- Loudoun Zoning ---
    print("\n[6] Loudoun County Zoning")
    info = check_geojson(RAW / "loudoun_zoning_sample.geojson")
    print(f"    Features: {info['features']}")
    print(f"    Has geometry: {info['has_geometry']}")

    # --- Loudoun Boundary ---
    print("\n[7] Loudoun County Boundary")
    info = check_geojson(RAW / "loudoun_boundary.geojson")
    print(f"    Features: {info['features']}")

    # --- PW County DC Buildings ---
    print("\n[8] Prince William County DC Buildings")
    info = check_geojson(RAW / "pwc_dc_buildings.geojson")
    print(f"    Features: {info['features']}")
    print(f"    Fields: {info['fields'][:10]}...")

    # --- TIGER Tracts ---
    print("\n[9] TIGER/Line Census Tract Boundaries (Virginia)")
    tiger_path = EXT / "tiger_tracts_51_2020.zip"
    if tiger_path.exists():
        with zipfile.ZipFile(tiger_path) as z:
            shp_files = [f for f in z.namelist() if f.endswith(".shp")]
            print(f"    Shapefile: {shp_files[0] if shp_files else 'NOT FOUND'}")
            print(f"    ZIP size: {os.path.getsize(tiger_path) / 1e6:.1f} MB")

    # --- Summary ---
    print("\n" + "=" * 60)
    total_size = sum(
        os.path.getsize(f) for f in RAW.glob("*") if f.is_file()
    ) + sum(
        os.path.getsize(f) for f in EXT.glob("*") if f.is_file()
    )
    print(f"Total data size: {total_size / 1e6:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
