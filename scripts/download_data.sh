#!/bin/bash
# Download all publicly available datasets for the DC Property Impact analysis.
# Run from the project root: bash scripts/download_data.sh
#
# Requires: curl, python3
# Optional: Census API key (set CENSUS_API_KEY env var for higher rate limits)

set -euo pipefail
cd "$(dirname "$0")/.."

RAW="data/raw"
EXT="data/external"
mkdir -p "$RAW" "$EXT"

echo "=== DC Property Impact — Data Download ==="
echo ""

# 1. FHFA House Price Index (tract-level)
echo "[1/8] FHFA Tract-Level HPI..."
curl -sL -o "$RAW/fhfa_tract_hpi.csv" \
  "https://www.fhfa.gov/hpi/download/annual/hpi_at_tract.csv"
echo "  Downloaded: $(du -h "$RAW/fhfa_tract_hpi.csv" | cut -f1)"

# 2. FHFA County-Level HPI
echo "[2/8] FHFA County-Level HPI..."
curl -sL -o "$RAW/fhfa_county_hpi.csv" \
  "https://www.fhfa.gov/hpi/download/annual/hpi_at_county.csv"
echo "  Downloaded: $(du -h "$RAW/fhfa_county_hpi.csv" | cut -f1)"

# 3. TIGER/Line Census Tract Boundaries (Virginia)
echo "[3/8] TIGER/Line Census Tracts (Virginia)..."
curl -sL -o "$EXT/tiger_tracts_51_2020.zip" \
  "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_51_tract.zip"
echo "  Downloaded: $(du -h "$EXT/tiger_tracts_51_2020.zip" | cut -f1)"

# 4. Census ACS 5-Year Estimates (Loudoun County tracts, 2018-2022)
echo "[4/8] Census ACS 5-Year Estimates (2018-2022)..."
ACS_VARS="NAME,B19013_001E,B01003_001E,B25077_001E,B25064_001E,B25003_001E,B25003_002E,B15003_022E,B15003_023E,B15003_025E,B23025_005E,B23025_002E,B25035_001E,B25018_001E"
KEY_PARAM=""
if [ -n "${CENSUS_API_KEY:-}" ]; then
  KEY_PARAM="&key=$CENSUS_API_KEY"
fi
for YEAR in 2018 2019 2020 2021 2022; do
  curl -s -o "$RAW/acs_loudoun_tracts_${YEAR}.json" \
    "https://api.census.gov/data/${YEAR}/acs/acs5?get=${ACS_VARS}&for=tract:*&in=state:51%20county:107${KEY_PARAM}"
done
echo "  Downloaded 5 years of ACS data"

# 5. ByRight Data Center Parcels (Loudoun County, via ArcGIS)
echo "[5/8] ByRight Data Center Parcels..."
python3 -c "
import json, urllib.request
all_features, offset, batch = [], 0, 1000
while True:
    url = f'https://services1.arcgis.com/MxjRokvPm7bjslyR/arcgis/rest/services/ByRight_Data_Center_Parcels/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson&resultRecordCount={batch}&resultOffset={offset}'
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)
    feats = data.get('features', [])
    if not feats: break
    all_features.extend(feats)
    offset += batch
    if len(feats) < batch: break
with open('$RAW/byright_dc_parcels_full.geojson', 'w') as f:
    json.dump({'type': 'FeatureCollection', 'features': all_features}, f)
print(f'  Downloaded {len(all_features)} DC parcels')
"

# 6. Prince William County DC Buildings
echo "[6/8] Prince William County DC Buildings..."
curl -s -o "$RAW/pwc_dc_buildings.geojson" \
  "https://gisweb.pwcva.gov/arcgis/rest/services/Planning/Build_Out_Analysis/MapServer/9/query?where=1%3D1&outFields=*&f=geojson&resultRecordCount=5000"
echo "  Downloaded: $(du -h "$RAW/pwc_dc_buildings.geojson" | cut -f1)"

# 7. Loudoun County Boundary & Zoning
echo "[7/8] Loudoun County Boundary & Zoning..."
curl -s -o "$RAW/loudoun_boundary.geojson" \
  "https://logis.loudoun.gov/gis/rest/services/COL/CountyBoundary/MapServer/0/query?where=1%3D1&outFields=*&f=geojson"
curl -s -o "$RAW/loudoun_zoning_sample.geojson" \
  "https://logis.loudoun.gov/gis/rest/services/COL/Zoning/MapServer/3/query?where=1%3D1&outFields=*&f=geojson&resultRecordCount=5000"
echo "  Downloaded boundary and zoning"

# 8. Loudoun County Parcels (sample)
echo "[8/8] Loudoun County Parcels (first 21k)..."
python3 -c "
import json, urllib.request
all_features, offset, batch, max_records = [], 0, 3000, 21000
while offset < max_records:
    url = f'https://logis.loudoun.gov/gis/rest/services/COL/LandRecords/MapServer/5/query?where=1%3D1&outFields=*&f=geojson&resultRecordCount={batch}&resultOffset={offset}'
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    feats = data.get('features', [])
    if not feats: break
    all_features.extend(feats)
    offset += batch
    if len(feats) < batch: break
with open('$RAW/loudoun_parcels_20k.geojson', 'w') as f:
    json.dump({'type': 'FeatureCollection', 'features': all_features}, f)
print(f'  Downloaded {len(all_features)} parcels')
"

# 9. FHFA ZIP5-level HPI (Excel format)
echo "[9/14] FHFA ZIP5 HPI..."
curl -sL -o "$RAW/fhfa_zip5_hpi.csv" \
  "https://www.fhfa.gov/hpi/download/annual/hpi_at_zip5.csv"
echo "  Downloaded: $(du -h "$RAW/fhfa_zip5_hpi.csv" | cut -f1)"

# 10. FHFA ZIP3-level HPI
echo "[10/14] FHFA ZIP3 HPI..."
curl -sL -o "$RAW/fhfa_zip3_hpi.csv" \
  "https://www.fhfa.gov/hpi/download/annual/hpi_at_zip3.csv"
echo "  Downloaded: $(du -h "$RAW/fhfa_zip3_hpi.csv" | cut -f1)"

# 11. NVRC Data Centers (265 locations across NoVA with addresses & jurisdictions)
echo "[11/14] NVRC Regional Data Centers..."
curl -s -o "$RAW/nvrc_data_centers.geojson" \
  "https://services5.arcgis.com/6MUPhDX27Ne3DNOw/arcgis/rest/services/Data_Centers/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson&resultRecordCount=5000"
echo "  Downloaded: $(du -h "$RAW/nvrc_data_centers.geojson" | cut -f1)"

# 12. Full Loudoun County Parcels (132k)
echo "[12/14] Loudoun County Parcels (full 132k)..."
python3 -c "
import json, urllib.request, time
all_features, offset, batch = [], 0, 3000
while True:
    url = f'https://logis.loudoun.gov/gis/rest/services/COL/LandRecords/MapServer/5/query?where=1%3D1&outFields=*&f=geojson&resultRecordCount={batch}&resultOffset={offset}'
    try:
        with urllib.request.urlopen(url, timeout=90) as resp:
            data = json.load(resp)
    except Exception as e:
        print(f'  Retry at {offset}: {e}')
        time.sleep(2)
        continue
    feats = data.get('features', [])
    if not feats: break
    all_features.extend(feats)
    offset += batch
    if offset % 30000 == 0: print(f'  {len(all_features):,} parcels...')
    if len(feats) < batch: break
with open('$RAW/loudoun_parcels_full.geojson', 'w') as f:
    json.dump({'type': 'FeatureCollection', 'features': all_features}, f)
print(f'  Downloaded {len(all_features):,} parcels')
"

# 13. ACS for neighboring counties (Fairfax 059, Prince William 153)
echo "[13/14] ACS for Fairfax & Prince William counties..."
for county_fips in 059 153; do
  curl -s -o "$RAW/acs_51${county_fips}_tracts_2022.json" \
    "https://api.census.gov/data/2022/acs/acs5?get=${ACS_VARS}&for=tract:*&in=state:51%20county:${county_fips}${KEY_PARAM}"
done
echo "  Downloaded ACS for Fairfax and Prince William"

# 14. Realtor.com ZIP-level inventory (large file ~800MB)
echo "[14/14] Realtor.com ZIP inventory..."
curl -sL -o "$RAW/realtor_zip_inventory.csv" \
  "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_Zip_History.csv"
echo "  Downloaded: $(du -h "$RAW/realtor_zip_inventory.csv" | cut -f1)"

echo ""
echo "=== Download complete ==="
echo "Run 'python -m src.data.load_all' to verify all datasets."
