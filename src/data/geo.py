"""GIS and spatial data processing.

Handles spatial joins, distance calculations, and treatment group assignment.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


def assign_treatment_rings(
    properties: gpd.GeoDataFrame,
    dc_locations: gpd.GeoDataFrame,
    rings: dict[str, list[float]] | None = None,
) -> gpd.GeoDataFrame:
    """Assign each property to a treatment ring based on distance to nearest DC.

    Args:
        properties: GeoDataFrame of property locations (must have geometry in EPSG:4326).
        dc_locations: GeoDataFrame of data center locations.
        rings: Dict mapping ring names to [min_miles, max_miles].
               Defaults to standard rings from the config.

    Returns:
        Properties GeoDataFrame with added columns:
        - dist_to_nearest_dc_mi: Distance in miles to nearest data center
        - nearest_dc_id: ID of nearest data center
        - treatment_ring: Name of the assigned ring (or 'outside')
    """
    if rings is None:
        rings = {
            "inner": [0, 1.0],
            "middle": [1.0, 2.0],
            "outer": [2.0, 3.0],
            "buffer": [3.0, 4.0],
            "control": [4.0, 8.0],
        }

    # Project to a meter-based CRS for distance calculation (UTM zone 18N for VA)
    props_proj = properties.to_crs(epsg=32618)
    dc_proj = dc_locations.to_crs(epsg=32618)

    # Calculate distance to nearest DC for each property
    distances = []
    nearest_ids = []

    dc_points = dc_proj.geometry.values
    dc_indices = dc_proj.index.values

    for _, prop in props_proj.iterrows():
        dists = np.array([prop.geometry.distance(dc_pt) for dc_pt in dc_points])
        min_idx = dists.argmin()
        distances.append(dists[min_idx])
        nearest_ids.append(dc_indices[min_idx])

    result = properties.copy()
    result["dist_to_nearest_dc_m"] = distances
    result["dist_to_nearest_dc_mi"] = np.array(distances) / 1609.34  # meters to miles
    result["nearest_dc_id"] = nearest_ids

    # Assign to rings
    def classify_ring(dist_mi: float) -> str:
        for ring_name, (lo, hi) in rings.items():
            if lo <= dist_mi < hi:
                return ring_name
        return "outside"

    result["treatment_ring"] = result["dist_to_nearest_dc_mi"].apply(classify_ring)

    return result


def create_distance_matrix(
    properties: gpd.GeoDataFrame,
    dc_locations: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Create a full distance matrix between all properties and all DCs.

    Args:
        properties: GeoDataFrame of properties.
        dc_locations: GeoDataFrame of data center locations.

    Returns:
        DataFrame with shape (n_properties, n_dcs), values in miles.
    """
    props_proj = properties.to_crs(epsg=32618)
    dc_proj = dc_locations.to_crs(epsg=32618)

    matrix = np.zeros((len(props_proj), len(dc_proj)))
    for j, (_, dc) in enumerate(dc_proj.iterrows()):
        matrix[:, j] = props_proj.geometry.distance(dc.geometry).values / 1609.34

    return pd.DataFrame(
        matrix,
        index=properties.index,
        columns=dc_locations.index,
    )


def geocode_addresses(df: pd.DataFrame, address_col: str) -> gpd.GeoDataFrame:
    """Convert a DataFrame with addresses to a GeoDataFrame using Census geocoder.

    Uses the free Census Bureau batch geocoding service.
    For large datasets, consider the batch endpoint or a local geocoder.

    Args:
        df: DataFrame with an address column.
        address_col: Name of the column containing full addresses.

    Returns:
        GeoDataFrame with Point geometries for successfully geocoded addresses.
    """
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    geolocator = Nominatim(user_agent="dc-property-impact")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)

    geometries = []
    for addr in df[address_col]:
        try:
            location = geocode(addr)
            if location:
                geometries.append(Point(location.longitude, location.latitude))
            else:
                geometries.append(None)
        except Exception:
            geometries.append(None)

    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    return gdf
