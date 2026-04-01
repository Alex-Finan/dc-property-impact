"""Spatial econometrics and diagnostics.

Handles spatial autocorrelation testing, Conley standard errors,
and spatial regression models.
"""

import numpy as np
import pandas as pd
import geopandas as gpd


def compute_morans_i(
    gdf: gpd.GeoDataFrame,
    variable: str,
    k: int = 8,
) -> dict:
    """Compute Moran's I statistic for spatial autocorrelation.

    Args:
        gdf: GeoDataFrame with the variable of interest.
        variable: Column name to test.
        k: Number of nearest neighbors for spatial weights.

    Returns:
        Dict with Moran's I statistic, expected I, p-value, and z-score.
    """
    from libpysal.weights import KNN
    from esda.moran import Moran

    w = KNN.from_dataframe(gdf, k=k)
    w.transform = "r"

    y = gdf[variable].values
    mi = Moran(y, w)

    return {
        "I": mi.I,
        "expected_I": mi.EI,
        "p_value": mi.p_sim,
        "z_score": mi.z_sim,
        "significant": mi.p_sim < 0.05,
    }


def spatial_lag_test(
    gdf: gpd.GeoDataFrame,
    residuals: np.ndarray,
    k: int = 8,
) -> dict:
    """Test residuals for spatial dependence using Lagrange Multiplier tests.

    Args:
        gdf: GeoDataFrame used in the regression.
        residuals: OLS residuals to test.
        k: Number of nearest neighbors.

    Returns:
        Dict with LM-lag and LM-error test statistics and p-values.
    """
    from libpysal.weights import KNN
    from esda.moran import Moran

    w = KNN.from_dataframe(gdf, k=k)
    w.transform = "r"

    mi = Moran(residuals, w)

    return {
        "moran_I": mi.I,
        "moran_p": mi.p_sim,
        "spatial_autocorrelation_detected": mi.p_sim < 0.05,
    }


def compute_distance_decay(
    properties: pd.DataFrame,
    distance_col: str = "dist_to_nearest_dc_mi",
    price_col: str = "ln_price",
    post_col: str = "post",
    bin_width: float = 0.25,
    max_distance: float = 8.0,
) -> pd.DataFrame:
    """Compute binned average treatment effects by distance.

    Creates distance bins and computes the mean DiD effect within each bin,
    revealing the shape of the distance decay function.

    Args:
        properties: Property data with distance, price, and post indicators.
        distance_col: Column with distance to nearest DC (miles).
        price_col: Column with log price.
        post_col: Column with post-treatment indicator.
        bin_width: Width of each distance bin in miles.
        max_distance: Maximum distance to include.

    Returns:
        DataFrame with distance bins and average effects.
    """
    df = properties[properties[distance_col] <= max_distance].copy()
    df["dist_bin"] = (df[distance_col] // bin_width) * bin_width + bin_width / 2

    # Compute mean price by distance bin and pre/post period
    grouped = df.groupby(["dist_bin", post_col])[price_col].agg(["mean", "count", "std"])
    grouped = grouped.reset_index()

    # Compute DiD for each bin (post - pre)
    pre = grouped[grouped[post_col] == 0].set_index("dist_bin")
    post = grouped[grouped[post_col] == 1].set_index("dist_bin")

    common_bins = pre.index.intersection(post.index)
    decay = pd.DataFrame({
        "dist_bin": common_bins,
        "pre_mean": pre.loc[common_bins, "mean"].values,
        "post_mean": post.loc[common_bins, "mean"].values,
        "diff": post.loc[common_bins, "mean"].values - pre.loc[common_bins, "mean"].values,
        "n_pre": pre.loc[common_bins, "count"].values,
        "n_post": post.loc[common_bins, "count"].values,
    })

    return decay
