"""Geographic visualization for data center impact analysis."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


def plot_treatment_map(
    properties: gpd.GeoDataFrame,
    dc_locations: gpd.GeoDataFrame,
    ring_col: str = "treatment_ring",
    title: str = "Treatment and Control Groups: Loudoun County",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a map showing treatment rings around data centers.

    Args:
        properties: GeoDataFrame with treatment ring assignments.
        dc_locations: GeoDataFrame of data center locations.
        ring_col: Column with treatment ring names.
        title: Plot title.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    ring_colors = {
        "inner": "#d73027",
        "middle": "#fc8d59",
        "outer": "#fee090",
        "buffer": "#e0e0e0",
        "control": "#91bfdb",
        "outside": "#f0f0f0",
    }

    fig, ax = plt.subplots(figsize=(14, 10))

    for ring_name, color in ring_colors.items():
        subset = properties[properties[ring_col] == ring_name]
        if not subset.empty:
            subset.plot(ax=ax, color=color, markersize=2, alpha=0.6, label=ring_name)

    dc_locations.plot(ax=ax, color="black", markersize=30, marker="^",
                      label="Data Centers", zorder=5)

    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs=properties.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    except Exception:
        pass

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_impact_heatmap(
    properties: gpd.GeoDataFrame,
    value_col: str = "price_change_pct",
    title: str = "Property Value Change Heatmap",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a heatmap of property value changes around data centers.

    Args:
        properties: GeoDataFrame with a value change column.
        value_col: Column with percentage change values.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    properties.plot(
        ax=ax,
        column=value_col,
        cmap="RdYlGn",
        legend=True,
        legend_kwds={"label": "Price Change (%)", "shrink": 0.6},
        markersize=3,
        alpha=0.7,
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
