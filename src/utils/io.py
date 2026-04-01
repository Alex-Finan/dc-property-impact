"""Data I/O helpers."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.utils.config import PROCESSED_DIR, RAW_DIR


def save_raw(df: pd.DataFrame | gpd.GeoDataFrame, name: str) -> Path:
    """Save a dataframe to the raw data directory as parquet."""
    path = RAW_DIR / f"{name}.parquet"
    if isinstance(df, gpd.GeoDataFrame):
        df.to_parquet(path)
    else:
        df.to_parquet(path, index=False)
    return path


def load_raw(name: str) -> pd.DataFrame:
    """Load a parquet file from the raw data directory."""
    return pd.read_parquet(RAW_DIR / f"{name}.parquet")


def save_processed(df: pd.DataFrame | gpd.GeoDataFrame, name: str) -> Path:
    """Save a dataframe to the processed data directory as parquet."""
    path = PROCESSED_DIR / f"{name}.parquet"
    if isinstance(df, gpd.GeoDataFrame):
        df.to_parquet(path)
    else:
        df.to_parquet(path, index=False)
    return path


def load_processed(name: str) -> pd.DataFrame:
    """Load a parquet file from the processed data directory."""
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
