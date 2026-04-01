"""Hedonic pricing model for property valuation.

The hedonic model decomposes property prices into implicit prices for
individual characteristics (Rosen 1974). This serves as the foundation
for the DiD specification.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col


def prepare_hedonic_features(
    df: pd.DataFrame,
    property_controls: list[str],
    neighborhood_controls: list[str],
) -> pd.DataFrame:
    """Prepare feature matrix for hedonic regression.

    Args:
        df: Property transaction DataFrame.
        property_controls: Property-level hedonic variables (sqft, beds, etc.).
        neighborhood_controls: Neighborhood-level variables (income, schools, etc.).

    Returns:
        DataFrame with log price, controls, and dummy variables for categoricals.
    """
    result = df.copy()

    # Log transform price
    price_col = _find_price_column(df)
    result["ln_price"] = np.log(result[price_col].clip(lower=1))

    # Log transform skewed continuous variables
    skewed_cols = ["square_feet", "lot_size", "median_household_income"]
    for col in skewed_cols:
        if col in result.columns:
            result[f"ln_{col}"] = np.log(result[col].clip(lower=1))

    # Create age variable if year_built exists
    if "year_built" in result.columns:
        sale_year = _extract_sale_year(result)
        result["age"] = sale_year - result["year_built"]
        result["age_sq"] = result["age"] ** 2

    return result


def run_hedonic_regression(
    df: pd.DataFrame,
    controls: list[str],
    fe_vars: list[str] | None = None,
    cluster_var: str | None = None,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run a hedonic pricing regression.

    Specification: ln(P) = a + b*X + FE + e

    Args:
        df: Prepared DataFrame with ln_price and control variables.
        controls: List of control variable column names.
        fe_vars: Columns to include as fixed effects (converted to dummies).
        cluster_var: Column to cluster standard errors on.

    Returns:
        Statsmodels regression results.
    """
    y = df["ln_price"].dropna()
    available_controls = [c for c in controls if c in df.columns]
    X = df.loc[y.index, available_controls].copy()

    # Add fixed effects as dummies
    if fe_vars:
        for fe in fe_vars:
            if fe in df.columns:
                dummies = pd.get_dummies(df.loc[y.index, fe], prefix=fe, drop_first=True)
                X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = y.loc[X.index]

    if cluster_var and cluster_var in df.columns:
        groups = df.loc[y.index, cluster_var]
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
    else:
        model = sm.OLS(y, X).fit(cov_type="HC1")

    return model


def _find_price_column(df: pd.DataFrame) -> str:
    """Find the sale price column by name heuristics."""
    candidates = ["sale_price", "price", "sales_price", "sold_price", "amount"]
    for c in candidates:
        if c in df.columns:
            return c
    price_cols = [c for c in df.columns if "price" in c.lower() or "sale" in c.lower()]
    if price_cols:
        return price_cols[0]
    raise ValueError(f"No price column found. Columns: {list(df.columns)}")


def _extract_sale_year(df: pd.DataFrame) -> pd.Series:
    """Extract the sale year from date columns."""
    date_cols = [c for c in df.columns if "date" in c.lower() and "sale" in c.lower()]
    if date_cols:
        return pd.to_datetime(df[date_cols[0]], errors="coerce").dt.year
    year_cols = [c for c in df.columns if "year" in c.lower() and "sale" in c.lower()]
    if year_cols:
        return df[year_cols[0]]
    return pd.Series(2023, index=df.index)
