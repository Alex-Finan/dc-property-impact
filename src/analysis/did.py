"""Difference-in-Differences estimation for data center property impact.

Implements both naive TWFE and modern staggered DiD estimators.

References:
- Callaway & Sant'Anna (2021). "DiD with Multiple Time Periods." JoE.
- Sun & Abraham (2021). "Estimating Dynamic Treatment Effects." JoE.
- Goodman-Bacon (2021). "Difference-in-Differences with Variation in Treatment Timing." JoE.
"""

import pandas as pd
import numpy as np


def prepare_did_panel(
    properties: pd.DataFrame,
    treatment_col: str = "treatment_ring",
    time_col: str = "sale_quarter",
    price_col: str = "ln_price",
    treated_rings: list[str] | None = None,
    control_rings: list[str] | None = None,
) -> pd.DataFrame:
    """Prepare a panel dataset for DiD estimation.

    Args:
        properties: Property transaction data with treatment ring assignments.
        treatment_col: Column indicating the treatment ring.
        time_col: Column with the time period identifier.
        price_col: Column with the (log) price.
        treated_rings: Ring names that constitute the treatment group.
        control_rings: Ring names that constitute the control group.

    Returns:
        Panel DataFrame with treat, post, and treat_x_post indicators.
    """
    if treated_rings is None:
        treated_rings = ["inner", "middle", "outer"]
    if control_rings is None:
        control_rings = ["control"]

    # Filter to treatment and control groups (exclude buffer)
    mask = properties[treatment_col].isin(treated_rings + control_rings)
    panel = properties[mask].copy()

    # Binary treatment indicator
    panel["treat"] = panel[treatment_col].isin(treated_rings).astype(int)

    return panel


def estimate_twfe_did(
    panel: pd.DataFrame,
    controls: list[str] | None = None,
    entity_fe: str | None = None,
    time_fe: str = "sale_quarter",
    cluster_var: str | None = None,
) -> object:
    """Estimate a two-way fixed effects DiD model using pyfixest.

    Specification:
        ln(P_it) = b * (Treat_i x Post_t) + g*X_it + d_i + t_t + e_it

    Args:
        panel: Prepared panel data from prepare_did_panel.
        controls: List of control variable names.
        entity_fe: Column for entity (property/tract) fixed effects.
        time_fe: Column for time fixed effects.
        cluster_var: Column to cluster standard errors on.

    Returns:
        pyfixest regression result.
    """
    import pyfixest as pf

    # Build formula
    dep_var = "ln_price"
    treat_var = "treat_x_post"

    if treat_var not in panel.columns:
        raise ValueError(
            "Panel must contain 'treat_x_post' column. "
            "Use assign_post_treatment() first."
        )

    rhs_parts = [treat_var]
    if controls:
        available = [c for c in controls if c in panel.columns]
        rhs_parts.extend(available)

    rhs = " + ".join(rhs_parts)

    # Add fixed effects
    fe_parts = []
    if entity_fe and entity_fe in panel.columns:
        fe_parts.append(entity_fe)
    if time_fe and time_fe in panel.columns:
        fe_parts.append(time_fe)

    if fe_parts:
        fe_str = " + ".join(fe_parts)
        formula = f"{dep_var} ~ {rhs} | {fe_str}"
    else:
        formula = f"{dep_var} ~ {rhs}"

    # Cluster specification
    vcov = {"CRV1": cluster_var} if cluster_var else "hetero"

    result = pf.feols(formula, data=panel, vcov=vcov)
    return result


def estimate_distance_gradient(
    panel: pd.DataFrame,
    distance_col: str = "dist_to_nearest_dc_mi",
    post_col: str = "post",
    controls: list[str] | None = None,
    time_fe: str = "sale_quarter",
    cluster_var: str | None = None,
) -> object:
    """Estimate the continuous distance gradient of DC impact.

    Instead of discrete treatment rings, uses continuous distance
    interacted with post-treatment to trace the decay function.

    Specification:
        ln(P_it) = b1*(1/dist_i * Post_t) + b2*(dist_i) + g*X_it + t_t + e_it
    """
    import pyfixest as pf

    result = panel.copy()
    result["inv_distance"] = 1 / result[distance_col].clip(lower=0.1)
    result["inv_dist_x_post"] = result["inv_distance"] * result[post_col]

    rhs_parts = ["inv_dist_x_post", "inv_distance"]
    if controls:
        available = [c for c in controls if c in result.columns]
        rhs_parts.extend(available)

    rhs = " + ".join(rhs_parts)
    fe_str = time_fe if time_fe in result.columns else ""
    formula = f"ln_price ~ {rhs} | {fe_str}" if fe_str else f"ln_price ~ {rhs}"

    vcov = {"CRV1": cluster_var} if cluster_var else "hetero"
    return pf.feols(formula, data=result, vcov=vcov)


def assign_post_treatment(
    panel: pd.DataFrame,
    dc_event_dates: pd.DataFrame,
    nearest_dc_col: str = "nearest_dc_id",
    sale_date_col: str = "sale_date",
) -> pd.DataFrame:
    """Assign post-treatment indicators based on DC-specific event dates.

    For staggered adoption: each property's 'post' indicator depends
    on when its nearest DC was announced/permitted.

    Args:
        panel: Panel data with nearest DC assignments.
        dc_event_dates: DataFrame mapping dc_id -> event_date.
        nearest_dc_col: Column identifying the nearest DC.
        sale_date_col: Column with the property sale date.

    Returns:
        Panel with 'post' and 'treat_x_post' columns added.
    """
    result = panel.copy()

    # Merge event dates
    result = result.merge(
        dc_event_dates[["dc_id", "event_date"]],
        left_on=nearest_dc_col,
        right_on="dc_id",
        how="left",
    )

    result["post"] = (
        pd.to_datetime(result[sale_date_col]) >= pd.to_datetime(result["event_date"])
    ).astype(int)

    result["treat_x_post"] = result["treat"] * result["post"]

    return result
