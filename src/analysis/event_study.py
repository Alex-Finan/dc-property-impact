"""Event study specification for validating parallel trends and tracing dynamic effects.

The event study replaces the single Post indicator with a full set of
relative-time dummies to visualize the treatment effect over time.

References:
- Sun & Abraham (2021). Interaction-weighted estimator for staggered settings.
- Roth (2022). "Pre-test with Caution." AER: Insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_relative_time(
    panel: pd.DataFrame,
    sale_date_col: str = "sale_date",
    event_date_col: str = "event_date",
    time_unit: str = "quarter",
) -> pd.DataFrame:
    """Create relative time variable (periods since treatment).

    Args:
        panel: Panel data with sale dates and event dates.
        sale_date_col: Column with property sale date.
        event_date_col: Column with DC event date.
        time_unit: 'quarter', 'month', or 'year'.

    Returns:
        Panel with 'relative_time' column added.
    """
    result = panel.copy()
    sale_dt = pd.to_datetime(result[sale_date_col])
    event_dt = pd.to_datetime(result[event_date_col])

    if time_unit == "quarter":
        sale_q = sale_dt.dt.year * 4 + sale_dt.dt.quarter
        event_q = event_dt.dt.year * 4 + event_dt.dt.quarter
        result["relative_time"] = sale_q - event_q
    elif time_unit == "month":
        sale_m = sale_dt.dt.year * 12 + sale_dt.dt.month
        event_m = event_dt.dt.year * 12 + event_dt.dt.month
        result["relative_time"] = sale_m - event_m
    elif time_unit == "year":
        result["relative_time"] = sale_dt.dt.year - event_dt.dt.year
    else:
        raise ValueError(f"Unknown time_unit: {time_unit}")

    return result


def bin_relative_time(
    panel: pd.DataFrame,
    min_period: int = -8,
    max_period: int = 12,
    omit_period: int = -1,
) -> pd.DataFrame:
    """Bin relative time into endpoint-aggregated categories.

    Periods beyond the window are grouped into endpoint bins to avoid
    losing observations while keeping the coefficient count manageable.

    Args:
        panel: Panel with 'relative_time' column.
        min_period: Minimum relative period (everything below is binned here).
        max_period: Maximum relative period (everything above is binned here).
        omit_period: Reference period to omit (typically -1).

    Returns:
        Panel with 'rel_time_binned' column and dummy variables.
    """
    result = panel.copy()

    result["rel_time_binned"] = result["relative_time"].clip(
        lower=min_period, upper=max_period
    )

    # Create dummies for each period except the omitted one
    periods = sorted(result["rel_time_binned"].unique())
    for p in periods:
        if p == omit_period:
            continue
        col_name = f"rt_{p}" if p < 0 else f"rt_plus_{p}"
        result[col_name] = ((result["rel_time_binned"] == p) & (result["treat"] == 1)).astype(int)

    return result


def estimate_event_study(
    panel: pd.DataFrame,
    min_period: int = -8,
    max_period: int = 12,
    omit_period: int = -1,
    controls: list[str] | None = None,
    time_fe: str = "sale_quarter",
    cluster_var: str | None = None,
) -> object:
    """Estimate the event study specification using pyfixest.

    Specification:
        ln(P_it) = SUM_k [b_k * (Treat_i x 1{t = event + k})] + g*X + d_i + t_t + e

    Args:
        panel: Panel with relative time and treatment indicators.
        min_period: Earliest relative period to include.
        max_period: Latest relative period to include.
        omit_period: Reference period (coefficient normalized to zero).
        controls: Control variable names.
        time_fe: Time fixed effect column.
        cluster_var: Clustering variable.

    Returns:
        pyfixest regression result.
    """
    import pyfixest as pf

    # Prepare binned dummies if not already done
    if "rel_time_binned" not in panel.columns:
        panel = bin_relative_time(panel, min_period, max_period, omit_period)

    # Collect the relative-time dummy columns
    rt_cols = [c for c in panel.columns if c.startswith("rt_")]
    if not rt_cols:
        raise ValueError("No relative-time dummy columns found. Run bin_relative_time first.")

    rhs_parts = rt_cols.copy()
    if controls:
        available = [c for c in controls if c in panel.columns]
        rhs_parts.extend(available)

    rhs = " + ".join(rhs_parts)
    fe_str = time_fe if time_fe in panel.columns else ""
    formula = f"ln_price ~ {rhs} | {fe_str}" if fe_str else f"ln_price ~ {rhs}"

    vcov = {"CRV1": cluster_var} if cluster_var else "hetero"
    return pf.feols(formula, data=panel, vcov=vcov)


def plot_event_study(
    result,
    omit_period: int = -1,
    title: str = "Event Study: DC Impact on Property Values",
    ylabel: str = "Effect on ln(price)",
    xlabel: str = "Quarters relative to DC announcement",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot event study coefficients with confidence intervals.

    Args:
        result: pyfixest or statsmodels regression result.
        omit_period: The omitted reference period (plotted as zero).
        title: Plot title.
        ylabel: Y-axis label.
        xlabel: X-axis label.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    # Extract coefficients and CIs for relative-time variables
    params = result.params() if hasattr(result, "params") else result.params
    conf = result.confint() if hasattr(result, "confint") else result.conf_int()

    rt_params = {k: v for k, v in params.items() if k.startswith("rt_")}

    if not rt_params:
        raise ValueError("No relative-time coefficients found in results.")

    # Parse period numbers from column names
    periods = []
    coefs = []
    ci_low = []
    ci_high = []

    for col, coef in sorted(rt_params.items()):
        if "plus_" in col:
            period = int(col.split("plus_")[1])
        else:
            period = int(col.split("rt_")[1])
        periods.append(period)
        coefs.append(coef)
        if isinstance(conf, pd.DataFrame):
            ci_low.append(conf.loc[col].iloc[0])
            ci_high.append(conf.loc[col].iloc[1])
        else:
            ci_low.append(conf[col][0])
            ci_high.append(conf[col][1])

    # Add the omitted period as zero
    periods.append(omit_period)
    coefs.append(0)
    ci_low.append(0)
    ci_high.append(0)

    # Sort by period
    order = np.argsort(periods)
    periods = np.array(periods)[order]
    coefs = np.array(coefs)[order]
    ci_low = np.array(ci_low)[order]
    ci_high = np.array(ci_high)[order]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(periods, coefs, yerr=[coefs - ci_low, ci_high - coefs],
                fmt="o-", capsize=3, color="steelblue", markersize=5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=-0.5, color="red", linestyle="--", alpha=0.5, label="Treatment")
    ax.fill_between(periods, ci_low, ci_high, alpha=0.15, color="steelblue")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
