"""Statistical plots for analysis results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distance_decay(
    decay_df: pd.DataFrame,
    title: str = "Distance Decay of DC Impact on Property Values",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot the distance decay function from binned DiD estimates.

    Args:
        decay_df: Output from spatial.compute_distance_decay().
        title: Plot title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(decay_df["dist_bin"], decay_df["diff"], width=0.2,
           color="steelblue", alpha=0.7, edgecolor="white")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)

    ax.set_xlabel("Distance to Nearest Data Center (miles)", fontsize=12)
    ax.set_ylabel("DiD Effect on ln(price)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_synthetic_control(
    treated_values: pd.Series,
    synth_values: pd.Series,
    treatment_year: int,
    title: str = "Synthetic Control: Treated vs. Synthetic",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot treated unit against its synthetic control.

    Args:
        treated_values: Time series of treated unit outcomes.
        synth_values: Time series of synthetic control outcomes.
        treatment_year: Year treatment begins.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    # Top panel: Levels
    ax1 = axes[0]
    ax1.plot(treated_values.index, treated_values.values,
             "b-", linewidth=2, label="Treated")
    ax1.plot(synth_values.index, synth_values.values,
             "r--", linewidth=2, label="Synthetic Control")
    ax1.axvline(x=treatment_year, color="gray", linestyle="--", alpha=0.7)
    ax1.set_ylabel("Outcome", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Gap (treatment effect)
    ax2 = axes[1]
    gap = treated_values - synth_values
    ax2.plot(gap.index, gap.values, "k-", linewidth=2)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    ax2.axvline(x=treatment_year, color="gray", linestyle="--", alpha=0.7)
    ax2.fill_between(gap.index, 0, gap.values,
                     where=gap.values > 0, alpha=0.3, color="green", label="Positive effect")
    ax2.fill_between(gap.index, 0, gap.values,
                     where=gap.values < 0, alpha=0.3, color="red", label="Negative effect")
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Treatment Effect", fontsize=12)
    ax2.set_title("Gap: Treated - Synthetic", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_coefficient_comparison(
    models: dict[str, object],
    var_name: str = "treat_x_post",
    title: str = "DiD Coefficient Across Specifications",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot DiD coefficient estimates across multiple model specifications.

    Useful for robustness checks: showing the estimate is stable across
    different control sets, distance thresholds, and time windows.

    Args:
        models: Dict mapping model names to regression results.
        var_name: Name of the DiD coefficient to compare.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    names = []
    coefs = []
    ci_low = []
    ci_high = []

    for name, result in models.items():
        params = result.params() if hasattr(result, "params") else result.params
        conf = result.confint() if hasattr(result, "confint") else result.conf_int()

        if var_name in params.index if hasattr(params, "index") else var_name in params:
            names.append(name)
            coefs.append(params[var_name])
            if isinstance(conf, pd.DataFrame):
                ci_low.append(conf.loc[var_name].iloc[0])
                ci_high.append(conf.loc[var_name].iloc[1])
            else:
                ci_low.append(conf[var_name][0])
                ci_high.append(conf[var_name][1])

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.6)))

    y_pos = range(len(names))
    coefs = np.array(coefs)
    ci_low = np.array(ci_low)
    ci_high = np.array(ci_high)

    ax.errorbar(coefs, y_pos, xerr=[coefs - ci_low, ci_high - coefs],
                fmt="o", capsize=5, color="steelblue", markersize=8)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names)
    ax.set_xlabel(f"Coefficient on {var_name}", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
