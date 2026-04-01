"""Synthetic control methods for aggregate-level analysis.

Constructs a weighted combination of untreated geographic units that
best approximates the treated unit's pre-treatment trajectory.

References:
- Abadie, Diamond & Hainmueller (2010, 2015). Synthetic control method.
- Abadie (2021). "Using Synthetic Controls." JEL.
"""

import pandas as pd
import numpy as np


def prepare_synth_panel(
    tract_data: pd.DataFrame,
    outcome_col: str = "median_price",
    time_col: str = "year",
    unit_col: str = "tract_fips",
    treated_tracts: list[str] | None = None,
    donor_tracts: list[str] | None = None,
    predictors: list[str] | None = None,
) -> dict:
    """Prepare data for synthetic control estimation.

    Args:
        tract_data: Panel data at the geographic unit level.
        outcome_col: Column with the outcome variable.
        time_col: Column with the time period.
        unit_col: Column identifying each geographic unit.
        treated_tracts: List of tract FIPS codes in the treatment area.
        donor_tracts: List of potential donor tract FIPS codes.
        predictors: Columns to use as matching predictors.

    Returns:
        Dict with keys: 'treated', 'donors', 'outcome', 'predictors', 'time'.
    """
    if treated_tracts is None:
        raise ValueError("Must specify treated_tracts.")

    all_tracts = tract_data[unit_col].unique()
    if donor_tracts is None:
        donor_tracts = [t for t in all_tracts if t not in treated_tracts]

    # Aggregate treated tracts into a single unit (population-weighted average)
    treated_mask = tract_data[unit_col].isin(treated_tracts)
    treated_panel = (
        tract_data[treated_mask]
        .groupby(time_col)[outcome_col]
        .mean()
        .reset_index()
    )
    treated_panel["unit"] = "treated"

    donor_panel = tract_data[tract_data[unit_col].isin(donor_tracts)].copy()

    return {
        "treated_panel": treated_panel,
        "donor_panel": donor_panel,
        "outcome_col": outcome_col,
        "time_col": time_col,
        "unit_col": unit_col,
        "predictors": predictors or [],
        "treated_tracts": treated_tracts,
        "donor_tracts": donor_tracts,
    }


def estimate_synthetic_control(
    synth_data: dict,
    treatment_year: int,
) -> dict:
    """Estimate a synthetic control using pysyncon.

    Args:
        synth_data: Output from prepare_synth_panel.
        treatment_year: Year the treatment begins.

    Returns:
        Dict with synthetic control weights, fitted values, and treatment effect.
    """
    from pysyncon import Dataprep, Synth

    donor_panel = synth_data["donor_panel"]
    treated_panel = synth_data["treated_panel"]
    outcome_col = synth_data["outcome_col"]
    time_col = synth_data["time_col"]
    unit_col = synth_data["unit_col"]

    # Combine into a single panel for pysyncon
    treated_expanded = treated_panel.rename(columns={"unit": unit_col})
    treated_expanded[unit_col] = "treated"
    combined = pd.concat([donor_panel, treated_expanded], ignore_index=True)

    all_years = sorted(combined[time_col].unique())
    pre_years = [y for y in all_years if y < treatment_year]
    post_years = [y for y in all_years if y >= treatment_year]

    dataprep = Dataprep(
        foo=combined,
        predictors=[outcome_col],
        predictors_op="mean",
        dependent=outcome_col,
        unit_variable=unit_col,
        time_variable=time_col,
        treatment_identifier="treated",
        controls_identifier=synth_data["donor_tracts"],
        time_predictors_prior=pre_years,
        time_optimize_ssr=pre_years,
        time_period=all_years,
    )

    synth = Synth()
    synth.fit(dataprep=dataprep)

    return {
        "synth": synth,
        "dataprep": dataprep,
        "pre_years": pre_years,
        "post_years": post_years,
        "treatment_year": treatment_year,
    }


def run_placebo_tests(
    synth_data: dict,
    treatment_year: int,
    n_placebos: int | None = None,
) -> pd.DataFrame:
    """Run in-space placebo tests for inference.

    Iteratively assigns each donor unit as the 'treated' unit and
    estimates a synthetic control. The treatment effect is compared
    to the distribution of placebo effects for inference.

    Args:
        synth_data: Output from prepare_synth_panel.
        treatment_year: Year the treatment begins.
        n_placebos: Number of placebo units to test (default: all donors).

    Returns:
        DataFrame with placebo effect estimates for each donor unit.
    """
    donor_tracts = synth_data["donor_tracts"]
    if n_placebos is not None:
        donor_tracts = donor_tracts[:n_placebos]

    results = []
    for tract in donor_tracts:
        placebo_treated = [tract]
        placebo_donors = [t for t in synth_data["donor_tracts"] if t != tract]

        placebo_data = {
            **synth_data,
            "treated_tracts": placebo_treated,
            "donor_tracts": placebo_donors,
        }

        try:
            placebo_result = estimate_synthetic_control(placebo_data, treatment_year)
            results.append({
                "unit": tract,
                "type": "placebo",
                "result": placebo_result,
            })
        except Exception:
            continue

    return results
