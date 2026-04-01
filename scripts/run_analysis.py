"""
DC Property Impact — Difference-in-Differences Analysis
========================================================
Tract-level DiD using FHFA House Price Index as outcome and
proximity to data center parcels as treatment.

Produces:
  1. Treatment/control map
  2. Parallel trends plot (pre-treatment)
  3. Event study coefficients
  4. DiD regression table
  5. Distance decay plot
"""

import json
import csv
import os
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from shapely.geometry import shape, Point
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
EXT = PROJECT_ROOT / "data" / "external"
OUT = PROJECT_ROOT / "output"
OUT.mkdir(exist_ok=True)

print("=" * 70)
print("DC PROPERTY IMPACT — DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 70)

# =====================================================================
# 1. LOAD DATA
# =====================================================================
print("\n[1] Loading data...")

# --- FHFA Tract HPI ---
hpi = pd.read_csv(RAW / "fhfa_tract_hpi.csv")
hpi["tract"] = hpi["tract"].astype(str).str.zfill(11)
hpi_loudoun = hpi[hpi["tract"].str.startswith("51107")].copy()
hpi_loudoun["year"] = hpi_loudoun["year"].astype(int)
hpi_loudoun["hpi"] = pd.to_numeric(hpi_loudoun["hpi"], errors="coerce")
hpi_loudoun = hpi_loudoun.dropna(subset=["hpi"])
print(f"  FHFA HPI: {len(hpi_loudoun)} obs, {hpi_loudoun['tract'].nunique()} tracts, "
      f"{hpi_loudoun['year'].min()}-{hpi_loudoun['year'].max()}")

# --- DC Parcels ---
with open(RAW / "byright_dc_parcels_full.geojson") as f:
    dc_raw = json.load(f)

# Filter to actual data centers (LU_DISPLAY = 'Data Center')
dc_features = []
for feat in dc_raw["features"]:
    lu = feat["properties"].get("LU_DISPLAY", "")
    if lu == "Data Center":
        dc_features.append(feat)

dc_gdf = gpd.GeoDataFrame.from_features(dc_features, crs="EPSG:4326")
print(f"  Data center parcels: {len(dc_gdf)}")

# Also include broader industrial parcels that may be DCs
dc_all_features = []
dc_labels = {"Data Center", "Light Industrial Flex"}
for feat in dc_raw["features"]:
    lu = feat["properties"].get("LU_DISPLAY", "")
    if lu in dc_labels:
        dc_all_features.append(feat)
dc_all_gdf = gpd.GeoDataFrame.from_features(dc_all_features, crs="EPSG:4326")
print(f"  DC + Industrial Flex parcels: {len(dc_all_gdf)}")

# --- TIGER Tract Boundaries (Virginia) ---
import zipfile
tiger_zip = EXT / "tiger_tracts_51_2020.zip"
tracts_gdf = gpd.read_file(f"zip://{tiger_zip}")
tracts_gdf = tracts_gdf.to_crs(epsg=4326)

# Filter to Loudoun County
loudoun_tracts = tracts_gdf[tracts_gdf["COUNTYFP"] == "107"].copy()
print(f"  Loudoun census tracts: {len(loudoun_tracts)}")

# --- Census ACS (2022 for controls) ---
with open(RAW / "acs_loudoun_tracts_2022.json") as f:
    acs_raw = json.load(f)
acs_df = pd.DataFrame(acs_raw[1:], columns=acs_raw[0])
acs_df["tract_fips"] = acs_df["state"] + acs_df["county"] + acs_df["tract"]
for col in ["B19013_001E", "B01003_001E", "B25077_001E"]:
    acs_df[col] = pd.to_numeric(acs_df[col], errors="coerce")
acs_df = acs_df.rename(columns={
    "B19013_001E": "median_income",
    "B01003_001E": "population",
    "B25077_001E": "median_home_value",
})
print(f"  ACS tracts: {len(acs_df)}")

# =====================================================================
# 2. COMPUTE DISTANCE FROM EACH TRACT TO NEAREST DC
# =====================================================================
print("\n[2] Computing tract-to-DC distances...")

# Project to UTM 18N for meter-based distances
loudoun_proj = loudoun_tracts.to_crs(epsg=32618)
dc_proj = dc_gdf.to_crs(epsg=32618)

# Tract centroids
loudoun_proj["centroid"] = loudoun_proj.geometry.centroid
tract_coords = np.array([(g.x, g.y) for g in loudoun_proj["centroid"]])

# DC parcel centroids (drop null geometries)
dc_proj = dc_proj[dc_proj.geometry.notnull()].copy()
dc_proj = dc_proj[~dc_proj.geometry.is_empty].copy()
dc_centroids = dc_proj.geometry.centroid
dc_coords = np.array([(g.x, g.y) for g in dc_centroids])

# Nearest DC distance for each tract
tree = cKDTree(dc_coords)
distances, indices = tree.query(tract_coords, k=1)
distances_miles = distances / 1609.34

loudoun_tracts["dist_to_dc_mi"] = distances_miles
loudoun_tracts["nearest_dc_idx"] = indices

# Assign treatment rings
def assign_ring(d):
    if d < 1.0:
        return "0-1 mi"
    elif d < 2.0:
        return "1-2 mi"
    elif d < 3.0:
        return "2-3 mi"
    elif d < 5.0:
        return "3-5 mi"
    else:
        return "5+ mi"

loudoun_tracts["ring"] = loudoun_tracts["dist_to_dc_mi"].apply(assign_ring)
loudoun_tracts["treated"] = (loudoun_tracts["dist_to_dc_mi"] < 2.0).astype(int)

print("  Distance distribution:")
print(loudoun_tracts["ring"].value_counts().to_string(name=False))

# =====================================================================
# 3. BUILD PANEL DATASET
# =====================================================================
print("\n[3] Building panel dataset...")

# Merge HPI with tract characteristics
tract_info = loudoun_tracts[["GEOID", "dist_to_dc_mi", "ring", "treated"]].copy()
tract_info = tract_info.rename(columns={"GEOID": "tract"})

panel = hpi_loudoun.merge(tract_info, on="tract", how="inner")

# Add ACS controls
acs_slim = acs_df[["tract_fips", "median_income", "population", "median_home_value"]].copy()
panel = panel.merge(acs_slim, left_on="tract", right_on="tract_fips", how="left")

# Focus on 2000-2025 (DC boom era)
panel = panel[(panel["year"] >= 2000) & (panel["year"] <= 2025)].copy()

# Treatment timing: DC construction accelerated ~2010 in Loudoun
# Major buildout: 2010-2015 per industry reports
TREATMENT_YEAR = 2012
panel["post"] = (panel["year"] >= TREATMENT_YEAR).astype(int)
panel["treat_x_post"] = panel["treated"] * panel["post"]

# Create relative time for event study
panel["rel_year"] = panel["year"] - TREATMENT_YEAR

# Log HPI for percentage interpretation
panel["ln_hpi"] = np.log(panel["hpi"].clip(lower=1))

print(f"  Panel: {len(panel)} observations")
print(f"  Treated tracts: {panel[panel['treated']==1]['tract'].nunique()}")
print(f"  Control tracts: {panel[panel['treated']==0]['tract'].nunique()}")
print(f"  Treatment year: {TREATMENT_YEAR}")
print(f"  Years: {panel['year'].min()}-{panel['year'].max()}")

# =====================================================================
# 4. FIGURE 1: TREATMENT MAP
# =====================================================================
print("\n[4] Generating treatment map...")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

ring_colors = {
    "0-1 mi": "#d73027",
    "1-2 mi": "#fc8d59",
    "2-3 mi": "#fee090",
    "3-5 mi": "#91bfdb",
    "5+ mi": "#4575b4",
}

for ring_name, color in ring_colors.items():
    subset = loudoun_tracts[loudoun_tracts["ring"] == ring_name]
    if not subset.empty:
        subset.plot(ax=ax, color=color, edgecolor="white", linewidth=0.5, alpha=0.8)

# Plot DC locations
dc_gdf.plot(ax=ax, color="black", markersize=15, marker="^", zorder=5, alpha=0.7)

patches = [mpatches.Patch(color=c, label=r) for r, c in ring_colors.items()]
patches.append(plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
                          markersize=10, label="Data Centers"))
ax.legend(handles=patches, loc="upper left", fontsize=10, title="Distance to Nearest DC")
ax.set_title("Loudoun County Census Tracts by Distance to Nearest Data Center", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_axis_off()

plt.tight_layout()
fig.savefig(OUT / "fig1_treatment_map.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT / 'fig1_treatment_map.png'}")
plt.close()

# =====================================================================
# 5. FIGURE 2: PARALLEL TRENDS
# =====================================================================
print("\n[5] Generating parallel trends plot...")

trends = panel.groupby(["year", "treated"])["hpi"].mean().reset_index()
treated_trend = trends[trends["treated"] == 1]
control_trend = trends[trends["treated"] == 0]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(treated_trend["year"], treated_trend["hpi"], "o-", color="#d73027",
        linewidth=2, markersize=5, label="Treated (<2 mi from DC)")
ax.plot(control_trend["year"], control_trend["hpi"], "s-", color="#4575b4",
        linewidth=2, markersize=5, label="Control (>2 mi from DC)")
ax.axvline(x=TREATMENT_YEAR, color="gray", linestyle="--", alpha=0.7,
           label=f"Treatment ({TREATMENT_YEAR})")
ax.axvline(x=2022, color="green", linestyle=":", alpha=0.5,
           label="Silver Line Phase 2 (2022)")

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("House Price Index", fontsize=12)
ax.set_title("House Price Trends: Treated vs. Control Tracts\nLoudoun County, VA", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "fig2_parallel_trends.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT / 'fig2_parallel_trends.png'}")
plt.close()

# =====================================================================
# 6. FIGURE 3: TRENDS BY DISTANCE RING
# =====================================================================
print("\n[6] Generating trends by distance ring...")

ring_trends = panel.groupby(["year", "ring"])["hpi"].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
ring_order = ["0-1 mi", "1-2 mi", "2-3 mi", "3-5 mi", "5+ mi"]
for ring_name in ring_order:
    subset = ring_trends[ring_trends["ring"] == ring_name]
    if not subset.empty:
        ax.plot(subset["year"], subset["hpi"], "o-", color=ring_colors[ring_name],
                linewidth=2, markersize=4, label=ring_name)

ax.axvline(x=TREATMENT_YEAR, color="gray", linestyle="--", alpha=0.7)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("House Price Index", fontsize=12)
ax.set_title("House Price Trends by Distance to Nearest Data Center\nLoudoun County, VA",
             fontsize=14)
ax.legend(fontsize=10, title="Distance Ring")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "fig3_trends_by_ring.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT / 'fig3_trends_by_ring.png'}")
plt.close()

# =====================================================================
# 7. DiD REGRESSIONS
# =====================================================================
print("\n[7] Running DiD regressions...")

# --- Model 1: Basic DiD (no controls) ---
model1 = smf.ols("ln_hpi ~ treated + post + treat_x_post", data=panel).fit(
    cov_type="cluster", cov_kwds={"groups": panel["tract"]}
)

# --- Model 2: DiD with year fixed effects ---
panel["year_fe"] = pd.Categorical(panel["year"])
model2 = smf.ols("ln_hpi ~ treated + treat_x_post + C(year_fe)", data=panel).fit(
    cov_type="cluster", cov_kwds={"groups": panel["tract"]}
)

# --- Model 3: DiD with tract + year FE ---
panel["tract_fe"] = pd.Categorical(panel["tract"])
model3 = smf.ols("ln_hpi ~ treat_x_post + C(tract_fe) + C(year_fe)", data=panel).fit(
    cov_type="cluster", cov_kwds={"groups": panel["tract"]}
)

# --- Model 4: Continuous distance ---
panel["inv_dist"] = 1 / panel["dist_to_dc_mi"].clip(lower=0.1)
panel["inv_dist_x_post"] = panel["inv_dist"] * panel["post"]
model4 = smf.ols("ln_hpi ~ inv_dist_x_post + inv_dist + C(tract_fe) + C(year_fe)",
                  data=panel).fit(
    cov_type="cluster", cov_kwds={"groups": panel["tract"]}
)

print("\n" + "=" * 70)
print("TABLE 1: DIFFERENCE-IN-DIFFERENCES REGRESSION RESULTS")
print("Dependent Variable: ln(House Price Index)")
print(f"Treatment: Tract centroid < 2 miles from Data Center")
print(f"Treatment Year: {TREATMENT_YEAR}")
print("Standard errors clustered at tract level")
print("=" * 70)

results_table = []
for name, model, var in [
    ("(1) Basic DiD", model1, "treat_x_post"),
    ("(2) + Year FE", model2, "treat_x_post"),
    ("(3) + Tract & Year FE", model3, "treat_x_post"),
    ("(4) Continuous Dist (Tract+Year FE)", model4, "inv_dist_x_post"),
]:
    coef = model.params[var]
    se = model.bse[var]
    pval = model.pvalues[var]
    ci = model.conf_int().loc[var]
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    pct_effect = (np.exp(coef) - 1) * 100

    results_table.append({
        "Model": name,
        "Coefficient": f"{coef:.4f}{stars}",
        "SE": f"({se:.4f})",
        "95% CI": f"[{ci.iloc[0]:.4f}, {ci.iloc[1]:.4f}]",
        "% Effect": f"{pct_effect:.1f}%",
        "N": model.nobs,
        "R²": f"{model.rsquared:.3f}",
    })
    print(f"\n{name}")
    print(f"  Coefficient: {coef:.4f}{stars}  SE: ({se:.4f})")
    print(f"  95% CI: [{ci.iloc[0]:.4f}, {ci.iloc[1]:.4f}]")
    print(f"  Implied % effect: {pct_effect:+.1f}%")
    print(f"  N={int(model.nobs)}, R²={model.rsquared:.3f}")

# =====================================================================
# 8. FIGURE 4: EVENT STUDY
# =====================================================================
print("\n\n[8] Running event study...")

# Create relative-year dummies interacted with treatment
# Omit year -1 as reference
# Use patsy-safe names: rtm10 = relative time minus 10, rtp5 = relative time plus 5
event_dummies = []
event_dummy_map = {}  # col_name -> relative year
for t in range(-10, 14):
    if t == -1:
        continue
    col = f"rtm{abs(t)}" if t < 0 else f"rtp{t}"
    panel[col] = ((panel["rel_year"] == t) & (panel["treated"] == 1)).astype(int)
    event_dummies.append(col)
    event_dummy_map[col] = t

# Regression with tract and year FE — use matrix approach to avoid patsy formula length issues
Y = panel["ln_hpi"].values
X_event = panel[event_dummies].values

# Build tract and year dummies manually
tract_dummies = pd.get_dummies(panel["tract"], drop_first=True, prefix="tr")
year_dummies = pd.get_dummies(panel["year"], drop_first=True, prefix="yr")

X_full = np.hstack([X_event, tract_dummies.values, year_dummies.values])
col_names = event_dummies + list(tract_dummies.columns) + list(year_dummies.columns)
X_full = sm.add_constant(X_full)
col_names = ["const"] + col_names

event_model = sm.OLS(Y, X_full).fit(
    cov_type="cluster", cov_kwds={"groups": panel["tract"].values}
)
# Map back column names
event_model_params = pd.Series(event_model.params, index=col_names)
event_model_bse = pd.Series(event_model.bse, index=col_names)
ci_arr = event_model.conf_int()
event_model_ci = pd.DataFrame(ci_arr, index=col_names, columns=["ci_low", "ci_high"])

# Extract coefficients
periods = []
coefs = []
ci_low = []
ci_high = []

for col in event_dummies:
    t = event_dummy_map[col]
    if col in event_model_params.index:
        periods.append(t)
        coefs.append(event_model_params[col])
        ci_low.append(event_model_ci.loc[col, "ci_low"])
        ci_high.append(event_model_ci.loc[col, "ci_high"])

# Add omitted period
periods.append(-1)
coefs.append(0)
ci_low.append(0)
ci_high.append(0)

# Sort
order = np.argsort(periods)
periods = np.array(periods)[order]
coefs = np.array(coefs)[order]
ci_low = np.array(ci_low)[order]
ci_high = np.array(ci_high)[order]

fig, ax = plt.subplots(figsize=(14, 7))
ax.errorbar(periods, coefs, yerr=[coefs - ci_low, ci_high - coefs],
            fmt="o", capsize=3, color="#d73027", markersize=5, linewidth=1.5,
            ecolor="#fc8d59", elinewidth=1)
ax.fill_between(periods, ci_low, ci_high, alpha=0.12, color="#d73027")
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
ax.axvline(x=-0.5, color="black", linestyle="--", alpha=0.5, linewidth=2)

ax.annotate("Treatment\n(DC boom)", xy=(-0.5, ax.get_ylim()[1] * 0.85),
            fontsize=10, ha="center", color="black", alpha=0.7)

ax.set_xlabel("Years Relative to Treatment (2012)", fontsize=12)
ax.set_ylabel("Effect on ln(HPI)\n(Treated vs Control, relative to t=-1)", fontsize=12)
ax.set_title("Event Study: Effect of Data Center Proximity on House Prices\n"
             "Loudoun County, VA | Tract + Year FE | Clustered SEs", fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "fig4_event_study.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT / 'fig4_event_study.png'}")
plt.close()

# =====================================================================
# 9. FIGURE 5: DISTANCE DECAY
# =====================================================================
print("\n[9] Computing distance decay...")

# For each tract, compute pre vs post change in HPI
pre_hpi = panel[panel["post"] == 0].groupby("tract")["hpi"].mean()
post_hpi = panel[panel["post"] == 1].groupby("tract")["hpi"].mean()
change = (post_hpi - pre_hpi).reset_index()
change.columns = ["tract", "hpi_change"]

dist_info = panel[["tract", "dist_to_dc_mi"]].drop_duplicates()
decay_df = change.merge(dist_info, on="tract")

# Compute % change
pre_mean = panel[panel["post"] == 0].groupby("tract")["hpi"].mean().reset_index()
pre_mean.columns = ["tract", "pre_hpi"]
decay_df = decay_df.merge(pre_mean, on="tract")
decay_df["pct_change"] = (decay_df["hpi_change"] / decay_df["pre_hpi"]) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter
ax1 = axes[0]
scatter = ax1.scatter(decay_df["dist_to_dc_mi"], decay_df["pct_change"],
                      c=decay_df["dist_to_dc_mi"], cmap="RdYlBu", s=60,
                      edgecolors="white", linewidth=0.5, alpha=0.8)
z = np.polyfit(decay_df["dist_to_dc_mi"], decay_df["pct_change"], 2)
p = np.poly1d(z)
x_smooth = np.linspace(decay_df["dist_to_dc_mi"].min(),
                       decay_df["dist_to_dc_mi"].max(), 100)
ax1.plot(x_smooth, p(x_smooth), "k--", linewidth=2, alpha=0.7, label="Quadratic fit")
ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
ax1.set_xlabel("Distance to Nearest Data Center (miles)", fontsize=12)
ax1.set_ylabel("HPI Change (%)\n(Pre-2012 avg vs Post-2012 avg)", fontsize=12)
ax1.set_title("Distance Decay of DC Impact", fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label="Distance (mi)")

# Binned bar chart
ax2 = axes[1]
bins = [0, 1, 2, 3, 5, 10]
labels = ["0-1", "1-2", "2-3", "3-5", "5+"]
decay_df["dist_bin"] = pd.cut(decay_df["dist_to_dc_mi"], bins=bins, labels=labels)
binned = decay_df.groupby("dist_bin", observed=True)["pct_change"].agg(["mean", "std", "count"])
binned["se"] = binned["std"] / np.sqrt(binned["count"])

colors = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]
bars = ax2.bar(range(len(binned)), binned["mean"], yerr=binned["se"] * 1.96,
               capsize=5, color=colors[:len(binned)], edgecolor="white", linewidth=1)
ax2.set_xticks(range(len(binned)))
ax2.set_xticklabels(binned.index, fontsize=11)
ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Distance to Nearest DC (miles)", fontsize=12)
ax2.set_ylabel("Mean HPI Change (%)", fontsize=12)
ax2.set_title("HPI Change by Distance Ring", fontsize=13)
ax2.grid(True, alpha=0.3, axis="y")

# Add count labels
for i, (_, row) in enumerate(binned.iterrows()):
    ax2.text(i, row["mean"] + row["se"] * 1.96 + 1, f"n={int(row['count'])}",
             ha="center", fontsize=9, color="gray")

plt.tight_layout()
fig.savefig(OUT / "fig5_distance_decay.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT / 'fig5_distance_decay.png'}")
plt.close()

# =====================================================================
# 10. FIGURE 6: ROBUSTNESS — VARYING TREATMENT YEAR
# =====================================================================
print("\n[10] Robustness: varying treatment year...")

treatment_years = [2008, 2010, 2012, 2014, 2016, 2018, 2020]
robustness_results = []

for ty in treatment_years:
    panel_r = panel.copy()
    panel_r["post_r"] = (panel_r["year"] >= ty).astype(int)
    panel_r["txp_r"] = panel_r["treated"] * panel_r["post_r"]

    try:
        m = smf.ols("ln_hpi ~ txp_r + C(tract_fe) + C(year_fe)", data=panel_r).fit(
            cov_type="cluster", cov_kwds={"groups": panel_r["tract"]}
        )
        ci = m.conf_int().loc["txp_r"]
        robustness_results.append({
            "year": ty,
            "coef": m.params["txp_r"],
            "se": m.bse["txp_r"],
            "ci_low": ci.iloc[0],
            "ci_high": ci.iloc[1],
            "pval": m.pvalues["txp_r"],
        })
    except Exception as e:
        print(f"  Skipped {ty}: {e}")

rob_df = pd.DataFrame(robustness_results)

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(rob_df["year"], rob_df["coef"],
            yerr=[rob_df["coef"] - rob_df["ci_low"], rob_df["ci_high"] - rob_df["coef"]],
            fmt="o-", capsize=5, color="#d73027", markersize=8, linewidth=2)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
ax.fill_between(rob_df["year"], rob_df["ci_low"], rob_df["ci_high"],
                alpha=0.15, color="#d73027")

# Mark significance
for _, row in rob_df.iterrows():
    if row["pval"] < 0.05:
        ax.annotate("*", xy=(row["year"], row["ci_high"]),
                    fontsize=16, ha="center", color="#d73027")

ax.set_xlabel("Treatment Year", fontsize=12)
ax.set_ylabel("DiD Coefficient on ln(HPI)", fontsize=12)
ax.set_title("Robustness: DiD Estimate Across Alternative Treatment Years\n"
             "Tract + Year FE | Clustered SEs | * = p<0.05", fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "fig6_robustness_treatment_year.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT / 'fig6_robustness_treatment_year.png'}")
plt.close()

# =====================================================================
# 11. FIGURE 7: HPI LEVEL MAP (CHOROPLETH)
# =====================================================================
print("\n[11] Generating HPI choropleth map...")

latest_hpi = panel[panel["year"] == panel["year"].max()].groupby("tract")["hpi"].mean()
loudoun_tracts_map = loudoun_tracts.merge(
    latest_hpi.reset_index(), left_on="GEOID", right_on="tract", how="left"
)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
loudoun_tracts_map.plot(
    ax=ax, column="hpi", cmap="RdYlGn", legend=True,
    legend_kwds={"label": f"HPI ({panel['year'].max()})", "shrink": 0.6},
    edgecolor="white", linewidth=0.5, missing_kwds={"color": "lightgray"}
)
dc_gdf.plot(ax=ax, color="black", markersize=15, marker="^", zorder=5, alpha=0.7)
ax.set_title(f"House Price Index by Census Tract ({panel['year'].max()})\nLoudoun County, VA",
             fontsize=14)
ax.set_axis_off()

plt.tight_layout()
fig.savefig(OUT / "fig7_hpi_choropleth.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT / 'fig7_hpi_choropleth.png'}")
plt.close()

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {OUT}")
print(f"Figures generated:")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}")

print(f"\nKey finding (Model 3 — Tract + Year FE):")
coef3 = model3.params["treat_x_post"]
se3 = model3.bse["treat_x_post"]
pval3 = model3.pvalues["treat_x_post"]
pct3 = (np.exp(coef3) - 1) * 100
stars3 = "***" if pval3 < 0.01 else "**" if pval3 < 0.05 else "*" if pval3 < 0.1 else ""
print(f"  Treat x Post coefficient: {coef3:.4f}{stars3} (SE: {se3:.4f})")
print(f"  Implied effect: {pct3:+.1f}% on house prices")
print(f"  p-value: {pval3:.4f}")
