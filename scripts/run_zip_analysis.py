"""
DC Property Impact — ZIP-Code Level Analysis
=============================================
Leverages the expanded dataset:
  - FHFA ZIP5 HPI: 52 NoVA ZIPs, 1975-2025
  - NVRC DC locations: 265 DCs across Loudoun/Fairfax/PW
  - Realtor.com: median listing price, DOM, inventory (2016-2026)
  - 132k Loudoun parcels for spatial context

Key improvements over tract-level:
  - 52 ZIPs across 3 counties (vs 53 tracts in 1 county)
  - Cross-county variation: Fairfax (20 DCs) vs PW (46 DCs) vs Loudoun (199 DCs)
  - Dose-response: ZIP-level DC count varies from 0 to 80+
  - Realtor.com gives actual listing prices (not just an index)
  - 1975-2025 time series (vs 2000-2025 before)

Produces figures in output/zip/
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import statsmodels.api as sm
from shapely.geometry import Point
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
})

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
EXT = PROJECT / "data" / "external"
OUT = PROJECT / "output" / "zip"
OUT.mkdir(parents=True, exist_ok=True)

TREATMENT_YEAR = 2012

print("=" * 70)
print("ZIP-CODE LEVEL ANALYSIS")
print("=" * 70)

# =====================================================================
# LOAD DATA
# =====================================================================
print("\n[Loading data...]")

# --- FHFA ZIP5 HPI ---
hpi = pd.read_excel(RAW / "fhfa_zip5_hpi.csv", engine="openpyxl", header=5)
hpi.columns = ["zip5", "year", "annual_change", "hpi", "hpi_1990", "hpi_2000"]
hpi["zip5"] = hpi["zip5"].astype(str).str.zfill(5)
hpi["year"] = pd.to_numeric(hpi["year"], errors="coerce")
hpi["hpi"] = pd.to_numeric(hpi["hpi"], errors="coerce")
hpi = hpi.dropna(subset=["year", "hpi"])
hpi["year"] = hpi["year"].astype(int)

loudoun_zips = {'20105','20117','20120','20129','20130','20132','20134','20135','20141',
                '20147','20148','20152','20158','20164','20165','20166','20175','20176','20180'}
fairfax_zips = {'22003','22015','22027','22030','22031','22032','22033','22039','22041','22042',
                '22043','22044','22046','22060','22066','22079','22101','22102','22124','22150',
                '22151','22152','22153','22180','22181','22182'}
pw_zips = {'20109','20110','20111','20112','20136','20137','20143','20155','20169','20181'}
nova_zips = loudoun_zips | fairfax_zips | pw_zips

hpi_nova = hpi[hpi["zip5"].isin(nova_zips)].copy()
hpi_nova["county"] = hpi_nova["zip5"].apply(
    lambda z: "Loudoun" if z in loudoun_zips else "Fairfax" if z in fairfax_zips else "PW"
)
print(f"  FHFA ZIP5: {len(hpi_nova):,} rows, {hpi_nova['zip5'].nunique()} ZIPs, "
      f"{hpi_nova['year'].min()}-{hpi_nova['year'].max()}")

# --- NVRC Data Centers ---
with open(RAW / "nvrc_data_centers.geojson") as f:
    nvrc = json.load(f)
dc_gdf = gpd.GeoDataFrame.from_features(nvrc["features"], crs="EPSG:4326")
dc_gdf = dc_gdf[dc_gdf.geometry.notnull() & ~dc_gdf.geometry.is_empty]
dc_gdf["county"] = dc_gdf["Juisdiction"].str.replace(" County", "").str.strip()
dc_gdf.loc[dc_gdf["county"] == "Prince William", "county"] = "PW"
print(f"  NVRC DCs: {len(dc_gdf)} ({dc_gdf['county'].value_counts().to_dict()})")

# --- ZIP code boundaries (approximate from TIGER ZCTA) ---
# Use ZIP centroids derived from known locations
# For a proper analysis we'd use ZCTA shapefiles, but we can approximate
# with the FHFA data + DC locations
zip_centroids = {
    # Loudoun ZIPs (approximate centroids)
    "20105": (38.86, -77.55), "20117": (39.02, -77.56), "20120": (38.92, -77.50),
    "20129": (39.09, -77.63), "20130": (39.02, -77.65), "20132": (39.10, -77.72),
    "20134": (39.13, -77.65), "20135": (39.12, -77.55), "20141": (39.07, -77.60),
    "20147": (39.04, -77.46), "20148": (39.02, -77.50), "20152": (38.93, -77.52),
    "20158": (39.14, -77.66), "20164": (39.03, -77.40), "20165": (39.06, -77.40),
    "20166": (38.97, -77.44), "20175": (39.10, -77.56), "20176": (39.06, -77.53),
    "20180": (39.16, -77.67),
    # Fairfax ZIPs
    "22003": (38.83, -77.22), "22015": (38.78, -77.28), "22027": (38.90, -77.21),
    "22030": (38.85, -77.34), "22031": (38.86, -77.29), "22032": (38.82, -77.29),
    "22033": (38.87, -77.38), "22039": (38.75, -77.30), "22041": (38.85, -77.14),
    "22042": (38.87, -77.19), "22043": (38.90, -77.18), "22044": (38.85, -77.16),
    "22046": (38.88, -77.17), "22060": (38.72, -77.17), "22066": (39.00, -77.24),
    "22079": (38.68, -77.24), "22101": (38.94, -77.14), "22102": (38.95, -77.20),
    "22124": (38.90, -77.30), "22150": (38.77, -77.17), "22151": (38.79, -77.20),
    "22152": (38.76, -77.22), "22153": (38.74, -77.25), "22180": (38.88, -77.25),
    "22181": (38.89, -77.26), "22182": (38.93, -77.26),
    # PW ZIPs
    "20109": (38.78, -77.53), "20110": (38.75, -77.47), "20111": (38.77, -77.44),
    "20112": (38.67, -77.45), "20136": (38.74, -77.57), "20137": (38.82, -77.65),
    "20143": (38.80, -77.58), "20155": (38.81, -77.63), "20169": (38.85, -77.60),
    "20181": (38.63, -77.31),
}

# --- Compute DC count per ZIP ---
dc_proj = dc_gdf.to_crs(epsg=32618)
dc_coords = np.array([(g.x, g.y) for g in dc_proj.geometry.centroid])
dc_tree = cKDTree(dc_coords)

zip_data = []
for z, (lat, lon) in zip_centroids.items():
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=32618)
    coord = np.array([[pt.iloc[0].x, pt.iloc[0].y]])

    dist_m, _ = dc_tree.query(coord, k=1)
    n_1mi = len(dc_tree.query_ball_point(coord[0], r=1609.34))
    n_2mi = len(dc_tree.query_ball_point(coord[0], r=3218.69))
    n_5mi = len(dc_tree.query_ball_point(coord[0], r=8046.72))

    zip_data.append({
        "zip5": z, "lat": lat, "lon": lon,
        "dist_dc_mi": dist_m[0] / 1609.34,
        "dc_1mi": n_1mi, "dc_2mi": n_2mi, "dc_5mi": n_5mi,
        "county": "Loudoun" if z in loudoun_zips else "Fairfax" if z in fairfax_zips else "PW",
    })

zip_info = pd.DataFrame(zip_data)
print(f"  ZIP centroids: {len(zip_info)} ZIPs")
print(f"  DC count (2mi) range: {zip_info['dc_2mi'].min()}-{zip_info['dc_2mi'].max()}")

# --- Realtor.com ---
realtor = pd.read_csv(RAW / "realtor_nova.csv", dtype={"postal_code": str})
realtor["postal_code"] = realtor["postal_code"].str.zfill(5)
realtor["date"] = pd.to_datetime(realtor["month_date_yyyymm"], format="%Y%m")
realtor["year"] = realtor["date"].dt.year
print(f"  Realtor.com: {len(realtor):,} rows, {realtor['postal_code'].nunique()} ZIPs, "
      f"{realtor['year'].min()}-{realtor['year'].max()}")

# --- Build Panel ---
panel = hpi_nova.merge(zip_info, on="zip5", how="inner", suffixes=("", "_zi"))
panel = panel[(panel["year"] >= 2000) & (panel["year"] <= 2025)].copy()
panel["ln_hpi"] = np.log(panel["hpi"].clip(lower=1))
panel["post"] = (panel["year"] >= TREATMENT_YEAR).astype(int)
panel["high_dc"] = (panel["dc_2mi"] >= 10).astype(int)
panel["txp"] = panel["high_dc"] * panel["post"]
panel["dc_x_post"] = panel["dc_2mi"] * panel["post"]
panel["rel_year"] = panel["year"] - TREATMENT_YEAR
panel["post_2022"] = (panel["year"] >= 2022).astype(int)

print(f"\n  Panel: {len(panel):,} obs, {panel['zip5'].nunique()} ZIPs")
print(f"  High-DC ZIPs (10+ within 2mi): {panel[panel['high_dc']==1]['zip5'].nunique()}")
print(f"  Low-DC ZIPs: {panel[panel['high_dc']==0]['zip5'].nunique()}")
print(f"  By county: {panel.groupby('county_zi')['zip5'].nunique().to_dict()}")

# --- Helper ---
def run_did(df, treat_var="txp", extra_vars=None, label=""):
    Y = df["ln_hpi"].values
    X_vars = [treat_var] + (extra_vars or [])
    X_treat = df[[v for v in X_vars if v in df.columns]].values
    var_names = [v for v in X_vars if v in df.columns]

    zip_d = pd.get_dummies(df["zip5"], drop_first=True)
    yr_d = pd.get_dummies(df["year"], drop_first=True)
    X = sm.add_constant(np.hstack([X_treat, zip_d.values, yr_d.values]))
    names = ["const"] + var_names + list(zip_d.columns) + list(yr_d.columns)

    m = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": df["zip5"].values})
    p = pd.Series(m.params, index=names)
    se = pd.Series(m.bse, index=names)
    ci = pd.DataFrame(m.conf_int(), index=names, columns=["lo", "hi"])
    pv = pd.Series(m.pvalues, index=names)

    return {
        "coef": p[treat_var], "se": se[treat_var],
        "ci_lo": ci.loc[treat_var, "lo"], "ci_hi": ci.loc[treat_var, "hi"],
        "pval": pv[treat_var], "pct": (np.exp(p[treat_var]) - 1) * 100,
        "n": int(m.nobs), "r2": m.rsquared, "label": label, "model": m,
    }

# =====================================================================
# FIGURE 1: DC DENSITY MAP (ZIP-level)
# =====================================================================
print("\n[Fig 1] DC density map...")

fig, ax = plt.subplots(figsize=(12, 8))
sc = ax.scatter(zip_info["lon"], zip_info["lat"], c=zip_info["dc_2mi"],
                cmap="YlOrRd", s=zip_info["dc_2mi"].clip(lower=2) * 8 + 30,
                edgecolors="black", linewidth=0.5, alpha=0.85, zorder=3)
dc_gdf.plot(ax=ax, color="black", markersize=3, alpha=0.3, zorder=2)

for _, row in zip_info.iterrows():
    ax.annotate(row["zip5"], (row["lon"], row["lat"]), fontsize=5,
                ha="center", va="bottom", alpha=0.6)

plt.colorbar(sc, ax=ax, label="Data Centers within 2 miles", shrink=0.7)
ax.set_title("Northern Virginia ZIP Codes by Data Center Density\n"
             "Bubble size = DC count within 2mi | Dots = individual DCs", fontsize=13)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.tight_layout()
fig.savefig(OUT / "fig1_dc_density_map.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUT / 'fig1_dc_density_map.png'}")

# =====================================================================
# FIGURE 2: PARALLEL TRENDS BY COUNTY
# =====================================================================
print("[Fig 2] Parallel trends by county...")

fig, ax = plt.subplots(figsize=(13, 6))
colors = {"Loudoun": "#d73027", "Fairfax": "#4575b4", "PW": "#fc8d59"}
for county, color in colors.items():
    trend = panel[panel["county_zi"] == county].groupby("year")["hpi"].mean()
    ax.plot(trend.index, trend.values, "o-", color=color, lw=2, ms=4, label=county)

ax.axvline(x=TREATMENT_YEAR, color="gray", ls="--", alpha=0.7, label="DC boom (2012)")
ax.axvline(x=2022, color="green", ls=":", alpha=0.5, label="Silver Line (2022)")
ax.set_xlabel("Year"); ax.set_ylabel("House Price Index")
ax.set_title("House Price Trends by County — Northern Virginia\n"
             "FHFA ZIP5 HPI, 2000-2025", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(OUT / "fig2_county_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUT / 'fig2_county_trends.png'}")

# =====================================================================
# FIGURE 3: HIGH-DC vs LOW-DC ZIP TRENDS
# =====================================================================
print("[Fig 3] High-DC vs Low-DC trends...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: Binary split
ax = axes[0]
for label, val, color in [("High DC (10+ within 2mi)", 1, "#d73027"),
                            ("Low DC (<10 within 2mi)", 0, "#4575b4")]:
    trend = panel[panel["high_dc"] == val].groupby("year")["hpi"].mean()
    ax.plot(trend.index, trend.values, "o-", color=color, lw=2, ms=4, label=label)
ax.axvline(x=TREATMENT_YEAR, color="gray", ls="--", alpha=0.7)
ax.set_xlabel("Year"); ax.set_ylabel("HPI")
ax.set_title("A) Binary Treatment (10+ DCs within 2mi)", fontsize=12)
ax.legend(fontsize=9)

# Panel B: Tercile split
ax = axes[1]
dc_first = panel.groupby("zip5")["dc_2mi"].first()
panel["dc_group"] = panel["zip5"].map(
    lambda z: "0 DCs" if dc_first.get(z, 0) == 0
    else "1-9 DCs" if dc_first.get(z, 0) < 10
    else "10+ DCs"
)
tercile_colors = {"0 DCs": "#4575b4", "1-9 DCs": "#fee090", "10+ DCs": "#d73027"}
for tercile in ["0 DCs", "1-9 DCs", "10+ DCs"]:
    subset = panel[panel["dc_group"] == tercile]
    if not subset.empty:
        trend = subset.groupby("year")["hpi"].mean()
        ax.plot(trend.index, trend.values, "o-", color=tercile_colors[tercile],
                lw=2, ms=4, label=tercile)
ax.axvline(x=TREATMENT_YEAR, color="gray", ls="--", alpha=0.7)
ax.set_xlabel("Year"); ax.set_ylabel("HPI")
ax.set_title("B) DC Density Terciles", fontsize=12)
ax.legend(fontsize=9)

fig.suptitle("Figure 3: House Price Trends by Data Center Exposure", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(OUT / "fig3_dc_exposure_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUT / 'fig3_dc_exposure_trends.png'}")

# =====================================================================
# FIGURE 4: DiD REGRESSION TABLE
# =====================================================================
print("\n[Fig 4] Running DiD regressions...")

specs = [
    run_did(panel, label="Binary (10+ DCs), all NoVA"),
    run_did(panel, treat_var="dc_x_post", label="Continuous (DC count x Post)"),
    run_did(panel[panel["county_zi"] == "Loudoun"], label="Loudoun only"),
    run_did(panel[panel["county_zi"] != "Loudoun"], label="Fairfax + PW only"),
    run_did(panel[panel["year"] <= 2021], label="Pre-Silver Line (2000-2021)"),
]

# Cross-county DiD: use Fairfax (few DCs) as control for Loudoun (many DCs)
county_panel = panel.copy()
county_panel["loudoun"] = (county_panel["county_zi"] == "Loudoun").astype(int)
county_panel["loud_x_post"] = county_panel["loudoun"] * county_panel["post"]
specs.append(run_did(county_panel, treat_var="loud_x_post",
                     label="Loudoun vs Fairfax+PW (county DiD)"))

print("\n" + "=" * 70)
print("TABLE: ZIP-LEVEL DiD RESULTS")
print("=" * 70)
for s in specs:
    stars = "***" if s["pval"] < 0.01 else "**" if s["pval"] < 0.05 else "*" if s["pval"] < 0.1 else ""
    print(f"  {s['label']}")
    print(f"    coef={s['coef']:.4f}{stars} (SE={s['se']:.4f})  "
          f"[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]  effect={s['pct']:+.1f}%  "
          f"N={s['n']}  R²={s['r2']:.3f}")

# Forest plot
fig, ax = plt.subplots(figsize=(11, 6))
for i, s in enumerate(specs):
    color = "#d73027" if s["pval"] < 0.05 else "#fc8d59" if s["pval"] < 0.1 else "#999"
    ax.errorbar(s["coef"], i, xerr=[[s["coef"] - s["ci_lo"]], [s["ci_hi"] - s["coef"]]],
                fmt="o", capsize=5, ms=8, lw=2, color=color, ecolor=color)
    ax.annotate(f"{s['pct']:+.1f}% {'***' if s['pval']<0.01 else '**' if s['pval']<0.05 else '*' if s['pval']<0.1 else ''}  (N={s['n']})",
                xy=(s["ci_hi"] + 0.005, i), fontsize=8, va="center", color="#555")
ax.axvline(x=0, color="gray", ls="--", alpha=0.7)
ax.set_yticks(range(len(specs)))
ax.set_yticklabels([s["label"] for s in specs], fontsize=10)
ax.set_xlabel("DiD Coefficient on ln(HPI)", fontsize=12)
ax.set_title("Figure 4: ZIP-Level DiD — DC Proximity Effect\n"
             "ZIP + Year FE | Clustered SEs | 52 NoVA ZIPs", fontsize=13)
plt.tight_layout()
fig.savefig(OUT / "fig4_zip_did_forest.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUT / 'fig4_zip_did_forest.png'}")

# =====================================================================
# FIGURE 5: EVENT STUDY (ZIP-level)
# =====================================================================
print("\n[Fig 5] Event study...")

dummies, dummy_map = [], {}
es_panel = panel.copy()
for t in range(-12, 14):
    if t == -1: continue
    col = f"rtm{abs(t)}" if t < 0 else f"rtp{t}"
    es_panel[col] = ((es_panel["rel_year"] == t) & (es_panel["high_dc"] == 1)).astype(int)
    dummies.append(col); dummy_map[col] = t

Y = es_panel["ln_hpi"].values
X_ev = es_panel[dummies].values
zip_d = pd.get_dummies(es_panel["zip5"], drop_first=True)
yr_d = pd.get_dummies(es_panel["year"], drop_first=True)
X = sm.add_constant(np.hstack([X_ev, zip_d.values, yr_d.values]))
names = ["const"] + dummies + list(zip_d.columns) + list(yr_d.columns)
em = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": es_panel["zip5"].values})
ep = pd.Series(em.params, index=names)
eci = pd.DataFrame(em.conf_int(), index=names, columns=["lo", "hi"])

periods, coefs, clo, chi = [-1], [0], [0], [0]
for col in dummies:
    periods.append(dummy_map[col])
    coefs.append(ep[col]); clo.append(eci.loc[col, "lo"]); chi.append(eci.loc[col, "hi"])
order = np.argsort(periods)
periods = np.array(periods)[order]; coefs = np.array(coefs)[order]
clo = np.array(clo)[order]; chi = np.array(chi)[order]

fig, ax = plt.subplots(figsize=(14, 7))
ax.fill_between(periods, clo, chi, alpha=0.12, color="#d73027")
ax.plot(periods, coefs, "o-", color="#d73027", ms=5, lw=1.5)
ax.axhline(y=0, color="gray", ls="--", alpha=0.7)
ax.axvline(x=-0.5, color="black", ls="--", lw=2, alpha=0.5)
ax.axvline(x=10, color="green", ls=":", lw=1.5, alpha=0.7)
ax.annotate("DC Boom (2012)", xy=(-0.5, max(coefs)*0.9), fontsize=9, ha="center", alpha=0.7)
ax.annotate("Silver Line (2022)", xy=(10, max(coefs)*0.9), fontsize=9, ha="center", color="green")
ax.set_xlabel("Years Relative to 2012", fontsize=12)
ax.set_ylabel("Effect on ln(HPI)", fontsize=12)
ax.set_title("Figure 5: Event Study — ZIP-Level, 52 NoVA ZIPs\n"
             "Treatment: 10+ DCs within 2mi | ZIP + Year FE", fontsize=13)
plt.tight_layout()
fig.savefig(OUT / "fig5_zip_event_study.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUT / 'fig5_zip_event_study.png'}")

# =====================================================================
# FIGURE 6: DOSE-RESPONSE (DC count vs effect)
# =====================================================================
print("[Fig 6] Dose-response...")

dose_results = []
# Run DiD for each DC-density bin vs ZIPs with 0 DCs
control = panel[panel["dc_2mi"] == 0]
for lo, hi, label in [(1, 5, "1-5"), (6, 15, "6-15"), (16, 40, "16-40"), (41, 200, "41+")]:
    treated = panel[(panel["dc_2mi"] >= lo) & (panel["dc_2mi"] <= hi)]
    if treated["zip5"].nunique() < 2: continue
    combined = pd.concat([
        treated.assign(treat_bin=1, txp_bin=lambda x: x["post"]),
        control.assign(treat_bin=0, txp_bin=0),
    ])
    try:
        r = run_did(combined, treat_var="txp_bin", label=label)
        dose_results.append({"bin": label, "mid": (lo+hi)/2, **r})
    except Exception:
        continue

dose_df = pd.DataFrame(dose_results)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#fee090", "#fc8d59", "#d73027", "#a50026"]
for i, (_, row) in enumerate(dose_df.iterrows()):
    ax.bar(i, row["coef"], yerr=[[row["coef"]-row["ci_lo"]], [row["ci_hi"]-row["coef"]]],
           capsize=8, color=colors[i % len(colors)], edgecolor="white", lw=2, width=0.6)
    stars = "***" if row["pval"]<0.01 else "**" if row["pval"]<0.05 else "*" if row["pval"]<0.1 else ""
    ax.text(i, row["ci_hi"] + 0.005, f"{row['pct']:+.1f}%{stars}",
            ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(range(len(dose_df)))
ax.set_xticklabels([f"{r['bin']} DCs" for _, r in dose_df.iterrows()], fontsize=11)
ax.axhline(y=0, color="gray", ls="--", alpha=0.7)
ax.set_ylabel("DiD Coefficient on ln(HPI)", fontsize=12)
ax.set_xlabel("Number of DCs within 2 miles of ZIP centroid", fontsize=12)
ax.set_title("Figure 6: Dose-Response — More DCs = Bigger Effect?\n"
             "Control: ZIPs with 0 DCs within 2mi | ZIP + Year FE", fontsize=13)
plt.tight_layout()
fig.savefig(OUT / "fig6_dose_response.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUT / 'fig6_dose_response.png'}")

# =====================================================================
# FIGURE 7: REALTOR.COM — LISTING PRICES & MARKET DYNAMICS
# =====================================================================
print("[Fig 7] Realtor.com analysis...")

realtor_m = realtor.merge(zip_info[["zip5", "dc_2mi", "county"]],
                          left_on="postal_code", right_on="zip5", how="inner")
realtor_m["high_dc"] = (realtor_m["dc_2mi"] >= 10).astype(int)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# A: Median listing price
ax = axes[0, 0]
for label, val, color in [("High DC", 1, "#d73027"), ("Low DC", 0, "#4575b4")]:
    sub = realtor_m[realtor_m["high_dc"] == val]
    trend = sub.groupby("date")["median_listing_price"].median()
    ax.plot(trend.index, trend.values / 1000, color=color, lw=2, label=label, alpha=0.8)
ax.set_ylabel("Median Listing Price ($K)"); ax.set_title("A) Median Listing Price")
ax.legend(); ax.axvline(x=pd.Timestamp("2022-11-15"), color="green", ls=":", alpha=0.5)

# B: Days on market
ax = axes[0, 1]
for label, val, color in [("High DC", 1, "#d73027"), ("Low DC", 0, "#4575b4")]:
    sub = realtor_m[realtor_m["high_dc"] == val]
    trend = sub.groupby("date")["median_days_on_market"].median()
    ax.plot(trend.index, trend.values, color=color, lw=2, label=label, alpha=0.8)
ax.set_ylabel("Median Days on Market"); ax.set_title("B) Days on Market")
ax.legend()

# C: Active inventory
ax = axes[1, 0]
for label, val, color in [("High DC", 1, "#d73027"), ("Low DC", 0, "#4575b4")]:
    sub = realtor_m[realtor_m["high_dc"] == val]
    trend = sub.groupby("date")["active_listing_count"].median()
    ax.plot(trend.index, trend.values, color=color, lw=2, label=label, alpha=0.8)
ax.set_ylabel("Active Listings (median per ZIP)"); ax.set_title("C) Active Inventory")
ax.legend()

# D: Price per square foot
ax = axes[1, 1]
for label, val, color in [("High DC", 1, "#d73027"), ("Low DC", 0, "#4575b4")]:
    sub = realtor_m[realtor_m["high_dc"] == val]
    trend = sub.groupby("date")["median_listing_price_per_square_foot"].median()
    ax.plot(trend.index, trend.values, color=color, lw=2, label=label, alpha=0.8)
ax.set_ylabel("Median $/sqft"); ax.set_title("D) Price per Square Foot")
ax.legend()

fig.suptitle("Figure 7: Realtor.com Market Dynamics — High-DC vs Low-DC ZIPs\n"
             "Northern Virginia, 2016-2026", fontsize=14)
plt.tight_layout()
fig.savefig(OUT / "fig7_realtor_dynamics.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUT / 'fig7_realtor_dynamics.png'}")

# =====================================================================
# FIGURE 8: SPECIFICATION CURVE (ZIP-level)
# =====================================================================
print("[Fig 8] Specification curve...")

spec_results = []
for sample_name, sdf in [("All NoVA", panel), ("Loudoun", panel[panel["county_zi"]=="Loudoun"]),
                          ("Fairfax+PW", panel[panel["county_zi"]!="Loudoun"])]:
    for dc_thresh in [5, 10, 15, 20]:
        for ty in [2010, 2012, 2014]:
            df = sdf.copy()
            df["treat_v"] = (df["dc_2mi"] >= dc_thresh).astype(int)
            df["post_v"] = (df["year"] >= ty).astype(int)
            df["txp_v"] = df["treat_v"] * df["post_v"]
            nt = df[df["treat_v"]==1]["zip5"].nunique()
            nc = df[df["treat_v"]==0]["zip5"].nunique()
            if nt < 2 or nc < 2: continue
            try:
                r = run_did(df, treat_var="txp_v")
                spec_results.append({"sample": sample_name, "dc_thresh": dc_thresh,
                                     "treat_year": ty, **r})
            except: continue

sdf = pd.DataFrame(spec_results).sort_values("coef").reset_index(drop=True)
print(f"  {len(sdf)} specifications")
print(f"  Range: [{sdf['coef'].min():.4f}, {sdf['coef'].max():.4f}]")
print(f"  Sig at 5%: {(sdf['pval'] < 0.05).sum()}/{len(sdf)}")

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.5], hspace=0.05)
ax_top = fig.add_subplot(gs[0])
colors_s = ["#d73027" if p < 0.05 else "#fc8d59" if p < 0.1 else "#ccc" for p in sdf["pval"]]
ax_top.scatter(range(len(sdf)), sdf["coef"], c=colors_s, s=20, zorder=5)
ax_top.errorbar(range(len(sdf)), sdf["coef"],
                yerr=[sdf["coef"]-sdf["ci_lo"], sdf["ci_hi"]-sdf["coef"]],
                fmt="none", ecolor="#ddd", elinewidth=0.5)
ax_top.axhline(y=0, color="gray", ls="--", alpha=0.7)
ax_top.set_ylabel("DiD Coefficient"); ax_top.set_xticks([])
ax_top.set_title(f"Figure 8: Specification Curve — {len(sdf)} ZIP-Level Specifications\n"
                 "Red=p<0.05 | Orange=p<0.10 | Gray=n.s.", fontsize=13)

ax_bot = fig.add_subplot(gs[1])
y = 0
for var in ["sample", "dc_thresh", "treat_year"]:
    for val in sorted(sdf[var].unique()):
        active = sdf[var] == val
        ax_bot.scatter(sdf.index[active], [y]*active.sum(), s=6, color="#333", alpha=0.7)
        ax_bot.scatter(sdf.index[~active], [y]*(~active).sum(), s=6, color="#eee", alpha=0.3)
        ax_bot.text(-1, y, f"{var}={val}", fontsize=7, ha="right", va="center")
        y += 1
    y += 0.3
ax_bot.set_xlim(-1, len(sdf)); ax_bot.set_ylim(-1, y)
ax_bot.set_yticks([]); ax_bot.invert_yaxis()
ax_bot.set_xlabel("Specification (sorted by estimate)")
plt.tight_layout()
fig.savefig(OUT / "fig8_zip_spec_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUT / 'fig8_zip_spec_curve.png'}")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("ZIP-LEVEL ANALYSIS COMPLETE")
print("=" * 70)
baseline = specs[0]
continuous = specs[1]
county_did = specs[5]
pre_sl = specs[4]
stars = lambda p: "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
print(f"""
BINARY DiD (10+ DCs within 2mi):
  Coefficient: {baseline['coef']:.4f}{stars(baseline['pval'])} (SE: {baseline['se']:.4f})
  Effect: {baseline['pct']:+.1f}%  p={baseline['pval']:.4f}  N={baseline['n']}

CONTINUOUS (DC count x Post):
  Coefficient: {continuous['coef']:.4f}{stars(continuous['pval'])} (SE: {continuous['se']:.4f})
  Per-DC effect: {continuous['pct']:+.2f}%  p={continuous['pval']:.4f}

COUNTY DiD (Loudoun vs Fairfax+PW):
  Coefficient: {county_did['coef']:.4f}{stars(county_did['pval'])} (SE: {county_did['se']:.4f})
  Effect: {county_did['pct']:+.1f}%  p={county_did['pval']:.4f}

PRE-SILVER LINE (2000-2021):
  Coefficient: {pre_sl['coef']:.4f}{stars(pre_sl['pval'])} (SE: {pre_sl['se']:.4f})
  Effect: {pre_sl['pct']:+.1f}%  p={pre_sl['pval']:.4f}

SPEC CURVE: {len(sdf)} specs, [{sdf['coef'].min():.4f}, {sdf['coef'].max():.4f}]
  Significant at 5%: {(sdf['pval'] < 0.05).sum()}/{len(sdf)}
""")

print("Figures:", OUT)
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}")
