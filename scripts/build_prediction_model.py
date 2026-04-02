"""
DC Siting Probability Model + Property Value Risk Scores
=========================================================
Predicts which undeveloped/industrial parcels are likely to become data
centers, then combines with the distance-decay coefficients to produce
a property value risk score for every residential parcel.

Output:
  - Trained classifier (features → P(becomes DC))
  - Risk heatmap for all residential parcels
  - Top 50 highest-probability future DC sites
  - Top residential parcels at risk of value decline
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                              roc_curve, precision_recall_curve, average_precision_score)
from sklearn.calibration import CalibratedClassifierCV
import statsmodels.api as sm

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                      "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "output" / "prediction"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DC SITING PREDICTION MODEL")
print("=" * 70)

# =====================================================================
# 1. LOAD & PREPARE FEATURES
# =====================================================================
print("\n[1] Loading data...")

# --- DC parcels (positive class) ---
with open(RAW / "byright_dc_parcels_full.geojson") as f:
    dc_raw = json.load(f)

dc_parcels = []
for feat in dc_raw["features"]:
    p = feat["properties"]
    geom = feat.get("geometry")
    if not geom or p.get("LU_DISPLAY") != "Data Center":
        continue
    coords = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    dc_parcels.append({
        "mcpi": str(p.get("PA_MCPI", "")),
        "acres": p.get("PA_GIS_ACR", 0),
        "zone": p.get("ZO_ZONE", ""),
        "lat": cy, "lon": cx,
        "is_dc": 1,
    })
dc_df = pd.DataFrame(dc_parcels)
print(f"  DC parcels (positive): {len(dc_df)}")

# --- All parcels (build feature set) ---
with open(RAW / "loudoun_parcels_full.geojson") as f:
    all_parcels_raw = json.load(f)

all_parcels = []
for feat in all_parcels_raw["features"]:
    p = feat["properties"]
    geom = feat.get("geometry")
    if not geom:
        continue
    try:
        if geom["type"] == "Polygon":
            coords = geom["coordinates"][0]
        elif geom["type"] == "MultiPolygon":
            coords = geom["coordinates"][0][0]
        else:
            continue
        cx = np.mean([c[0] for c in coords])
        cy = np.mean([c[1] for c in coords])
    except (IndexError, TypeError):
        continue

    all_parcels.append({
        "mcpi": str(p.get("PA_MCPI", "")),
        "type": p.get("PA_TYPE", ""),
        "acres": p.get("PA_LEGAL_ACRE", 0) or 0,
        "sqft": p.get("PA_LEGAL_SQFT", 0) or 0,
        "subd": p.get("PA_SUBD_NAME", ""),
        "lat": cy, "lon": cx,
        "shape_area": p.get("SHAPE_Area", 0) or 0,
    })
all_df = pd.DataFrame(all_parcels)
print(f"  All parcels: {len(all_df):,}")

# --- Zoning overlay ---
with open(RAW / "loudoun_zoning_sample.geojson") as f:
    zoning_raw = json.load(f)
zone_gdf = gpd.GeoDataFrame.from_features(zoning_raw["features"], crs="EPSG:4326")
zone_gdf = zone_gdf[zone_gdf.geometry.notnull()]

# --- NVRC DC locations (for proximity features) ---
with open(RAW / "nvrc_data_centers.geojson") as f:
    nvrc = json.load(f)
nvrc_gdf = gpd.GeoDataFrame.from_features(nvrc["features"], crs="EPSG:4326")
nvrc_gdf = nvrc_gdf[nvrc_gdf.geometry.notnull() & ~nvrc_gdf.geometry.is_empty]
nvrc_proj = nvrc_gdf.to_crs(epsg=32618)
dc_coords_existing = np.array([(g.x, g.y) for g in nvrc_proj.geometry.centroid])
dc_tree = cKDTree(dc_coords_existing)

# =====================================================================
# 2. ENGINEER FEATURES
# =====================================================================
print("\n[2] Engineering features...")

# Label: is this parcel a known DC?
dc_mcpis = set(dc_df["mcpi"].values)
all_df["is_dc"] = all_df["mcpi"].isin(dc_mcpis).astype(int)
print(f"  Labeled DCs in full parcel set: {all_df['is_dc'].sum()}")

# Spatial features
parcel_gdf = gpd.GeoDataFrame(all_df,
    geometry=gpd.points_from_xy(all_df["lon"], all_df["lat"]), crs="EPSG:4326")
parcel_proj = parcel_gdf.to_crs(epsg=32618)
parcel_coords = np.array([(g.x, g.y) for g in parcel_proj.geometry])

# Distance to nearest existing DC
dist_to_dc, _ = dc_tree.query(parcel_coords, k=1)
all_df["dist_to_dc_m"] = dist_to_dc
all_df["dist_to_dc_mi"] = dist_to_dc / 1609.34

# DC count within various radii
for r_mi, r_m in [(0.5, 804.67), (1, 1609.34), (2, 3218.69), (5, 8046.72)]:
    counts = dc_tree.query_ball_point(parcel_coords, r=r_m)
    all_df[f"dc_within_{r_mi}mi"] = [len(c) for c in counts]

# Distance to Dulles Airport (approx centroid)
dulles = np.array([[314000, 4313000]])  # approx UTM coords for IAD
all_df["dist_to_dulles_m"] = np.sqrt(((parcel_coords - dulles) ** 2).sum(axis=1))

# Distance to Route 28 corridor (approx N-S line at lon ~-77.44)
rt28_x = 306000  # approx UTM easting for Route 28
all_df["dist_to_rt28_m"] = np.abs(parcel_coords[:, 0] - rt28_x)

# Parcel size features
all_df["log_acres"] = np.log1p(all_df["acres"].clip(lower=0))
all_df["log_sqft"] = np.log1p(all_df["sqft"].clip(lower=0))
all_df["log_shape_area"] = np.log1p(all_df["shape_area"].clip(lower=0))

# Parcel type dummies
all_df["is_commercial"] = (all_df["type"] == "C").astype(int)
all_df["is_vacant_land"] = (all_df["type"] == "B").astype(int)

# Spatial join to zoning (approximate: nearest zone polygon)
# Use the parcel's zone from DC data if available, otherwise spatial join
# For simplicity, use the nearest zone polygon centroid
zone_proj = zone_gdf.to_crs(epsg=32618)
zone_centroids = np.array([(g.centroid.x, g.centroid.y) for g in zone_proj.geometry])
zone_tree = cKDTree(zone_centroids)
_, zone_idx = zone_tree.query(parcel_coords, k=1)

zone_types = zone_gdf["ZO_ZONE"].values
all_df["nearest_zone"] = [zone_types[i] if i < len(zone_types) else "" for i in zone_idx]

# Industrial zoning flags
industrial_zones = {"IP", "GI", "PDIP", "PDGI", "MRHI"}
all_df["industrial_zone"] = all_df["nearest_zone"].isin(industrial_zones).astype(int)

# Easting/Northing (location matters — DCs cluster in eastern Loudoun)
all_df["easting"] = parcel_coords[:, 0]
all_df["northing"] = parcel_coords[:, 1]

print(f"  Features engineered: {len(all_df.columns)} columns")
print(f"  Industrial-zoned parcels: {all_df['industrial_zone'].sum():,}")

# =====================================================================
# 3. TRAIN MODEL
# =====================================================================
print("\n[3] Training model...")

feature_cols = [
    "log_acres", "log_sqft", "log_shape_area",
    "dist_to_dc_m", "dc_within_0.5mi", "dc_within_1mi", "dc_within_2mi", "dc_within_5mi",
    "dist_to_dulles_m", "dist_to_rt28_m",
    "is_commercial", "is_vacant_land", "industrial_zone",
    "easting", "northing",
]

X = all_df[feature_cols].fillna(0).values
y = all_df["is_dc"].values

print(f"  Samples: {len(X):,} (positive: {y.sum()}, negative: {(1-y).sum():,})")
print(f"  Class ratio: 1:{(1-y).sum()//y.sum()}")

# Gradient Boosting with class weight handling
model = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=20, random_state=42,
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"  Cross-validated AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

# Train on full data
model.fit(X, y)

# Calibrate probabilities
cal_model = CalibratedClassifierCV(model, cv=5, method="isotonic")
cal_model.fit(X, y)

# Predict probabilities
all_df["dc_probability"] = cal_model.predict_proba(X)[:, 1]

print(f"  Probability distribution:")
for thresh in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]:
    n = (all_df["dc_probability"] >= thresh).sum()
    print(f"    P >= {thresh:.0%}: {n:,} parcels")

# =====================================================================
# 4. FEATURE IMPORTANCE
# =====================================================================
print("\n[4] Feature importance...")

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances.to_string())

# =====================================================================
# 5. FIGURE 1: ROC & PR CURVES
# =====================================================================
print("\n[5] Generating figures...")

proba = cal_model.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, proba)
prec, rec, _ = precision_recall_curve(y, proba)
auc_val = roc_auc_score(y, proba)
ap_val = average_precision_score(y, proba)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(fpr, tpr, color="#d73027", lw=2, label=f"AUC = {auc_val:.3f}")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend(fontsize=12)

ax = axes[1]
ax.plot(rec, prec, color="#4575b4", lw=2, label=f"AP = {ap_val:.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve"); ax.legend(fontsize=12)

fig.suptitle("Figure 1: DC Siting Prediction Model Performance\n"
             f"Gradient Boosting | {len(X):,} parcels | 5-fold CV AUC = {auc_scores.mean():.3f}",
             fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(OUT / "fig1_model_performance.png", dpi=150, bbox_inches="tight")
plt.close()

# ===== FIGURE 2: Feature importance =====
fig, ax = plt.subplots(figsize=(10, 6))
importances.plot(kind="barh", ax=ax, color="#4575b4")
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Figure 2: What Predicts Data Center Siting?", fontsize=13)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(OUT / "fig2_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ===== FIGURE 3: Probability heatmap =====
print("  Generating heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# A) All parcels colored by DC probability
ax = axes[0]
# Subsample for plotting speed
plot_df = all_df.sample(min(50000, len(all_df)), random_state=42)
sc = ax.scatter(plot_df["lon"], plot_df["lat"], c=plot_df["dc_probability"],
                cmap="YlOrRd", s=1, alpha=0.5, norm=mcolors.LogNorm(vmin=0.001, vmax=1))
plt.colorbar(sc, ax=ax, label="P(becomes DC)", shrink=0.7)
# Overlay existing DCs
dc_lons, dc_lats = [], []
for feat in dc_raw["features"]:
    if feat["properties"].get("LU_DISPLAY") != "Data Center" or not feat.get("geometry"):
        continue
    try:
        g = feat["geometry"]
        coords = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
        dc_lons.append(np.mean([c[0] for c in coords]))
        dc_lats.append(np.mean([c[1] for c in coords]))
    except (IndexError, TypeError):
        continue
ax.scatter(dc_lons[:100], dc_lats[:100], c="black", s=10, marker="^", alpha=0.7, label="Existing DCs")
ax.set_title("A) DC Probability — All Parcels", fontsize=12)
ax.legend(fontsize=9)

# B) High-probability parcels only (P > 5%)
ax = axes[1]
high_prob = all_df[(all_df["dc_probability"] > 0.05) & (all_df["is_dc"] == 0)]
sc = ax.scatter(high_prob["lon"], high_prob["lat"], c=high_prob["dc_probability"],
                cmap="YlOrRd", s=high_prob["dc_probability"] * 200 + 5,
                alpha=0.7, edgecolors="black", linewidth=0.3)
plt.colorbar(sc, ax=ax, label="P(becomes DC)", shrink=0.7)
ax.scatter(dc_lons[:100], dc_lats[:100], c="blue", s=10, marker="^", alpha=0.5, label="Existing DCs")
ax.set_title(f"B) High-Probability Non-DC Parcels (P>5%) — {len(high_prob):,} parcels", fontsize=12)
ax.legend(fontsize=9)

fig.suptitle("Figure 3: Data Center Siting Probability Heatmap — Loudoun County", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(OUT / "fig3_probability_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# =====================================================================
# 6. PROPERTY VALUE RISK SCORES
# =====================================================================
print("\n[6] Computing property value risk scores...")

# Distance-decay coefficients from property-level analysis
# 0-0.5mi: +17.8%, 0.5-1mi: -6.4%, 1-2mi: -17.5%, 2-3mi: -0.4%
DECAY = {0.5: 0.178, 1.0: -0.064, 2.0: -0.175, 3.0: -0.004}

# For each residential parcel, compute expected value impact
# = sum over all high-prob parcels of: P(becomes DC) * impact_at_distance
residential = all_df[all_df["type"] == "P"].copy()  # P = private/residential
high_prob_candidates = all_df[(all_df["dc_probability"] > 0.03) & (all_df["is_dc"] == 0)].copy()

print(f"  Residential parcels: {len(residential):,}")
print(f"  High-prob DC candidates (P>3%): {len(high_prob_candidates):,}")

# Build KD-tree of candidate DC sites
cand_coords = np.array(list(zip(
    high_prob_candidates["easting"].values,
    high_prob_candidates["northing"].values
)))
cand_probs = high_prob_candidates["dc_probability"].values

# For each residential parcel, compute risk
res_coords = np.array(list(zip(residential["easting"].values, residential["northing"].values)))

# Vectorized: for each residential parcel, find candidates within 3mi
cand_tree = cKDTree(cand_coords)
risk_scores = np.zeros(len(residential))

for i in range(len(residential)):
    # Find all candidate DCs within 3 miles
    nearby_idx = cand_tree.query_ball_point(res_coords[i], r=3 * 1609.34)
    if not nearby_idx:
        continue

    total_risk = 0
    for j in nearby_idx:
        dist_mi = np.sqrt(((res_coords[i] - cand_coords[j]) ** 2).sum()) / 1609.34
        prob = cand_probs[j]

        # Apply distance decay
        if dist_mi < 0.5:
            impact = DECAY[0.5]
        elif dist_mi < 1.0:
            impact = DECAY[1.0]
        elif dist_mi < 2.0:
            impact = DECAY[2.0]
        elif dist_mi < 3.0:
            impact = DECAY[3.0]
        else:
            continue

        total_risk += prob * impact

    risk_scores[i] = total_risk

residential["risk_score"] = risk_scores
residential["risk_pct"] = residential["risk_score"] * 100

print(f"\n  Risk score distribution:")
print(f"    Min:  {residential['risk_pct'].min():.2f}%")
print(f"    Max:  {residential['risk_pct'].max():.2f}%")
print(f"    Mean: {residential['risk_pct'].mean():.3f}%")
print(f"    Parcels with negative risk (potential decline): "
      f"{(residential['risk_pct'] < -0.1).sum():,}")
print(f"    Parcels with positive risk (potential gain): "
      f"{(residential['risk_pct'] > 0.1).sum():,}")

# ===== FIGURE 4: Risk heatmap =====
fig, ax = plt.subplots(figsize=(14, 10))
# Plot all residential parcels, colored by risk
plot_res = residential.sample(min(60000, len(residential)), random_state=42)
vmax = max(abs(plot_res["risk_pct"].quantile(0.01)), plot_res["risk_pct"].quantile(0.99))
sc = ax.scatter(plot_res["lon"], plot_res["lat"], c=plot_res["risk_pct"],
                cmap="RdYlGn_r", s=1, alpha=0.5,
                vmin=-vmax, vmax=vmax)
plt.colorbar(sc, ax=ax, label="Expected Value Impact (%)\nNegative = at risk | Positive = may benefit",
             shrink=0.7)
ax.scatter(dc_lons[:100], dc_lats[:100], c="black", s=15, marker="^", alpha=0.7, zorder=5)
ax.set_title("Figure 4: Residential Property Value Risk from Future Data Centers\n"
             "Loudoun County | Based on DC siting model + distance-decay coefficients",
             fontsize=13)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_axis_off()
plt.tight_layout()
fig.savefig(OUT / "fig4_risk_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ===== FIGURE 5: Top at-risk areas =====
print("\n  Top 20 most at-risk residential parcels:")
at_risk = residential.nsmallest(20, "risk_pct")
for _, row in at_risk.iterrows():
    subd = row.get("subd", "N/A")
    print(f"    MCPI={row['mcpi']}  risk={row['risk_pct']:.2f}%  "
          f"acres={row['acres']:.2f}  subd={subd}")

# Top 20 highest-probability future DC sites
print("\n  Top 20 predicted future DC sites:")
future_dc = all_df[(all_df["is_dc"] == 0) & (all_df["dc_probability"] > 0.01)]\
    .nlargest(20, "dc_probability")
for _, row in future_dc.iterrows():
    print(f"    MCPI={row['mcpi']}  P={row['dc_probability']:.1%}  "
          f"acres={row['acres']:.1f}  zone={row['nearest_zone']}  "
          f"dist_to_dc={row['dist_to_dc_mi']:.1f}mi")

# ===== FIGURE 6: Risk distribution histogram =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
nonzero_risk = residential[residential["risk_pct"].abs() > 0.01]["risk_pct"]
ax.hist(nonzero_risk, bins=80, color="#4575b4", edgecolor="white", alpha=0.8)
ax.axvline(x=0, color="red", ls="--", alpha=0.7)
ax.set_xlabel("Expected Value Impact (%)")
ax.set_ylabel("Number of Parcels")
ax.set_title(f"A) Risk Distribution ({len(nonzero_risk):,} affected parcels)")

ax = axes[1]
top50 = all_df[all_df["is_dc"] == 0].nlargest(50, "dc_probability")
ax.barh(range(min(25, len(top50))),
        top50["dc_probability"].values[:25],
        color="#d73027", edgecolor="white")
ax.set_yticks(range(min(25, len(top50))))
ax.set_yticklabels([f"{row['mcpi']} ({row['acres']:.0f}ac)"
                    for _, row in top50.head(25).iterrows()], fontsize=7)
ax.set_xlabel("P(becomes Data Center)")
ax.set_title("B) Top 25 Predicted Future DC Sites")
ax.invert_yaxis()

fig.suptitle("Figure 6: Risk Scores & Predicted DC Sites", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(OUT / "fig6_risk_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ===== Save outputs =====
residential[["mcpi", "lat", "lon", "acres", "subd", "risk_score", "risk_pct",
             "dist_to_dc_mi", "dc_within_1mi", "dc_within_2mi"]]\
    .to_csv(OUT / "residential_risk_scores.csv", index=False)

all_df[all_df["is_dc"] == 0][["mcpi", "lat", "lon", "acres", "nearest_zone",
       "dc_probability", "industrial_zone", "dist_to_dc_mi"]]\
    .to_csv(OUT / "parcel_dc_probabilities.csv", index=False)

print(f"\n{'='*70}")
print("MODEL COMPLETE")
print(f"{'='*70}")
print(f"""
MODEL PERFORMANCE:
  5-fold CV AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}
  Full-data AUC: {auc_val:.3f}
  Average Precision: {ap_val:.3f}

TOP FEATURES:
{importances.head(5).to_string()}

RISK SUMMARY:
  Residential parcels scored: {len(residential):,}
  At-risk (negative impact > 0.1%): {(residential['risk_pct'] < -0.1).sum():,}
  May benefit (positive impact > 0.1%): {(residential['risk_pct'] > 0.1).sum():,}

  Predicted future DC sites (P > 10%): {(all_df['dc_probability'] > 0.1).sum() - all_df['is_dc'].sum()}
  Predicted future DC sites (P > 25%): {(all_df['dc_probability'] > 0.25).sum() - all_df['is_dc'].sum()}

OUTPUT FILES:
  {OUT / 'residential_risk_scores.csv'}
  {OUT / 'parcel_dc_probabilities.csv'}
""")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}")
