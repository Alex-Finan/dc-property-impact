# DC Property Impact

**Quantifying the causal effect of data center construction on nearby property values using difference-in-differences analysis.**

---

## Problem

Data centers are the fastest-growing category of US commercial construction ($77.7B in 2025 starts), yet **no rigorous causal study exists** on their impact on nearby residential property values. The only academic work (GMU 2025) is cross-sectional and cannot establish causation. Meanwhile:

- **$64B+ in projects** have been blocked or delayed by community opposition citing property value concerns
- **142+ activist groups** across 24 states are fighting data center construction
- Neither developers nor communities have credible, causal evidence to support their claims

This project fills that gap with academic-quality causal inference, starting with Loudoun County, VA -- the world's largest data center market (199 operational DCs, 70% of global internet traffic).

## Methodology

### Primary: Hedonic Difference-in-Differences

```
ln(P_it) = a + b(Treat_i x Post_t) + g*X_it + d_i + t_t + e_it
```

- **Treatment**: Properties within distance rings (0-1 mi, 1-2 mi, 2-3 mi) of a data center site
- **Control**: Properties 3-5 mi away (buffer zone excluded) matched on observables
- **Event**: Data center announcement/permitting date
- **Coefficient b**: Causal % change in property value attributable to DC proximity

### Validation: Event Study Design

Full dynamic specification with relative-time dummies to:
1. Test the **parallel trends assumption** (pre-treatment coefficients should be ~zero)
2. Trace the **dynamic treatment effect** (anticipation, immediate impact, persistence)

### Supplementary: Synthetic Control

At the ZIP/census-tract level, construct weighted combinations of untreated areas that best approximate the treated area's pre-treatment trajectory. Provides compelling aggregate-level visualization and a complementary estimate.

### Robustness Battery

- Multiple control group definitions
- Varying distance thresholds (0.5, 1, 2, 3 mi)
- Placebo tests (random locations, backdated treatment)
- Staggered DiD estimators (Callaway & Sant'Anna 2021, Sun & Abraham 2021)
- Bacon decomposition for TWFE diagnostics
- Conley spatial HAC standard errors
- Moran's I on residuals

## Data Sources

### Property Transactions
| Source | Coverage | Access | Notes |
|--------|----------|--------|-------|
| Loudoun County Real Property Sales Reports | 2013-present | Free (Excel) | Official county assessor data |
| FHFA House Price Index | 1970s-present, tract-level | Free (CSV) | Repeat-sales index, good for DiD |
| Zillow ZTRAX via ICPSR | 1940-2020, parcel-level | Academic (DUA) | 400M+ records, 2,750 counties |
| BrightMLS | Current, property-level | Research partnership | Used by GMU study |

### Data Center Locations & Timelines
| Source | Coverage | Access |
|--------|----------|--------|
| NVRC Data Center Map | 250+ NoVA locations | Free (request GIS) |
| Loudoun County Building Permits | 2015-present | Free (Excel) |
| Shovels.ai | 2,000+ jurisdictions | Paid API |
| Cleanview.co Project Tracker | 550+ planned US DCs | Free web |

### Geography & Controls
| Source | Data |
|--------|------|
| Loudoun GeoHub | Parcels, zoning, all shapefiles |
| TIGER/Line | Census tract boundaries |
| ACS 5-Year | Demographics at tract level |
| VA School Quality Profiles | School ratings by school |
| FBI Crime Data Explorer | Crime by jurisdiction |
| Metro Silver Line opening | Nov 15, 2022 (critical confounder) |

## Key Identification Challenge

The **Metro Silver Line Phase 2** opened in Loudoun County in November 2022, creating a simultaneous positive infrastructure shock. The analysis must carefully separate data center effects from transit effects using:
- Properties far from Metro stations but near DCs (DC-only treatment)
- Properties near Metro stations but far from DCs (Metro-only treatment)
- Interaction terms for properties near both
- Pre-Silver Line data for clean DC-only identification

## Commercial Application

### Three Buyer Segments

**1. DC Developers** ($10-50K/project)
- Community impact assessments for permitting applications
- Evidence for Community Benefit Agreement (CBA) negotiations
- Ammunition against NIMBY opposition with rigorous causal evidence

**2. Municipal Planners** ($5-25K/year SaaS)
- Tax base impact modeling for zoning decisions
- Evidence-based CBA negotiation from the public side
- Fiscal impact projections beyond the current consultant patchwork

**3. Real Estate Investors** ($10-50K/year subscription)
- Identify neighborhoods about to be repriced by DC announcements
- Quantify the distance-decay function for investment targeting
- Early-warning signals from the 2,788-project US pipeline

### Market Context
- No existing product combines DC pipeline tracking + parcel data + causal impact modeling
- Adjacent products (CoStar, CoreLogic, datacenterHawk) cover pieces but not the integrated analysis
- Estimated TAM: $50-200M ARR across all segments

## Project Structure

```
dc-property-impact/
|-- src/
|   |-- data/              # Data acquisition & processing
|   |   |-- property.py    # Property transaction data fetchers
|   |   |-- datacenters.py # DC location & timeline data
|   |   |-- census.py      # ACS / demographic data
|   |   |-- geo.py         # GIS / spatial data processing
|   |   |-- confounders.py # School, crime, transit data
|   |-- analysis/          # Core analysis modules
|   |   |-- did.py         # Difference-in-differences estimation
|   |   |-- event_study.py # Event study specification
|   |   |-- synth.py       # Synthetic control methods
|   |   |-- spatial.py     # Spatial econometrics & diagnostics
|   |   |-- hedonic.py     # Hedonic pricing model
|   |-- visualization/     # Plotting & mapping
|   |   |-- maps.py        # Geographic visualizations
|   |   |-- plots.py       # Event study plots, coefficient plots
|   |-- utils/             # Shared utilities
|       |-- config.py      # Configuration management
|       |-- io.py          # Data I/O helpers
|-- data/
|   |-- raw/               # Untransformed source data
|   |-- processed/         # Analysis-ready datasets
|   |-- external/          # Third-party reference data
|-- docs/                  # Documentation & methodology notes
|-- tests/                 # Unit tests
|-- configs/               # Configuration files
|-- requirements.txt
|-- pyproject.toml
```

## Tech Stack

| Category | Libraries |
|----------|-----------|
| **DiD Estimation** | pyfixest, linearmodels, differences |
| **Synthetic Control** | pysyncon, SyntheticControlMethods, causalimpact |
| **Spatial** | geopandas, shapely, pysal, h3, folium |
| **Data** | pandas, numpy, requests |
| **Visualization** | matplotlib, seaborn, plotly, contextily |
| **Testing** | pytest |

## Roadmap

### Phase 1: Loudoun County MVP
- [ ] Acquire Loudoun County property sales data (2013-present)
- [ ] Geocode DC locations from NVRC map + building permits
- [ ] Build treatment/control group assignment by distance rings
- [ ] Run baseline hedonic DiD with property + time fixed effects
- [ ] Event study validation of parallel trends
- [ ] Tract-level synthetic control as supplementary analysis

### Phase 2: Robustness & Expansion
- [ ] Add ZTRAX data for deeper historical coverage
- [ ] Staggered DiD with Callaway & Sant'Anna estimator
- [ ] Expand to Prince William County, VA and Fairfax County, VA
- [ ] Commercial vs. residential property analysis

### Phase 3: National Scale & Product
- [ ] Expand to top 10 DC markets (NoVA, Dallas, Phoenix, Chicago, etc.)
- [ ] Build automated pipeline for new DC announcements
- [ ] API/dashboard for buyer segments
- [ ] White paper for academic publication

## Key References

- Waters & Clower (2025). "Data Centers and 2023 Home Sales in Northern Virginia." GMU Center for Regional Analysis.
- Currie, Davis, Greenstone & Walker (2015). "Environmental Health Risks and Housing Values." *AER*.
- Davis (2011). "The Effect of Power Plants on Local Housing Values and Rents." *REStat*.
- Callaway & Sant'Anna (2021). "Difference-in-Differences with Multiple Time Periods." *JoE*.
- Abadie (2021). "Using Synthetic Controls." *JEL*.
- Banzhaf (2021). "Difference-in-Differences Hedonics." *JPE*.

## License

MIT
