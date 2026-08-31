# ============================================================================
# app.py — FastAPI Pan-India Manganese Intelligence Backend (SIH25009 / MOIL)
# Multi-region: MP-Maharashtra (MOIL core), Odisha, Andhra Pradesh, Karnataka, Gujarat
# ============================================================================
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

try:
    import rasterio
    from rasterio.warp import transform as rio_transform
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

RNG = np.random.default_rng(42)
DATA_DIR = "data"
BAND_FILES = {b: os.path.join(DATA_DIR, "sentinel2", f"{b}.jp2") for b in ["B02", "B04", "B11", "B12"]}
DEM_FILE = os.path.join(DATA_DIR, "CartoDEM.tif")
LABELS_FILE = os.path.join(DATA_DIR, "mineral_labels.csv")
OPS_FILE = os.path.join(DATA_DIR, "operations_log.csv")

# ----------------------------------------------------------------------------------
# PAN-INDIA MANGANESE BELT REGISTRY
# ----------------------------------------------------------------------------------
REGIONS = {
    "MP_MH": {
        "name": "Central Belt (MP-Maharashtra, MOIL Core)",
        "bbox": {"min_lat": 20.5, "max_lat": 22.0, "min_lon": 79.0, "max_lon": 80.5},
        "mines": {"Bharveli": (21.75, 80.10), "Ukwa": (21.85, 80.30), "Tirodi": (21.60, 80.42),
                  "Dongri Buzurg": (21.05, 79.55)},
        "base_target": {"Bharveli": 950, "Ukwa": 700, "Tirodi": 820, "Dongri Buzurg": 610},
        "supports_raster": True, "map_center": [21.4, 79.8], "zoom": 8,
    },
    "ODISHA": {
        "name": "Eastern Belt (Keonjhar & Sundargarh, Odisha)",
        "bbox": {"min_lat": 21.3, "max_lat": 22.3, "min_lon": 84.5, "max_lon": 86.0},
        "mines": {"Keonjhar Mine": (21.85, 85.30), "Sundargarh Mine": (22.05, 84.85)},
        "base_target": {"Keonjhar Mine": 680, "Sundargarh Mine": 540},
        "supports_raster": False, "map_center": [21.9, 85.2], "zoom": 8,
    },
    "AP": {
        "name": "Southern Belt (Vizianagaram & Srikakulam, AP)",
        "bbox": {"min_lat": 18.0, "max_lat": 19.0, "min_lon": 83.0, "max_lon": 84.0},
        "mines": {"Garbham Mine": (18.55, 83.35), "Kakiriguma Mine": (18.75, 83.65)},
        "base_target": {"Garbham Mine": 420, "Kakiriguma Mine": 380},
        "supports_raster": False, "map_center": [18.6, 83.5], "zoom": 8,
    },
    "KARNATAKA": {
        "name": "South-Western Belt (Bellary & Shimoga, Karnataka)",
        "bbox": {"min_lat": 13.5, "max_lat": 14.8, "min_lon": 75.5, "max_lon": 76.8},
        "mines": {"Sandur Mine": (14.05, 76.30), "Shimoga Mine": (13.95, 75.60)},
        "base_target": {"Sandur Mine": 500, "Shimoga Mine": 360},
        "supports_raster": False, "map_center": [14.0, 76.1], "zoom": 8,
    },
    "GUJARAT": {
        "name": "Western Belt (Panchmahal, Gujarat)",
        "bbox": {"min_lat": 22.5, "max_lat": 23.2, "min_lon": 73.3, "max_lon": 74.0},
        "mines": {"Panchmahal Mine": (22.85, 73.65)},
        "base_target": {"Panchmahal Mine": 310},
        "supports_raster": False, "map_center": [22.85, 73.65], "zoom": 9,
    },
}
REGION_CODES = list(REGIONS.keys())

app = FastAPI(title="MOIL Pan-India AI-Geo Backend", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://MOIL.vercel.app",  # Replace with your actual Vercel URL
        "http://localhost:5173",             # Local Vite dev server
        "http://localhost:3000",             # Local React dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------------
# RASTER UTILITIES (used only for MP_MH if local files present)
# ----------------------------------------------------------------------------------
def sample_raster_at_points(path, lons, lats):
    if not RASTERIO_OK or not os.path.exists(path):
        return None
    try:
        with rasterio.open(path) as src:
            if src.crs is not None and src.crs.to_epsg() != 4326:
                xs, ys = rio_transform("EPSG:4326", src.crs, lons, lats)
            else:
                xs, ys = lons, lats
            coords = list(zip(xs, ys))
            vals = np.array([v[0] if v is not None else np.nan for v in src.sample(coords)], dtype=float)
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
            return vals
    except Exception:
        return None

def raster_available():
    return RASTERIO_OK and (any(os.path.exists(p) for p in BAND_FILES.values()) or os.path.exists(DEM_FILE))

# ----------------------------------------------------------------------------------
# SYNTHETIC HOTSPOT FIELD (region-aware, deterministic per region)
# ----------------------------------------------------------------------------------
def hotspot_field(lons, lats, region_code, seed_offset=0):
    mines = list(REGIONS[region_code]["mines"].values())
    dist = np.zeros(len(lons))
    for hy, hx in mines:
        d = np.sqrt((np.array(lats) - hy) ** 2 + (np.array(lons) - hx) ** 2)
        dist += np.exp(-(d ** 2) / (2 * 0.06 ** 2))
    dist = dist / (dist.max() + 1e-9)
    rng = np.random.default_rng(abs(hash(region_code)) % (2**31) + seed_offset)
    return dist, rng

def build_feature_table(points_df, region_code):
    lons, lats = points_df["longitude"].values, points_df["latitude"].values
    n = len(points_df)
    use_real = region_code == "MP_MH" and raster_available()
    band_vals, dem_vals = {}, None
    if use_real:
        for b, path in BAND_FILES.items():
            band_vals[b] = sample_raster_at_points(path, lons, lats)
        dem_vals = sample_raster_at_points(DEM_FILE, lons, lats)

    dist, rng = hotspot_field(lons, lats, region_code, seed_offset=1)
    if not use_real or any(v is None for v in band_vals.values()):
        band_vals["B02"] = np.clip(800 + 400 * (1 - dist) + rng.normal(0, 60, n), 200, 2000)
        band_vals["B04"] = np.clip(900 + 500 * (1 - dist) + rng.normal(0, 70, n), 200, 2200)
        band_vals["B11"] = np.clip(1500 + 900 * dist + rng.normal(0, 100, n), 300, 3500)
        band_vals["B12"] = np.clip(1300 + 500 * dist + rng.normal(0, 90, n), 300, 3000)
    if dem_vals is None:
        dem_vals = np.clip(280 + 300 * dist + rng.normal(0, 30, n), 200, 800)

    feat = pd.DataFrame({
        "latitude": lats,
        "longitude": lons,
        "B02": band_vals["B02"],
        "B04": band_vals["B04"],
        "B11": band_vals["B11"],
        "B12": band_vals["B12"],
        "elevation": dem_vals,
    }).ffill().bfill()

    feat["swir_ratio"] = feat["B11"] / (feat["B12"] + 1e-6)
    feat["ferrous_ratio"] = feat["B11"] / (feat["B04"] + 1e-6)
    return feat

# ----------------------------------------------------------------------------------
# GROUND-TRUTH LABELS (real CSV for MP_MH if present, else synthetic per region)
# ----------------------------------------------------------------------------------
def load_labels_for_region(region_code, n=150):
    if region_code == "MP_MH" and os.path.exists(LABELS_FILE):
        df = pd.read_csv(LABELS_FILE)
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        if {"latitude", "longitude", "has_manganese"}.issubset(df.columns):
            df = df[["latitude", "longitude", "has_manganese"]].dropna()
            df["region"] = region_code
            return df
    bbox = REGIONS[region_code]["bbox"]
    lats = RNG.uniform(bbox["min_lat"], bbox["max_lat"], n)
    lons = RNG.uniform(bbox["min_lon"], bbox["max_lon"], n)
    dist, rng = hotspot_field(lons, lats, region_code)
    prob = np.clip(dist + rng.normal(0, 0.1, n), 0, 1)
    labels = (prob > np.quantile(prob, 0.65)).astype(int)
    return pd.DataFrame({"latitude": lats, "longitude": lons, "has_manganese": labels, "region": region_code})

# ----------------------------------------------------------------------------------
# RESERVE PIPELINE — TRAINED PER REGION, MERGED FOR NATIONAL VIEW
# ----------------------------------------------------------------------------------
RESERVE_FEATS = ["swir_ratio", "ferrous_ratio", "B02", "B04", "elevation"]

def train_region_reserve(region_code):
    labels_df = load_labels_for_region(region_code)
    feats_df = build_feature_table(labels_df, region_code)
    full = pd.concat([feats_df, labels_df[["has_manganese", "region"]].reset_index(drop=True)], axis=1).dropna()
    X, y = full[RESERVE_FEATS], full["has_manganese"]
    model = RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=3, random_state=42)
    model.fit(X, y)
    imp = dict(zip(RESERVE_FEATS, model.feature_importances_.round(4)))

    bbox = REGIONS[region_code]["bbox"]
    n_grid = 30
    glats = np.linspace(bbox["min_lat"], bbox["max_lat"], n_grid)
    glons = np.linspace(bbox["min_lon"], bbox["max_lon"], n_grid)
    gla, glo = np.meshgrid(glats, glons)
    grid_df = pd.DataFrame({"latitude": gla.flatten(), "longitude": glo.flatten()})
    grid_feats = build_feature_table(grid_df, region_code)
    grid_feats["probability"] = model.predict_proba(grid_feats[RESERVE_FEATS])[:, 1]
    grid_feats["region"] = region_code
    grid_feats["region_name"] = REGIONS[region_code]["name"]

    nearest_mine = []
    mines = REGIONS[region_code]["mines"]
    for _, r in grid_feats.iterrows():
        best_m, best_d = None, 1e9
        for mname, (my, mx) in mines.items():
            d = (r.latitude - my) ** 2 + (r.longitude - mx) ** 2
            if d < best_d:
                best_d, best_m = d, mname
        nearest_mine.append(best_m)
    grid_feats["nearest_mine"] = nearest_mine

    reserves_mt = float(grid_feats["probability"].sum() * 0.045)
    return {
        "model": model,
        "importances": imp,
        "labels_df": labels_df,
        "grid_df": grid_feats[["latitude", "longitude", "probability", "region", "region_name", "nearest_mine"]],
        "reserves_mt": reserves_mt,
    }

RESERVE_BY_REGION = {code: train_region_reserve(code) for code in REGION_CODES}

# ----------------------------------------------------------------------------------
# OPERATIONS PIPELINE — PER-REGION SYNTHETIC (OR REAL CSV FOR MP_MH)
# ----------------------------------------------------------------------------------
def load_ops_log():
    if os.path.exists(OPS_FILE):
        df = pd.read_csv(OPS_FILE, parse_dates=["date"])
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        if "region" not in df.columns:
            df["region"] = "MP_MH"
        return df
    dates = pd.date_range(end=pd.Timestamp.today(), periods=365)
    rows = []
    for region_code, cfg in REGIONS.items():
        for pit, bt in cfg["base_target"].items():
            for d in dates:
                monsoon = 1.0 if d.month in [6, 7, 8, 9] else 0.15
                rainfall = max(0, RNG.normal(18 * monsoon, 9))
                dumper_dt = max(0, RNG.normal(2.5 + 3 * monsoon, 1.4))
                excav_dt = max(0, RNG.normal(1.8 + 2 * monsoon, 1.1))
                blast_delay = max(0, RNG.normal(15 + 40 * monsoon, 12))
                target = bt + RNG.normal(0, 20)
                loss = 0.55 * rainfall + 12 * dumper_dt + 9 * excav_dt + 0.35 * blast_delay
                actual = max(0, target - loss + RNG.normal(0, 25))
                rows.append([d, region_code, pit, target, actual, rainfall, excav_dt, dumper_dt, blast_delay])
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "region",
            "pit_id",
            "target_tons",
            "actual_tons",
            "rainfall_mm",
            "excavator_downtime_hrs",
            "dumper_downtime_hrs",
            "blasting_delay_mins",
        ],
    )

OPS_FEATS = ["target_tons", "rainfall_mm", "dumper_downtime_hrs", "excavator_downtime_hrs", "blasting_delay_mins"]

def train_shortfall_pipeline():
    df = load_ops_log()
    df["shortfall_tons"] = df["target_tons"] - df["actual_tons"]
    X, y = df[OPS_FEATS], df["shortfall_tons"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {"model": model, "df": df, "r2": float(r2_score(yte, pred)), "mae": float(mean_absolute_error(yte, pred))}

SHORTFALL = train_shortfall_pipeline()

# ----------------------------------------------------------------------------------
# API SCHEMAS
# ----------------------------------------------------------------------------------
class ShortfallRequest(BaseModel):
    region: str = "MP_MH"
    pit_id: str = "Bharveli"
    rainfall_mm: float
    equipment_downtime: float
    blasting_delay_mins: float = 20.0
    target_tons: float = 850.0

# ----------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------
def region_ops_slice(df, region_code):
    return df if region_code == "ALL" else df[df.region == region_code]

# ----------------------------------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------------------------------
@app.get("/api/regions")
def list_regions():
    return {
        code: {
            "name": cfg["name"],
            "mines": list(cfg["mines"].keys()),
            "map_center": cfg["map_center"],
            "zoom": cfg["zoom"],
        }
        for code, cfg in REGIONS.items()
    }

@app.get("/api/dashboard-kpis")
def dashboard_kpis(region: str = Query("ALL")):
    df = SHORTFALL["df"]
    sub = region_ops_slice(df, region)
    recent = sub.sort_values("date").groupby(["region", "pit_id"]).tail(30)
    monthly_output = float(recent["actual_tons"].sum())
    total_target = recent["target_tons"].sum()
    avg_shortfall_pct = float((recent["target_tons"] - recent["actual_tons"]).sum() / total_target * 100) if total_target else 0.0
    fleet_readiness = float(100 - (recent[["dumper_downtime_hrs", "excavator_downtime_hrs"]].sum(axis=1) / 24 * 100).mean())

    if region == "ALL":
        total_reserves = sum(RESERVE_BY_REGION[c]["reserves_mt"] for c in REGION_CODES)
        region_breakdown = {
            c: {"name": REGIONS[c]["name"], "reserves_mt": round(RESERVE_BY_REGION[c]["reserves_mt"], 2)}
            for c in REGION_CODES
        }
    else:
        total_reserves = RESERVE_BY_REGION[region]["reserves_mt"]
        region_breakdown = {region: {"name": REGIONS[region]["name"], "reserves_mt": round(total_reserves, 2)}}

    return {
        "region": region,
        "region_name": "All India" if region == "ALL" else REGIONS[region]["name"],
        "estimated_reserves_mt": round(total_reserves, 2),
        "projected_monthly_output_tons": round(monthly_output, 1),
        "shortfall_risk_pct": round(avg_shortfall_pct, 2),
        "fleet_availability_pct": round(fleet_readiness, 2),
        "model_r2": round(SHORTFALL["r2"], 3),
        "model_mae": round(SHORTFALL["mae"], 1),
        "region_breakdown": region_breakdown,
        "raster_mode": "REAL_RASTER" if raster_available() else "SYNTHETIC_FALLBACK",
    }

@app.get("/api/reserve-map")
def reserve_map(region: str = Query("ALL")):
    codes = REGION_CODES if region == "ALL" else [region]
    grid_frames, gt_frames, imp_agg = [], [], {}
    for c in codes:
        r = RESERVE_BY_REGION[c]
        grid_frames.append(r["grid_df"])
        gt = r["labels_df"].copy()
        gt["region"] = c
        gt["region_name"] = REGIONS[c]["name"]
        gt_frames.append(gt)
        for k, v in r["importances"].items():
            imp_agg[k] = imp_agg.get(k, 0) + v / len(codes)

    grid_all = pd.concat(grid_frames, ignore_index=True)
    gt_all = pd.concat(gt_frames, ignore_index=True)
    mine_markers = []
    for c in codes:
        for mname, (my, mx) in REGIONS[c]["mines"].items():
            mine_markers.append({
                "mine": mname,
                "region": c,
                "region_name": REGIONS[c]["name"],
                "latitude": my,
                "longitude": mx,
            })

    GRID_RESOLUTION_M = 100
    SEAM_DEPTH_M = 15.0
    ORE_DENSITY_TPM = 4.2

    high_conf_pixels = int((grid_all["probability"] > 0.7).sum())
    area_sq_m = high_conf_pixels * (GRID_RESOLUTION_M ** 2)
    area_sq_km = area_sq_m / 1e6
    area_hectares = area_sq_m / 10000.0

    volume_cubic_m = area_sq_m * SEAM_DEPTH_M
    estimated_reserve_mt = (volume_cubic_m * ORE_DENSITY_TPM) / 1e6

    return {
        "region": region,
        "grid_points": grid_all.round(4).to_dict(orient="records"),
        "ground_truth": gt_all.round(4).to_dict(orient="records"),
        "mine_markers": mine_markers,
        "feature_importances": {k: round(v, 4) for k, v in imp_agg.items()},
        "high_confidence_zones": high_conf_pixels,
        "mineralized_area_km2": round(area_sq_km, 2),
        "mineralized_area_hectares": round(area_hectares, 1),
        "estimated_reserves_mt": round(estimated_reserve_mt, 2),
    }

@app.post("/api/predict-shortfall")
def predict_shortfall(req: ShortfallRequest):
    region = req.region if req.region in REGIONS else "MP_MH"
    X = pd.DataFrame(
        [[
            req.target_tons,
            req.rainfall_mm,
            req.equipment_downtime * 0.55,
            req.equipment_downtime * 0.45,
            req.blasting_delay_mins,
        ]],
        columns=OPS_FEATS,
    )
    shortfall = float(SHORTFALL["model"].predict(X)[0])
    risk_pct = float(np.clip(shortfall / req.target_tons * 100, 0, 100))
    level = "CRITICAL" if risk_pct > 20 else ("ELEVATED" if risk_pct > 10 else "NOMINAL")

    region_mines = list(REGIONS[region]["mines"].keys())
    alt_mine = next((m for m in region_mines if m != req.pit_id), region_mines[0])
    recs = []
    if risk_pct > 10:
        if req.equipment_downtime > 5:
            recs.append(
                f"Machinery Re-deployment: Reallocate dumpers/excavators to {req.pit_id} ({REGIONS[region]['name']}) — combined downtime at {req.equipment_downtime:.1f} hrs is driving losses."
            )
        if req.rainfall_mm > 25:
            recs.append(
                f"Blasting Schedule Optimization: Postpone blasting at {req.pit_id} — rainfall ({req.rainfall_mm:.0f}mm) elevates soil-moisture/safety risk; hold for next dry window."
            )
        if req.blasting_delay_mins > 40:
            recs.append(
                "Blast Sequencing Fix: Compounding blasting delays detected — pre-stage crew/explosives and align shift handoff timing."
            )
        recs.append(
            f"Bench Re-balancing: Shift ~{min(15, int(risk_pct))}% of tonnage target from {req.pit_id} to {alt_mine} within {REGIONS[region]['name']} to offset shortfall."
        )
    else:
        recs.append(f"{req.pit_id} ({REGIONS[region]['name']}) operating within nominal parameters — maintain current cadence.")

    return {
        "region": region,
        "region_name": REGIONS[region]["name"],
        "pit_id": req.pit_id,
        "predicted_shortfall_tons": round(shortfall, 1),
        "predicted_actual_tons": round(req.target_tons - shortfall, 1),
        "risk_pct": round(risk_pct, 2),
        "risk_level": level,
        "recommendations": recs,
    }

@app.get("/")
def root():
    return {"status": "MOIL Pan-India AI-Geo Backend Online", "regions": REGION_CODES, "docs": "/docs"}
