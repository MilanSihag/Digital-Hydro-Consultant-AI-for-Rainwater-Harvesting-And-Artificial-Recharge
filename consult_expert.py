import pandas as pd
import numpy as np
import joblib
import warnings
import json

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_FILE = "INDIA_HYDROLOGY_FINAL_LABELED.parquet"
MODEL_FILE = "hydro_ultimate_rf_model.pkl"

# The exact 20 features the model expects
FEATURES = [
    "annual_avg_mm",
    "recent_10yr_avg_mm",
    "cv_reliability",
    "avg_max_dry_days",
    "p95_daily_mm",
    "peak_daily_mm",
    "trend_dry_days_per_year",
    "trend_rainy_days_per_year",
    "trend_sdii_intensity",
    "trend_p95_intensity",
    "trend_peak_intensity",
    "design_15min_filter_intensity_mm",
    "design_15min_overflow_intensity_mm",
    "avg_sand_pct",
    "avg_clay_pct",
    "ELEVATION_MEAN",
    "RELIEF_M",
    "RUGGEDNESS_TRI",
    "SLOPE_DEG",
    "CURVATURE",
]


def load_system():
    print("🔄 Booting up Digital Hydro-Consultant...")
    df = pd.read_parquet(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    return df, model


def get_nearest_grid(target_lat, target_lon, df):
    # Calculates the closest 25km grid center to the user's requested coordinate
    distances = np.sqrt(
        (df["LATITUDE"] - target_lat) ** 2 + (df["LONGITUDE"] - target_lon) ** 2
    )
    nearest_idx = distances.idxmin()
    return df.loc[nearest_idx]


def generate_report(target_lat, target_lon, df, model):
    # 1. Fetch Location Data
    grid = get_nearest_grid(target_lat, target_lon, df)

    # 2. Prepare the AI inputs
    X_input = grid[FEATURES].to_frame().T

    # 3. Ask the Expert (Prediction)
    predicted_zone = model.predict(X_input)[0]

    # 4. Extract Real-World Physics for the Report
    rain = grid["recent_10yr_avg_mm"]
    peak_storm = grid["peak_daily_mm"]
    dry_days = grid["avg_max_dry_days"]
    clay = grid["avg_clay_pct"]
    sand = grid["avg_sand_pct"]
    slope = grid["SLOPE_DEG"]
    rugged = grid["RUGGEDNESS_TRI"]

    # 5. Engineering Logic (Translating AI to Action)
    harvest_liters = rain * 150 * 0.85

    structure = "Custom Assessment Required"
    warning = "Standard operational parameters."

    if "Zone 4" in predicted_zone:
        structure = "Above-Ground Cisterns / Contour Trenches"
        warning = f"⚠️ HIGH RUNOFF RISK: Slope is {slope:.1f}°. Do not build open pits. Landslide/erosion danger."
    elif "Zone 3" in predicted_zone:
        structure = "Surface Ponds / Deep Recharge Shafts"
        warning = f"⚠️ LOW PERMEABILITY: Clay content is high ({clay:.1f}%). Passive pits will flood. Requires deep injection past clay layer."
    elif "Zone 1" in predicted_zone or "Zone 2" in predicted_zone:
        structure = "Standard Recharge Pit with Desilting Chamber"
        if peak_storm > 150:
            warning = f"⚠️ FLASH FLOOD RISK: Extreme historical storm ({peak_storm:.0f}mm/day). Upsize overflow pipes by 30%."
        else:
            warning = f"✅ Excellent infiltration expected. Sand content: {sand:.1f}%."
    elif "Zone 5" in predicted_zone:
        structure = "Hybrid System (Filter -> Tank -> Soak Pit)"
        warning = f"Moderate terrain. Longest dry spell is {dry_days:.0f} days. Size primary tank accordingly."

    # 6. RETURN AS JSON (Replacing the print statements)
    report_dict = {
        "location": {
            "requested_lat": float(target_lat),
            "requested_lon": float(target_lon),
            "matched_grid_lat": float(grid["LATITUDE"]),
            "matched_grid_lon": float(grid["LONGITUDE"]),
        },
        "hydrology": {
            "reliable_rainfall_mm_yr": float(round(rain, 1)),
            "max_dry_spell_days": int(round(dry_days, 0)),
            "est_roof_harvest_liters_yr": float(round(harvest_liters, 0)),
            "extreme_storm_load_mm_day": float(round(peak_storm, 1)),
        },
        "geotechnical": {
            "slope_degrees": float(round(slope, 2)),
            "ruggedness_tri": float(round(rugged, 1)),
            "sand_percent": float(round(sand, 1)),
            "clay_percent": float(round(clay, 1)),
        },
        "ai_diagnosis": {
            "expert_zone": predicted_zone,
            "recommended_structure": structure,
            "engineering_note": warning,
        },
    }

    # Return a formatted JSON string
    return json.dumps(report_dict, indent=4)


if __name__ == "__main__":
    df_master, ai_model = load_system()

    # Test it
    json_output = generate_report(29.6, 74.3, df_master, ai_model)
    print(json_output)


if __name__ == "__main__":
    try:
        df_master, ai_model = load_system()

        # Test Case: North Rajasthan Agricultural Belt (Hanumangarh / Suratgarh area)
        test_lat = 29.6
        test_lon = 74.3

        generate_report(test_lat, test_lon, df_master, ai_model)


    except FileNotFoundError:
        print(
            "❌ Error: Could not find the Dataset or Model files. Make sure they are in the same folder!"
        )
