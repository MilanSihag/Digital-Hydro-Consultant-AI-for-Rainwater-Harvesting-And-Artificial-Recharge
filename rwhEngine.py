import pandas as pd
import numpy as np
import joblib
import json
import requests
from pathlib import Path


class SmartRWHAdvisor:
    def __init__(
        self,
        data_path="rainfallDataSmartFeatures.parquet",
        model_path="rwh_suitability_model.pkl",
    ):
        """
        Initializes the advisor by loading the hydrological feature store
        and the trained ML Suitability Classifier.
        """
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)

        # 1. Load Rainfall Data
        print("⚙️ Loading Hydrological Feature Store...")
        if self.data_path.exists():
            self.df = pd.read_parquet(self.data_path)
            print("✅ Rainfall Data Loaded Successfully.")
        else:
            raise FileNotFoundError(
                f"CRITICAL: {data_path} not found. Run preprocessing script first."
            )

        # 2. Load ML Model
        print("🤖 Loading ML Suitability Model...")
        if self.model_path.exists():
            loaded_bundle = joblib.load(self.model_path)
            self.ml_model = loaded_bundle["model"]
            self.scaler = loaded_bundle["scaler"]
            self.label_map = loaded_bundle["label_map"]
            print("✅ ML Model Loaded Successfully.")
        else:
            print(
                "⚠️ WARNING: ML Model not found. Suitability Scoring will be disabled."
            )
            self.ml_model = None

    def fetch_soil_data(self, lat, lon):
        """
        Fetches Soil Texture from SoilGrids API (Open Source).
        Returns: Infiltration Rate (mm/hr) and Soil Type Label.
        """
        try:
            # SoilGrids API Endpoint for Point Query
            url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}&property=clay&property=sand&depth=0-30cm"

            # 3 second timeout to prevent app hanging
            response = requests.get(url, timeout=3)
            data = response.json()

            # Extract mean values for 0-30cm depth (Topsoil)
            # API returns values in dg/kg (decigrams). Divide by 10 to get %.
            clay_pct = 0
            sand_pct = 0

            for layer in data["properties"]["layers"]:
                if layer["name"] == "clay":
                    clay_pct = layer["depths"][0]["values"]["mean"] / 10
                elif layer["name"] == "sand":
                    sand_pct = layer["depths"][0]["values"]["mean"] / 10

            # Determine Soil Type & Hydraulic Properties
            if sand_pct > 60:
                return {
                    "label": "Sandy Soil (High Permeability)",
                    "sand_pct": sand_pct,
                    "clay_pct": clay_pct,
                    "infiltration_rate": 30.0,  # mm/hr
                    "recommended_structure": "Recharge Pit",
                    "depth_advice": "2.0m - 3.0m (Shallow is sufficient)",
                }
            elif clay_pct > 35:
                return {
                    "label": "Clay/Black Cotton (Low Permeability)",
                    "sand_pct": sand_pct,
                    "clay_pct": clay_pct,
                    "infiltration_rate": 3.0,  # mm/hr (Very Slow)
                    "recommended_structure": "Recharge Shaft (Drilled)",
                    "depth_advice": "10.0m - 15.0m (Must pierce clay layer to reach aquifer)",
                }
            else:
                return {
                    "label": "Loam (Balanced)",
                    "sand_pct": sand_pct,
                    "clay_pct": clay_pct,
                    "infiltration_rate": 15.0,  # mm/hr
                    "recommended_structure": "Recharge Trench",
                    "depth_advice": "3.0m - 5.0m",
                }

        except Exception as e:
            print(f"⚠️ Soil API Error: {e}. Defaulting to Loam.")
            return {
                "label": "Unknown (Defaulting to Loam)",
                "sand_pct": 40,
                "clay_pct": 20,
                "infiltration_rate": 15.0,
                "recommended_structure": "Recharge Trench",
                "depth_advice": "3.0m",
            }

    def calculate_recharge_design_dynamic(
        self, area_sqm, peak_intensity_mm_hr, inf_rate, structure_type
    ):
        """
        Calculates the dimensions of the Artificial Recharge Structure.
        Logic: The pit must handle the 'Surge Volume' of a 15-min cloudburst.
        """
        # 1. Calculate Surge Volume (Inflow)
        # Inflow Volume in 15 mins (meters cubed)
        # Factor 0.25 is for 15 minutes
        inflow_vol_m3 = (area_sqm * (peak_intensity_mm_hr / 1000)) * 0.25

        # 2. Calculate Outflow (Infiltration) during surge
        # We assume a standard effective wetted area for calculation based on structure type
        if structure_type == "Recharge Pit":
            wetted_area = 6.0  # roughly 2m x 3m pit walls + bottom
        elif structure_type == "Recharge Shaft":
            wetted_area = 2.0  # Smaller surface area, relying on depth
        else:
            wetted_area = 10.0  # Trench has large surface area

        outflow_vol_m3 = (wetted_area * (inf_rate / 1000)) * 0.25

        # 3. Net Buffer Volume Required
        buffer_vol_m3 = max(0.5, inflow_vol_m3 - outflow_vol_m3)

        return {
            "type": structure_type,
            "min_volume_m3": round(buffer_vol_m3, 2),
            "dimensions_hint": f"Approx {round(buffer_vol_m3 ** (1 / 3), 1)}m x {round(buffer_vol_m3 ** (1 / 3), 1)}m x {round(buffer_vol_m3 ** (1 / 3), 1)}m",
        }

    def get_runoff_coefficient(self, material_type):
        # CPWD Table 2 Mapping
        return {
            "Tiles": 0.85,
            "Corrugated Metal": 0.90,
            "Concrete": 0.80,
            "Brick Pavement": 0.75,
            "Green Roof": 0.50,
        }.get(material_type, 0.80)

    def calculate_pipe_diameter(self, area_sqm, intensity_mm_hr):
        # CPWD Table 4 Logic
        peak_flow_lpm = (area_sqm * intensity_mm_hr) / 60
        if peak_flow_lpm <= 50:
            return "75 mm"
        elif peak_flow_lpm <= 120:
            return "100 mm"
        elif peak_flow_lpm <= 300:
            return "150 mm"
        else:
            return "200 mm"

    def optimize_tank_size_simulation(
        self, loc_data, roof_area, daily_demand_liters=675
    ):
        # Optimization Logic
        max_dry_days = loc_data["avg_max_dry_days"]
        trend_dry = loc_data["trend_dry_days_per_year"]
        reliability_cv = loc_data["cv_reliability"]

        future_dry_gap = max_dry_days + (max(0, trend_dry) * 5)
        max_storage_req = daily_demand_liters * future_dry_gap

        optimization_factor = 0.5 + (0.5 * min(reliability_cv, 1.0))
        optimized_size = max_storage_req * optimization_factor

        if loc_data["trend_peak_intensity"] > 0:
            optimized_size *= 1.10

        annual_yield = loc_data["annual_avg_mm"] * roof_area * 0.80 * 0.85
        if optimized_size > (annual_yield * 0.6):
            optimized_size = annual_yield * 0.6

        return int(round(optimized_size / 500) * 500)

    def generate_assessment(
        self, lat, lon, roof_area, roof_material="Concrete", daily_demand=675
    ):
        """
        The Master Function.
        """
        # 1. Find Nearest Grid Point & Fetch Soil
        dist = (self.df["LATITUDE"] - lat) ** 2 + (self.df["LONGITUDE"] - lon) ** 2
        loc_data = self.df.loc[dist.idxmin()]

        print(f"🌍 Fetching live soil data for {lat}, {lon}...")
        soil_data = self.fetch_soil_data(lat, lon)
        print(f"   -> Detected: {soil_data['label']}")

        # 2. Extract Hydrological Data
        annual_rain = loc_data["annual_avg_mm"]
        peak_intensity_hr = loc_data["design_15min_overflow_intensity_mm"] * 4

        # 3. ML Suitability Scoring (With Soil Penalty)
        suitability_score = 0
        suitability_label = "N/A"
        if self.ml_model:
            feature_names = [
                "annual_avg_mm",
                "cv_reliability",
                "avg_max_dry_days",
                "trend_dry_days_per_year",
                "trend_peak_intensity",
            ]
            ml_features = pd.DataFrame(
                [
                    [
                        loc_data["annual_avg_mm"],
                        loc_data["cv_reliability"],
                        loc_data["avg_max_dry_days"],
                        loc_data["trend_dry_days_per_year"],
                        loc_data["trend_peak_intensity"],
                    ]
                ],
                columns=feature_names,
            )

            suitability_label = self.ml_model.predict(ml_features)[0]
            score_map = {
                "High Potential": 95,
                "Moderate Suitability": 75,
                "Water Stressed": 50,
                "Critical Scarcity": 30,
            }
            suitability_score = score_map.get(suitability_label, 50)

            # PENALTY: Clay soil makes RWH harder/more expensive
            if soil_data["infiltration_rate"] < 5.0:
                suitability_score -= 15
                suitability_label += " (Soil Constraints)"

        # 4. Infrastructure Calculations
        c_val = self.get_runoff_coefficient(roof_material)
        annual_yield_liters = annual_rain * roof_area * c_val * 0.85

        rec_pipe = self.calculate_pipe_diameter(roof_area, peak_intensity_hr)
        rec_tank = self.optimize_tank_size_simulation(loc_data, roof_area, daily_demand)

        # Dynamic Recharge Sizing
        recharge_specs = self.calculate_recharge_design_dynamic(
            roof_area,
            peak_intensity_hr,
            soil_data["infiltration_rate"],
            soil_data["recommended_structure"],
        )

        # 5. Smart Insights
        insights = []
        if loc_data["trend_peak_intensity"] > 0:
            insights.append(
                f"⚠️ Cloudburst Alert: Intensity increasing. Pipe size upgraded to {rec_pipe}."
            )
        if soil_data["infiltration_rate"] < 5.0:
            insights.append(
                f"🧱 Soil Alert: Low permeability clay detected. You must drill a Recharge Shaft {soil_data['depth_advice']} to be effective."
            )
        if not insights:
            insights.append("✅ Climate & Soil are favorable for standard RWH systems.")

        # 6. Final Report
        report = {
            "meta": {
                "latitude": lat,
                "longitude": lon,
                "grid_point": f"{loc_data['LATITUDE']}, {loc_data['LONGITUDE']}",
            },
            "suitability": {
                "score": suitability_score,
                "label": suitability_label,
                "annual_rainfall_mm": int(annual_rain),
            },
            "site_geology": {
                "soil_type": soil_data["label"],
                "composition": f"Sand: {int(soil_data['sand_pct'])}%, Clay: {int(soil_data['clay_pct'])}%",
                "infiltration_rate": f"{soil_data['infiltration_rate']} mm/hr",
            },
            "recommendations": {
                "tank_capacity": f"{rec_tank} Liters",
                "pipe_diameter": rec_pipe,
                "artificial_recharge": {
                    "structure_type": recharge_specs["type"],
                    "pit_volume_m3": recharge_specs["min_volume_m3"],
                    "depth_recommendation": soil_data["depth_advice"],
                    "note": "Designed for 15-min Peak Intensity Surge.",
                },
            },
            "smart_insights": insights,
        }

        return report


# --- Test Block ---
if __name__ == "__main__":
    try:
        engine = SmartRWHAdvisor()
        # Test 1: User in a Clay-heavy region (e.g., parts of Indore/Central India)
        # Lat/Lon for testing purposes
        print(
            json.dumps(
                engine.generate_assessment(22.71, 75.85, 120, "Concrete"), indent=4
            )
        )
    except Exception as e:
        print(f"Error: {e}")
