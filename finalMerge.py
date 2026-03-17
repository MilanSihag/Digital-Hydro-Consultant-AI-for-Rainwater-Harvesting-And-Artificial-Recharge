import pandas as pd

# 1. Load the Data
print("Loading datasets...")
df_master = pd.read_parquet("final_capstone_dataset.parquet")
df_terrain = pd.read_csv(
    "india_terrain_complete.csv"
)  # Or whatever you named the final CSV

# 2. Round Coordinates to guarantee a perfect match (4 decimal places)
df_master["LATITUDE"] = df_master["LATITUDE"].round(4)
df_master["LONGITUDE"] = df_master["LONGITUDE"].round(4)
df_terrain["LATITUDE"] = df_terrain["LATITUDE"].round(4)
df_terrain["LONGITUDE"] = df_terrain["LONGITUDE"].round(4)

# Drop any duplicates in terrain just in case the scraper tripped over itself
df_terrain = df_terrain.drop_duplicates(subset=["LATITUDE", "LONGITUDE"])

# 3. The Merge
print(f"Master rows before merge: {len(df_master)}")
df_final = pd.merge(df_master, df_terrain, on=["LATITUDE", "LONGITUDE"], how="left")
print(f"Master rows after merge:  {len(df_final)}")

# 4. Check for missing data
missing_terrain = df_final["SLOPE_DEG"].isna().sum()
if missing_terrain > 0:
    print(f"⚠️ Warning: {missing_terrain} grids are missing terrain data.")
else:
    print("✅ 100% Match! No missing terrain data.")

# 5. Save the Ultimate Dataset
output_file = "INDIA_HYDROLOGY_MASTER.parquet"
df_final.to_parquet(output_file, index=False)
print(f"🎉 Saved the final master dataset to: {output_file}")
