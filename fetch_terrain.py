import pandas as pd
import requests
import time
import numpy as np
import os
import math
import sys
import random  # <--- The Secret Weapon

# --- CONFIG ---
INPUT_FILE = "final_capstone_dataset.parquet"
OUTPUT_CSV = "india_terrain_complete.csv"
GRIDS_PER_BATCH = 2  # Light Payload (Safe for low-trust IPs)
BASE_URL = "https://api.open-meteo.com/v1/elevation"

# --- STEALTH SETTINGS ---
# We don't use a fixed delay. We use a range.
MIN_DELAY = 5.0
MAX_DELAY = 6.0
PENALTY_PAUSE = 120  # 2 Minutes (Standard "Cool off" bin)


def calculate_metrics_string(lats, lons, elevs):
    if not elevs or len(elevs) != 25:
        return None

    mean = np.mean(elevs)
    relief = np.max(elevs) - np.min(elevs)
    rugged = np.std(elevs)

    try:
        ref_lat, ref_lon = np.mean(lats), np.mean(lons)
        lat_scale = 111320.0
        lon_scale = 40075000.0 * math.cos(math.radians(ref_lat)) / 360.0
        x = np.array([(lo - ref_lon) * lon_scale for lo in lons])
        y = np.array([(la - ref_lat) * lat_scale for la in lats])
        z = np.array(elevs)

        A = np.c_[x, y, np.ones(len(x))]
        C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        slope = math.degrees(math.atan(math.sqrt(C[0] ** 2 + C[1] ** 2)))

        matrix = np.array(elevs).reshape(5, 5)
        mask = np.ones((5, 5), dtype=bool)
        mask[1:4, 1:4] = False
        curve = np.mean(matrix[mask]) - matrix[2, 2]
    except:
        slope, curve = 0, 0

    return f"{round(mean, 2)},{round(relief, 2)},{round(rugged, 2)},{round(slope, 2)},{round(curve, 2)}"


def main():
    print(f"🕵️ Starting STEALTH Scraper...")
    print(
        f"   Strategy: Batch {GRIDS_PER_BATCH}. Jitter Delay ({MIN_DELAY}-{MAX_DELAY}s)."
    )

    if not os.path.exists(INPUT_FILE):
        return
    df = pd.read_parquet(INPUT_FILE)
    unique_grids = df[["LATITUDE", "LONGITUDE"]].drop_duplicates().values.tolist()

    # Resume Logic
    processed = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            done = pd.read_csv(OUTPUT_CSV)
            for _, r in done.iterrows():
                processed.add((r["LATITUDE"], r["LONGITUDE"]))
        except:
            pass

    todo = [g for g in unique_grids if (g[0], g[1]) not in processed]
    total_batches = int(np.ceil(len(todo) / GRIDS_PER_BATCH))
    print(f"📋 Processing {len(todo)} grids in {total_batches} batches.")

    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w") as f:
            f.write(
                "LATITUDE,LONGITUDE,ELEVATION_MEAN,RELIEF_M,RUGGEDNESS_TRI,SLOPE_DEG,CURVATURE\n"
            )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    )

    offsets = np.linspace(-0.11, 0.11, 5)

    # --- BATCH LOOP ---
    for i in range(0, len(todo), GRIDS_PER_BATCH):
        batch = todo[i : i + GRIDS_PER_BATCH]

        # 1. Prepare Batch
        req_lats, req_lons = [], []
        meta = []
        cursor = 0
        for lat, lon in batch:
            g_lats, g_lons = [], []
            for la in offsets:
                for lo in offsets:
                    g_lats.append(round(lat + la, 4))
                    g_lons.append(round(lon + lo, 4))
            req_lats.extend(g_lats)
            req_lons.extend(g_lons)
            meta.append((lat, lon, g_lats, g_lons, cursor, cursor + 25))
            cursor += 25

        params = {
            "latitude": ",".join(map(str, req_lats)),
            "longitude": ",".join(map(str, req_lons)),
        }

        # 2. Fetch with Jitter
        success = False
        attempts = 0

        while attempts < 5:
            try:
                # Jitter Sleep *Before* Request
                jitter = random.uniform(MIN_DELAY, MAX_DELAY)
                time.sleep(jitter)

                r = session.get(BASE_URL, params=params, timeout=30)

                if r.status_code == 200:
                    data = r.json().get("elevation", [])
                    if len(data) == len(req_lats):
                        rows = []
                        for lat, lon, glats, glons, s, e in meta:
                            grid_elevs = data[s:e]
                            metrics = calculate_metrics_string(glats, glons, grid_elevs)
                            if metrics:
                                rows.append(f"{lat},{lon},{metrics}\n")

                        with open(OUTPUT_CSV, "a") as f:
                            for row in rows:
                                f.write(row)

                        success = True
                        break

                elif r.status_code == 429:
                    print(
                        f"\n🛑 Rate Limit! Pausing {PENALTY_PAUSE}s (New IP needed?)...",
                        end="",
                    )
                    time.sleep(PENALTY_PAUSE)
                    attempts += 1
                else:
                    time.sleep(2)
                    attempts += 1
            except:
                time.sleep(2)
                attempts += 1

        if success:
            progress = ((len(processed) + i + len(batch)) / len(unique_grids)) * 100
            print(
                f"\r[{progress:.1f}%] ✅ Saved Batch. (Sleep: {jitter:.1f}s)   ", end=""
            )
        else:
            print(f"\n❌ Failed Batch {i}")

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
