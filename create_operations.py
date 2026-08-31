import os
import pandas as pd
import numpy as np

# Create data directory if it does not exist
os.makedirs("data", exist_ok=True)

# Seed for reproducible operational records
np.random.seed(42)

# Date range: 180 days of daily operational logs
dates = pd.date_range(start="2025-01-01", periods=180, freq="D")

# Expanded list of pits covering all 4 major Indian Manganese Belts
pit_config = {
    # Madhya Pradesh & Maharashtra Central Belt (MOIL Core Operations)
    "Bharveli_Pit_1": 850,
    "Ukwa_Pit_2": 600,
    "Tirodi_Pit_3": 750,
    "Dongri_Buzurg_Pit_4": 900,
    "Mansar_Pit_5": 500,
    "Kandri_Pit_6": 650,
    
    # Odisha Belt
    "Keonjhar_Pit_1": 800,
    "Barbil_Pit_2": 850,
    "Joda_Pit_3": 700,
    
    # Andhra Pradesh Belt
    "Garividi_Pit_1": 600,
    "Vizianagaram_Pit_2": 550,
    
    # Karnataka Belt
    "Sandur_Pit_1": 750,
    "Ballari_Pit_2": 650
}

records = []
for date in dates:
    is_monsoon = date.month in [6, 7, 8, 9]
    for pit_id, target_tons in pit_config.items():
        # Simulate realistic weather and operational constraints
        rainfall_mm = round(
            np.random.exponential(scale=25.0) if is_monsoon else np.random.exponential(scale=2.0), 
            1
        )
        excavator_downtime_hrs = round(np.random.exponential(scale=1.1), 1)
        dumper_downtime_hrs = round(np.random.exponential(scale=1.6), 1)
        blasting_delay_mins = int(
            np.random.choice([0, 15, 30, 45, 60, 90], p=[0.5, 0.2, 0.15, 0.08, 0.05, 0.02])
        )
        
        # Production shortfall calculation with pit bottleneck dynamics
        shortfall = (
            rainfall_mm * 3.8 + 
            excavator_downtime_hrs * 32.0 + 
            dumper_downtime_hrs * 18.0 + 
            blasting_delay_mins * 1.2 + 
            np.random.normal(0, 10)
        )
        shortfall = max(0, min(shortfall, target_tons * 0.75))
        actual_tons = max(50, round(target_tons - shortfall))
        
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "pit_id": pit_id,
            "target_tons": target_tons,
            "actual_tons": actual_tons,
            "rainfall_mm": rainfall_mm,
            "excavator_downtime_hrs": excavator_downtime_hrs,
            "dumper_downtime_hrs": dumper_downtime_hrs,
            "blasting_delay_mins": blasting_delay_mins
        })

df = pd.DataFrame(records)
output_path = "data/operations_log.csv"
df.to_csv(output_path, index=False)
print(f"Successfully created operational log dataset with {len(df)} entries at: {output_path}")