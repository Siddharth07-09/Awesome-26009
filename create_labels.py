import os
import pandas as pd

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Real coordinates of verified Indian Manganese Mines (has_manganese = 1) 
# and surrounding non-mineral control areas (has_manganese = 0)
mineral_data = [
    # MP & MH Central Belt (MOIL Operational Mines)
    {"latitude": 21.85016, "longitude": 80.22742, "has_manganese": 1},  # Bharveli Mine
    {"latitude": 21.68351, "longitude": 79.72464, "has_manganese": 1},  # Tirodi Mine
    {"latitude": 21.54778, "longitude": 79.68222, "has_manganese": 1},  # Dongri Buzurg Mine
    {"latitude": 21.96800, "longitude": 80.46700, "has_manganese": 1},  # Ukwa Mine
    {"latitude": 21.39600, "longitude": 79.28800, "has_manganese": 1},  # Mansar Mine
    {"latitude": 21.37800, "longitude": 79.29900, "has_manganese": 1},  # Kandri Mine
    
    # Odisha Belt
    {"latitude": 21.87500, "longitude": 85.37000, "has_manganese": 1},  # Keonjhar / Barbil Pit
    {"latitude": 21.90100, "longitude": 85.38500, "has_manganese": 1},  # Joda Belt
    
    # Andhra Pradesh & Karnataka Belts
    {"latitude": 18.29300, "longitude": 83.52800, "has_manganese": 1},  # Garividi Mine (AP)
    {"latitude": 15.13900, "longitude": 76.92100, "has_manganese": 1},  # Sandur / Ballari (KA)
    
    # Background / Non-Mineral Points (Forest, Water, Agriculture)
    {"latitude": 21.75010, "longitude": 80.12040, "has_manganese": 0},
    {"latitude": 21.89050, "longitude": 80.25010, "has_manganese": 0},
    {"latitude": 21.61000, "longitude": 80.05000, "has_manganese": 0},
    {"latitude": 21.20000, "longitude": 79.10000, "has_manganese": 0},
    {"latitude": 21.40000, "longitude": 80.00000, "has_manganese": 0},
    {"latitude": 21.50000, "longitude": 85.20000, "has_manganese": 0},
    {"latitude": 18.10000, "longitude": 83.30000, "has_manganese": 0},
    {"latitude": 15.00000, "longitude": 76.50000, "has_manganese": 0},
]

df = pd.DataFrame(mineral_data)
df.to_csv("data/mineral_labels.csv", index=False)
print("File successfully created at data/mineral_labels.csv")