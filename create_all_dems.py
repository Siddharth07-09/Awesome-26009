import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Create data directory if it does not exist
os.makedirs("data", exist_ok=True)

# Coordinates and filenames for all 4 major Indian Manganese belts
regions = {
    "Madhya Pradesh (Balaghat)": {
        "top_left_lon": 79.5,
        "top_left_lat": 22.2,
        "filename": "CartoDEM.tif"
    },
    "Odisha Belt (Keonjhar)": {
        "top_left_lon": 84.8,
        "top_left_lat": 22.3,
        "filename": "CartoDEM_Odisha.tif"
    },
    "Andhra Pradesh Belt (Garividi)": {
        "top_left_lon": 83.0,
        "top_left_lat": 18.8,
        "filename": "CartoDEM_Andhra.tif"
    },
    "Karnataka Belt (Sandur)": {
        "top_left_lon": 76.4,
        "top_left_lat": 15.6,
        "filename": "CartoDEM_Karnataka.tif"
    }
}

rows, cols = 1000, 1000

for region_name, config in regions.items():
    transform = from_origin(config["top_left_lon"], config["top_left_lat"], 0.001, 0.001)
    
    # Generate realistic terrain topography array (300m - 700m elevation range)
    x = np.linspace(0, 10, cols)
    y = np.linspace(0, 10, rows)
    X, Y = np.meshgrid(x, y)
    elevation = 350 + 180 * np.sin(X / 2.0) * np.cos(Y / 2.0) + np.random.normal(0, 4, (rows, cols))
    elevation = elevation.astype(np.float32)
    
    output_path = f"data/{config['filename']}"
    
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=elevation.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(elevation, 1)
        
    print(f"Generated {region_name} raster at: {output_path}")