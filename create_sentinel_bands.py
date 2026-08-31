import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Create main folder and regional subfolders for Sentinel-2 bands
os.makedirs("data/sentinel2", exist_ok=True)

regions = {
    "Default": {"path": "data/sentinel2", "lon": 79.5, "lat": 22.2},
    "MadhyaPradesh": {"path": "data/sentinel2/MadhyaPradesh", "lon": 79.5, "lat": 22.2},
    "Odisha": {"path": "data/sentinel2/Odisha", "lon": 84.8, "lat": 22.3},
    "Andhra": {"path": "data/sentinel2/Andhra", "lon": 83.0, "lat": 18.8},
    "Karnataka": {"path": "data/sentinel2/Karnataka", "lon": 76.4, "lat": 15.6},
}

# Spectral reflectance parameters for Sentinel-2 Level-2A surface reflectance
bands = {
    "B02": {"base_val": 1500, "scale": 300},  # Blue (490 nm)
    "B04": {"base_val": 2200, "scale": 400},  # Red (665 nm)
    "B11": {"base_val": 4200, "scale": 600},  # SWIR-1 (1610 nm)
    "B12": {"base_val": 3500, "scale": 800},  # SWIR-2 (2190 nm - key for manganese detection)
}

rows, cols = 500, 500
np.random.seed(42)

for reg_name, reg_info in regions.items():
    os.makedirs(reg_info["path"], exist_ok=True)
    transform = from_origin(reg_info["lon"], reg_info["lat"], 0.001, 0.001)

    for band_name, properties in bands.items():
        # Generate surface reflectance array (scaled uint16 integer range 0-10000)
        noise = np.random.normal(0, properties["scale"], (rows, cols))
        grid = (properties["base_val"] + noise).clip(0, 10000).astype(np.uint16)

        file_path = os.path.join(reg_info["path"], f"{band_name}.jp2")

        try:
            # Write raster band using JP2OpenJPEG driver
            with rasterio.open(
                file_path,
                "w",
                driver="JP2OpenJPEG",
                height=rows,
                width=cols,
                count=1,
                dtype=grid.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(grid, 1)
        except Exception:
            # Fallback to GTiff if JP2 driver is unavailable on the local GDAL build
            with rasterio.open(
                file_path.replace(".jp2", ".tif"),
                "w",
                driver="GTiff",
                height=rows,
                width=cols,
                count=1,
                dtype=grid.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(grid, 1)

print("Successfully generated Sentinel-2 multispectral bands across all regions in data/sentinel2/")