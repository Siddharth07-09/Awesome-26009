import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Create data directory if it does not exist
os.makedirs("data", exist_ok=True)

# Set dimensions and geo-transform for Balaghat region (EPSG:4326)
# Top-left corner: 79.5° E, 22.2° N with ~100m pixel resolution
rows, cols = 1000, 1000
transform = from_origin(79.5, 22.2, 0.001, 0.001)

# Generate synthetic terrain topography (300m - 600m elevation range)
x = np.linspace(0, 10, cols)
y = np.linspace(0, 10, rows)
X, Y = np.meshgrid(x, y)
elevation = 300 + 150 * np.sin(X / 2.0) * np.cos(Y / 2.0) + np.random.normal(0, 5, (rows, cols))
elevation = elevation.astype(np.float32)

# Save GeoTIFF raster file
with rasterio.open(
    "data/CartoDEM.tif",
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

print("File successfully created at data/CartoDEM.tif")