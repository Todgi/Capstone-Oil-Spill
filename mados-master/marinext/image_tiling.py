import os
import rasterio
import numpy as np
from rasterio.windows import Window

# Set the input directory containing Sentinel-2 JP2 bands
input_dir = r"D:/Documents/Capstone/Deep-Learning/Optik/dataset_optik/sentinel_rad"
output_dir = r"D:/Documents/Capstone/Deep-Learning/Optik/mados-master/data/MADOS/Scene/10"
os.makedirs(output_dir, exist_ok=True)

# Define tiling parameters
# num_tiles_x = 4  # Number of tiles along width
# num_tiles_y = 4  # Number of tiles along height
tile_width = 120
tile_height = 120


# List all Sentinel-2 bands (JP2 files) in the directory
#  bands = [f for f in os.listdir(input_dir) if f.endswith(".jp2")]
bands = [f for f in os.listdir(input_dir) if f.endswith(".tif")]
bands.sort()  # Ensure consistent order

# Read the first band to determine dimensions
first_band_path = os.path.join(input_dir, bands[0])
with rasterio.open(first_band_path) as src:
    img_width = src.width
    img_height = src.height
    transform = src.transform

# Compute tile size
# tile_width = img_width // num_tiles_x
# tile_height = img_height // num_tiles_y
# Compute tile array number
num_tiles_x = img_width // tile_width
num_tiles_y = img_height // tile_height

# Process each Sentinel-2 band
for band in bands:
    band_path = os.path.join(input_dir, band)

    with rasterio.open(band_path) as src:
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                print('Tiling col: {} , row: {} '.format(i,j))
                # Define window for tiling
                window = Window(i * tile_width, j * tile_height, tile_width, tile_height)
                transform_tile = src.window_transform(window)

                # Read the tile
                tile_data = src.read(1, window=window)

                # Create output filename
                tile_filename = f"{band[:-4]}_tile_{i}_{j}.tif"
                tile_path = os.path.join(output_dir, tile_filename)

                # Save the tile
                with rasterio.open(
                    tile_path, "w",
                    driver="GTiff",
                    height=tile_height,
                    width=tile_width,
                    count=1,
                    dtype=tile_data.dtype,
                    crs=src.crs,
                    transform=transform_tile
                ) as dst:
                    dst.write(tile_data, 1)

print("✅ Tiling completed! Tiles saved in:", output_dir)
