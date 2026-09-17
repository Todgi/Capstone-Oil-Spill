import os
import rasterio
import numpy as np
from rasterio.windows import Window

# Set the input directory containing tiled images
input_dir = r"D:/Documents/Capstone/Deep-Learning/Optik/dataset_optik/marinext_predicted/predict-1"
output_path = r"D:/Documents/Capstone/Deep-Learning/Optik/dataset_optik/marinext_tiling/predict-1/scene_0_predicted_image.tif"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# List all tile files in the directory
tiles = [f for f in os.listdir(input_dir) if f.endswith(".tif")]
tiles.sort()

# Extract tile indices from filenames
import re

def extract_indices(filename):
    match = re.search(r'_(\d+)_(\d+)_marinext\.tif$', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        raise ValueError(f"Filename format not recognized: {filename}")

# Organize tiles into a grid
tile_dict = {}
for tile in tiles:
    i, j = extract_indices(tile)
    tile_dict[(i, j)] = tile

# Read the first tile to get metadata
first_tile_path = os.path.join(input_dir, tiles[0])
with rasterio.open(first_tile_path) as src:
    tile_width = src.width
    tile_height = src.height
    dtype = src.dtypes[0]
    crs = src.crs
    transform = src.transform
    count = src.count

# Determine grid dimensions
num_tiles_x = max(i for i, j in tile_dict.keys()) + 1
num_tiles_y = max(j for i, j in tile_dict.keys()) + 1

# Compute the full image dimensions
img_width = num_tiles_x * tile_width
img_height = num_tiles_y * tile_height

# Create output raster
with rasterio.open(
    output_path, "w",
    driver="GTiff",
    height=img_height,
    width=img_width,
    count=count,
    dtype=dtype,
    crs=crs,
    transform=transform
) as dst:
    
    # Merge tiles into the full image
    for (i, j), tile in tile_dict.items():
        tile_path = os.path.join(input_dir, tile)
        with rasterio.open(tile_path) as src:
            tile_data = src.read()
            x_offset = i * tile_width
            y_offset = j * tile_height
            dst.write(tile_data, window=Window(x_offset, y_offset, tile_width, tile_height))

print("✅ Combining completed! Merged image saved at:", output_path)
