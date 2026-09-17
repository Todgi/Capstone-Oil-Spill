import os
import sys
import time
import logging
from fastapi.responses import RedirectResponse
import pandas as pd
from io import BytesIO
import gc
import rasterio
from PIL import Image, ImageEnhance
import signal
import atexit
import tempfile
import threading

# Set environment variables before any other imports
conda_prefix = r'C:\Users\alvito\anaconda3\envs\capstone'

# Add GDAL DLL directories to PATH
os.add_dll_directory(os.path.join(conda_prefix, 'Library', 'bin'))

# Set GDAL environment variables
os.environ['GDAL_DATA'] = os.path.join(conda_prefix, 'Library', 'share', 'gdal')
os.environ['PROJ_LIB'] = os.path.join(conda_prefix, 'Library', 'share', 'proj')

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import io
import os
import base64
import tempfile
import shutil
import numpy as np
from PIL import Image
import rasterio
from fastapi import File, UploadFile
from logging import getLogger
import json
import rasterio.features
from shapely.geometry import shape, Point
from typing import List
import subprocess
import psutil
import requests
import httpx
from fastapi.responses import JSONResponse

# Configuration
MADOS_PORT = 5000
OPENOIL_PORT = 5001

# Configure logging
logger = getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("webgis_app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def is_port_in_use(port):
    """Check if a port is in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """Kill any process using the specified port"""
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    logger.info(f"Killing process {proc.pid} ({proc.name()}) using port {port}")
                    proc.kill()
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    # If psutil approach didn't work, try system commands
    try:
        if os.name == 'nt':  # Windows
            output = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()
            for line in output.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    pid = line.strip().split()[-1]
                    logger.info(f"Killing process {pid} using port {port} via taskkill")
                    os.system(f"taskkill /F /PID {pid}")
                    return True
        else:  # Linux/Mac
            output = subprocess.check_output(f'lsof -i :{port} -t', shell=True).decode()
            if output:
                pid = output.strip()
                logger.info(f"Killing process {pid} using port {port} via kill")
                os.system(f"kill -9 {pid}")
                return True
    except Exception as e:
        logger.warning(f"Error killing process on port {port}: {e}")
    
    return False

def kill_python_process_by_script(script_name):
    """Kill Python processes running a specific script"""
    count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] in ['python.exe', 'python', 'python3']:
                cmdline = " ".join(proc.info['cmdline'] if proc.info['cmdline'] else [])
                if script_name in cmdline:
                    logger.info(f"Killing process: PID {proc.pid}, script: {script_name}")
                    proc.kill()
                    count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            logger.debug(f"Error accessing process: {e}")
    
    logger.info(f"Killed {count} processes running {script_name}")
    return count > 0

def clean_temp_dirs():
    """Clean temporary MADOS directories"""
    try:
        temp_dir = tempfile.gettempdir()
        logger.info(f"Cleaning temp directories in {temp_dir}")
        
        count = 0
        for item in os.listdir(temp_dir):
            if item.startswith("mados_job_"):
                try:
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove temp dir {item}: {e}")
        
        logger.info(f"Removed {count} temporary MADOS job directories")
    except Exception as e:
        logger.warning(f"Error cleaning temp directories: {e}")

def clear_gpu_memory():
    """Clear GPU memory if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Clearing CUDA memory")
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_max_memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
            logger.info("CUDA memory cleared")
    except Exception as e:
        logger.debug(f"Failed to clear GPU memory: {e}")

def startup():
    """Initialize services during application startup"""
    logger.info("==== Starting WebGIS Oil Spill Detection Application ====")
    
    # Register cleanup handler
    atexit.register(clean_up)
    
    # Handle Ctrl+C (SIGINT)
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down...")
        clean_up()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Clean up any temporary files
    clean_temp_dirs()
    clear_gpu_memory()
    
    logger.info("Application startup complete")

def clean_up():
    """Clean up temporary files on shutdown"""
    logger.info("Cleaning up...")
    
    # Clean temporary directories
    clean_temp_dirs()
    
    logger.info("Cleanup completed")

def extract_oil_coordinates(segmentation_mask, src_transform, max_points=2000):
    """Ekstrak koordinat dari area oil (kelas 1) dalam segmentasi"""
    try:
        # Identifikasi area oil (kelas 1)
        oil_mask = segmentation_mask == 1
        
        if not np.any(oil_mask):
            logger.warning("Tidak ada area oil yang terdeteksi dalam segmentasi")
            return []
        
        # Dapatkan indeks piksel yang merupakan oil
        oil_pixels = np.where(oil_mask)
        
        # Jika terlalu banyak piksel, ambil sampel secara merata
        if len(oil_pixels[0]) > max_points:
            # Hitung interval sampling
            interval = len(oil_pixels[0]) // max_points
            y_indices = oil_pixels[0][::interval][:max_points]
            x_indices = oil_pixels[1][::interval][:max_points]
        else:
            y_indices = oil_pixels[0]
            x_indices = oil_pixels[1]
        
        # Konversi ke koordinat geografis
        oil_coordinates = []
        for y, x in zip(y_indices, x_indices):
            # Transformasi ke koordinat geografis
            lon, lat = rasterio.transform.xy(src_transform, y, x)
            oil_coordinates.append((lat, lon))
        
        # Log detailed information about the coordinates
        logger.info(f"Berhasil mengekstrak {len(oil_coordinates)} koordinat oil")
        if oil_coordinates:
            logger.info(f"Sample coordinates (first 5): {oil_coordinates[:5]}")
            logger.info(f"Coordinate range - Lat: [{min(c[0] for c in oil_coordinates):.6f}, {max(c[0] for c in oil_coordinates):.6f}], Lon: [{min(c[1] for c in oil_coordinates):.6f}, {max(c[1] for c in oil_coordinates):.6f}]")
        
        return oil_coordinates
        
    except Exception as e:
        logger.error(f"Error dalam ekstraksi koordinat oil: {str(e)}")
        return []

# Set TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info and warning messages

# Import ML-related libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import io
import base64
import cv2
# Define constants
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
IMG_CLASSES = 5
OVERLAP = 128 # Define overlap size in pixels
STEP = IMG_HEIGHT - OVERLAP # Step size for sliding window
# Define color map and other constants
COLOR_MAP = [
    [0, 0, 0],        # Background/Water (Black)
    [0, 255, 255],    # Oil (Cyan)
    [255, 0, 0],      # Lookalikes/Current (Red)
    [153, 76, 0],     # Antena Parabola (Brown)
    [0, 153, 0]       # Land (Green)
]
# Define UNet model function
def UNet(input_shape, num_classes):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    p3 = MaxPooling2D((2,2))(c3)
    
    c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
    p4 = MaxPooling2D((2,2))(c4)
    
    # Bridge
    c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)
    
    # Decoder
    u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)
    
    u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)
    
    u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)
    
    u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)
    
    outputs = Conv2D(num_classes, (1,1), activation="softmax")(c9)
    model = Model(inputs, outputs)
    return model
# Add this after the model definition and before FastAPI setup
# Remove or comment out the UNet function since we'll load the complete model
def load_model():
    try:
        # Basic TensorFlow configuration with minimal memory usage
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['TF_MKL_ALLOC_MAX_BYTES'] = '268435456'  # 256MB max allocation
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
        
        # Clear memory aggressively
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Configure CPU only
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        model_path = 'c:/Users/alvito/Documents/00. Capstone/webgis/models/saved_unetmodel_tf'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at path: {model_path}")
        
        # Load model with minimal memory usage
        with tf.device('/CPU:0'):
            try:
                # Load model with minimal options
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False
                )
                
                # Force garbage collection
                gc.collect()
                
                logger.info("Model loaded successfully with minimal memory settings")
                return model
                
            except Exception as model_error:
                logger.error(f"Model loading error: {str(model_error)}")
                raise RuntimeError(f"Model loading failed: {str(model_error)}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Model initialization failed: {str(e)}")

# Initialize model globally
model = load_model()

# FastAPI setup with automatic startup
app = FastAPI(
    title="WebGIS Oil Spill Detection",
    description="WebGIS Application for Oil Spill Detection and Analysis",
    version="1.0.0",
    on_startup=[startup]  # Call startup function when FastAPI starts
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_raster")
async def upload_raster(file: UploadFile = File(...)):
    try:
        # Validasi file
        if not file.filename.lower().endswith(('.tif', '.tiff')):
            raise HTTPException(status_code=400, detail="File harus berupa GeoTIFF")
            
        # Proses file dan kembalikan response
        return {
            "success": True,
            "bounds": {"south": -90, "west": -180, "north": 90, "east": 180},
            "image": "data:image/png;base64,..."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        # Read raster data
        raster_data = await file.read()
        
        # Process raster with rasterio
        with rasterio.io.MemoryFile(raster_data) as memfile:
            with memfile.open() as dataset:
                # Get raster metadata
                src_bounds = dataset.bounds
                src_crs = dataset.crs
                logger.info(f"Original bounds: {src_bounds}")
                logger.info(f"Original CRS: {src_crs}")

                # Transform bounds to EPSG:4326 if needed
                from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling
                import numpy as np
                import pyproj
                if src_crs is not None and src_crs.to_string() not in ["EPSG:4326", "WGS84", "+proj=longlat +datum=WGS84 +no_defs"]:
                    # Transform bounds
                    bounds_wgs84 = transform_bounds(src_crs, 'EPSG:4326',
                        src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top, densify_pts=21)
                    bounds = {
                        "south": float(bounds_wgs84[1]),
                        "west": float(bounds_wgs84[0]),
                        "north": float(bounds_wgs84[3]),
                        "east": float(bounds_wgs84[2])
                    }
                    # Reproject image to EPSG:4326 for display
                    dst_crs = 'EPSG:4326'
                    transform, width, height = calculate_default_transform(
                        src_crs, dst_crs, dataset.width, dataset.height, *dataset.bounds)
                    kwargs = dataset.meta.copy()
                    kwargs.update({
                        'crs': dst_crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })
                    # Only use first 3 bands for display
                    bands = min(3, dataset.count)
                    reprojected = np.zeros((bands, height, width), dtype=np.float32) #float32
                    for i in range(bands):
                        reproject(
                            source=dataset.read(i+1),
                            destination=reprojected[i],
                            src_transform=dataset.transform,
                            src_crs=src_crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest
                        )
                    rgb_image = np.dstack([reprojected[i] for i in range(bands)])
                else:
                    bounds = {
                        "south": float(src_bounds.bottom),
                        "west": float(src_bounds.left),
                        "north": float(src_bounds.top),
                        "east": float(src_bounds.right)
                    }
                    image = dataset.read()
                    if image.shape[0] >= 3:
                        rgb_image = np.dstack((image[0], image[1], image[2]))
                    else:
                        rgb_image = np.dstack((image[0], image[0], image[0]))
                
                # Handle NaN and Inf values
                rgb_image = np.nan_to_num(rgb_image, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Normalize to 0-255 range
                for i in range(3):
                    band = rgb_image[:,:,i]
                    if band.max() > band.min():
                        band = ((band - band.min()) * (255.0 / (band.max() - band.min())))
                    rgb_image[:,:,i] = band.clip(0, 255)
                
                # Convert to image and base64
                img = Image.fromarray(np.uint8(rgb_image))
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                
                return {
                    "success": True,
                    "bounds": bounds,
                    "crs": "EPSG:4326",
                    "image": image_base64
                }
                
    except Exception as e:
        logger.error(f"Failed to process raster: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process raster: {str(e)}")

@app.post("/simulate_oil_drift")
async def simulate_oil_drift(file: UploadFile = File(...), 
                           start_time: str = Form(None),
                           duration: int = Form(24),
                           latitude: float = Form(None),
                           longitude: float = Form(None)):
    try:
        # Baca file CSV
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        
        # Validasi format CSV
        required_columns = ['time', 'latitude', 'longitude', 'u_current', 'v_current']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV harus memiliki kolom {col}")
        
        # Tambahkan logging untuk melihat format waktu yang diterima
        logger.info(f"Format waktu yang diterima: {df['time'].head().tolist()}")
        
        # Coba beberapa metode parsing waktu
        # 1. Coba bersihkan string terlebih dahulu
        df['time'] = df['time'].astype(str).str.replace('UTC', '').str.strip()
        
        # 2. Tambahkan format ISO8601 dengan Z
        date_formats = ['%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M', 
                       '%Y-%m-%dT%H:%M:%S', '%Y%m%d%H%M%S', '%Y-%m-%d']
        
        parsed = False
        for fmt in date_formats:
            try:
                df['time'] = pd.to_datetime(df['time'], format=fmt, errors='raise')
                parsed = True
                logger.info(f"Berhasil parsing dengan format: {fmt}")
                break
            except Exception as e:
                logger.debug(f"Gagal parsing dengan format {fmt}: {str(e)}")
                continue
        
        # 3. Jika semua format gagal, coba parsing dengan format='ISO8601'
        if not parsed:
            try:
                df['time'] = pd.to_datetime(df['time'], format='ISO8601', errors='coerce')
                logger.info("Menggunakan parsing ISO8601")
            except Exception as e:
                # 4. Jika masih gagal, gunakan parsing default
                try:
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                    logger.info("Menggunakan parsing default")
                except Exception as e2:
                    logger.warning(f"Error parsing default format: {str(e2)}")
        
        # Check for NaT values after conversion
        if df['time'].isna().any():
            invalid_count = df['time'].isna().sum()
            invalid_examples = df.loc[df['time'].isna(), 'time'].head(3).tolist()
            logger.warning(f"{invalid_count} datetime values could not be parsed. Examples: {invalid_examples}")
            # Drop rows with invalid dates
            df = df.dropna(subset=['time'])
            if df.empty:
                raise ValueError("Tidak ada nilai waktu yang valid dalam data. Pastikan format waktu sesuai (YYYY-MM-DD HH:MM:SS)")
        
        # Pastikan timezone konsisten dengan menghapus timezone info
        if hasattr(df['time'].dt, 'tz_localize'):
            df['time'] = df['time'].dt.tz_localize(None)
        
        # Log data yang valid
        logger.info(f"Jumlah data valid: {len(df)}")
        logger.info(f"Range waktu data: {df['time'].min()} hingga {df['time'].max()}")
        
        if start_time:
            try:
                start_time = pd.to_datetime(start_time)
                if hasattr(start_time, 'tz_localize'):
                    start_time = start_time.tz_localize(None)
            except Exception as e:
                logger.warning(f"Error parsing start_time: {str(e)}")
                start_time = df['time'].min()
                logger.info(f"Menggunakan waktu awal dari data: {start_time}")
        else:
            # Gunakan waktu pertama dari data jika tidak ada input
            start_time = df['time'].min()
            logger.info(f"Tidak ada start_time yang diberikan, menggunakan: {start_time}")
        
        # Hitung waktu akhir berdasarkan durasi
        end_time = start_time + pd.Timedelta(hours=duration)
        
        # Log rentang waktu yang dipilih
        logger.info(f"Rentang waktu yang dipilih: {start_time} hingga {end_time}")
        
        # Filter data berdasarkan waktu dengan toleransi
        # Tambahkan toleransi 1 jam untuk mengatasi masalah perbedaan timezone atau rounding
        tolerance = pd.Timedelta(hours=1)
        filtered_df = df[(df['time'] >= (start_time - tolerance)) & 
                         (df['time'] <= (end_time + tolerance))]
        
        # Log jumlah data setelah filtering
        logger.info(f"Jumlah data setelah filtering: {len(filtered_df)}")
        
        if filtered_df.empty:
            # Jika tidak ada data dalam rentang, gunakan data terdekat
            if not df.empty:
                # Cari waktu terdekat dengan start_time yang *diberikan oleh pengguna* atau dari data
                # Prioritaskan start_time dari user jika ada
                reference_time = start_time if start_time else df['time'].min()
                
                # Hitung selisih waktu dari waktu referensi
                df['time_diff'] = abs(df['time'] - reference_time)
                closest_idx = df['time_diff'].idxmin()
                closest_time = df.loc[closest_idx, 'time']
                
                # Ambil semua data dalam rentang 24 jam dari waktu terdekat
                # Gunakan waktu awal sebagai acuan jika start_time dari user valid
                effective_start_time = start_time if start_time and (start_time >= df['time'].min() and start_time <= df['time'].max()) else closest_time

                filtered_df = df[(df['time'] >= effective_start_time) & 
                                 (df['time'] <= (effective_start_time + pd.Timedelta(hours=24)))]
                
                if filtered_df.empty:
                    # Jika masih kosong, gunakan minimal 1 data terdekat
                    filtered_df = df.loc[[closest_idx]].copy()
                
                logger.warning(f"Tidak ada data dalam rentang waktu yang dipilih. Menggunakan data terdekat atau efektif pada {filtered_df['time'].iloc[0]}")
            else:
                raise ValueError("Tidak ada data dalam rentang waktu yang dipilih")
        
        # Jika koordinat awal diberikan, gunakan untuk simulasi
        if latitude is not None and longitude is not None:
            # Implementasi algoritma simulasi oil drift sederhana
            drift_points = []
            current_lat = float(latitude)
            current_lon = float(longitude)
            
            # Faktor skala untuk simulasi (dapat disesuaikan)
            scale_factor = 0.1
            
            # Urutkan data berdasarkan waktu
            filtered_df = filtered_df.sort_values('time')
            
            for _, row in filtered_df.iterrows():
                # Pastikan nilai u_current dan v_current adalah float
                try:
                    u_current = float(row['u_current'])
                    v_current = float(row['v_current'])
                    
                    # Tangani nilai NaN atau Inf
                    if np.isnan(u_current) or np.isinf(u_current):
                        u_current = 0.0
                    if np.isnan(v_current) or np.isinf(v_current):
                        v_current = 0.0
                    
                    # Hitung pergerakan berdasarkan arus
                    current_lat += v_current * scale_factor
                    current_lon += u_current * scale_factor
                    
                    # Pastikan nilai latitude dan longitude valid
                    if np.isnan(current_lat) or np.isinf(current_lat):
                        current_lat = float(latitude)  # Reset ke nilai awal jika tidak valid
                    if np.isnan(current_lon) or np.isinf(current_lon):
                        current_lon = float(longitude)  # Reset ke nilai awal jika tidak valid
                    
                    drift_points.append({
                        'time': row['time'].isoformat(),
                        'latitude': float(current_lat),
                        'longitude': float(current_lon),
                        'u_current': float(u_current),
                        'v_current': float(v_current)
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Melewati baris dengan nilai tidak valid: {e}")
                    continue
        else:
            # Jika tidak ada koordinat awal, gunakan data langsung dari CSV
            drift_points = []
            # Urutkan data berdasarkan waktu
            filtered_df = filtered_df.sort_values('time')
            
            for _, row in filtered_df.iterrows():
                try:
                    # Pastikan nilai adalah float dan valid
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                    u_current = float(row['u_current'])
                    v_current = float(row['v_current'])
                    
                    # Tangani nilai NaN atau Inf
                    if np.isnan(lat) or np.isinf(lat) or np.isnan(lon) or np.isinf(lon):
                        continue  # Lewati baris dengan koordinat tidak valid
                    if np.isnan(u_current) or np.isinf(u_current):
                        u_current = 0.0
                    if np.isnan(v_current) or np.isinf(v_current):
                        v_current = 0.0
                    
                    drift_points.append({
                        'time': row['time'].isoformat(),
                        'latitude': float(lat),
                        'longitude': float(lon),
                        'u_current': float(u_current),
                        'v_current': float(v_current)
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Melewati baris dengan nilai tidak valid: {e}")
                    continue
        
        # Pastikan tidak ada nilai float yang tidak valid dalam respons JSON
        for point in drift_points:
            for key, value in point.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    point[key] = 0.0  # Ganti nilai tidak valid dengan 0.0
        
        return {
            "success": True,
            "drift_points": drift_points,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_hours": duration,
            "point_count": len(drift_points)
        }
        
    except Exception as e:
        logger.error(f"Error dalam simulasi oil drift: {str(e)}")
        return {"success": False, "error": str(e)}

def calculate_global_stats(image_data):
    """Calculate global statistics for the entire raster"""
    try:
        # Handle single-band SAR image
        if len(image_data.shape) == 3:
            img = image_data[0]
        else:
            img = image_data
            
        # Convert to float32 and handle invalid values
        img = img.astype(np.float32)
        
        # Handle int8 specific normalization
        if image_data.dtype == np.int8:
            # Convert from int8 range (-128 to 127) to 0-255 range
            img = (img + 128).astype(np.float32)
            # Then normalize to 0-1 range
            img = img / 255.0
        else:
            img[~np.isfinite(img)] = 0
        
        # Calculate global statistics
        global_min = np.min(img)
        global_max = np.max(img)
        global_mean = np.mean(img)
        global_std = np.std(img)
        
        # Log statistics
        logger.info(f"Global statistics:")
        logger.info(f"Min: {global_min}")
        logger.info(f"Max: {global_max}")
        logger.info(f"Mean: {global_mean}")
        logger.info(f"Std: {global_std}")
        
        return {
            'min': global_min,
            'max': global_max,
            'mean': global_mean,
            'std': global_std
        }
    except Exception as e:
        logger.error(f"Error calculating global statistics: {str(e)}")
        raise

def preprocess_image(image_data, global_stats=None):
    """Preprocess image with global normalization"""
    try:
        # Handle single-band SAR image
        if len(image_data.shape) == 3:
            img = image_data[0]
        else:
            img = image_data
            
        # Convert int8 to float32 and handle invalid values
        img = img.astype(np.float32)
        
        # Handle int8 specific normalization
        if image_data.dtype == np.int8:
            # Convert from int8 range (-128 to 127) to 0-255 range
            img = (img + 128).astype(np.float32)
            # Then normalize to 0-1 range
            img = img / 255.0
        else:
            img[~np.isfinite(img)] = 0
        
        # Use global statistics if available
        if global_stats:
            # Normalize using global statistics
            img = (img - global_stats['min']) / (global_stats['max'] - global_stats['min'])
        else:
            # Fallback to local normalization
            img_min = np.min(img)
            img_max = np.max(img)
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
        
        # Store original shape
        original_shape = img.shape
        
        # Resize image to 256x256
        img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        
        # Create RGB image by stacking the same channel
        img_rgb = np.stack([img_resized] * 3, axis=-1)
        
        return img_rgb, original_shape
        
    except Exception as e:
        logger.error(f"Error preprocessing: {str(e)}")
        raise

def postprocess_prediction(predictions, original_shape):
    """Postprocess model predictions to create final segmentation mask"""
    try:
        # Apply softmax to get probabilities
        if len(predictions.shape) == 3:
            predictions = tf.nn.softmax(predictions, axis=-1).numpy()
        
        # Get class with highest probability
        mask = np.argmax(predictions, axis=-1)
        
        # Resize back to original size
        h, w = original_shape
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Define adaptive thresholds based on data distribution
        thresholds = {
            0: 0.3,  # Background/Water
            1: 0.35, # Oil
            2: 0.35, # Lookalikes/Current
            3: 0.4,  # Antena Parabola
            4: 0.4   # Land
        }
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        
        # Clean up oil class
        oil_mask = mask == 1
        oil_mask = cv2.morphologyEx(oil_mask.astype(np.uint8), 
                                  cv2.MORPH_OPEN, 
                                  kernel,
                                  iterations=1)
        oil_mask = cv2.morphologyEx(oil_mask, 
                                  cv2.MORPH_CLOSE, 
                                  kernel,
                                  iterations=2)
        
        # Clean up lookalikes class
        lookalikes_mask = mask == 2
        lookalikes_mask = cv2.morphologyEx(lookalikes_mask.astype(np.uint8), 
                                         cv2.MORPH_OPEN, 
                                         kernel,
                                         iterations=1)
        lookalikes_mask = cv2.morphologyEx(lookalikes_mask, 
                                         cv2.MORPH_CLOSE, 
                                         kernel,
                                         iterations=2)
        
        # Update mask with cleaned classes
        mask[oil_mask > 0] = 1
        mask[lookalikes_mask > 0] = 2
        
        return mask
    except Exception as e:
        logger.error(f"Postprocessing error: {str(e)}")
        raise

@app.post("/predict_from_raster")
async def predict_from_raster(file: UploadFile = File(...)):
    try:
        content = await file.read()
        with rasterio.io.MemoryFile(content) as memfile:
            with memfile.open() as src:
                # Read raster data
                image_data = src.read()
                bounds = src.bounds
                transform = src.transform
                
                # Log input information
                logger.info(f"Input image shape: {image_data.shape}")
                logger.info(f"Data type: {image_data.dtype}")
                logger.info(f"Value range: [{float(np.min(image_data))}, {float(np.max(image_data))}]")
                
                # Calculate global statistics
                global_stats = calculate_global_stats(image_data)
                
                # Convert global_stats to Python native types
                global_stats = {
                    'min': float(global_stats['min']),
                    'max': float(global_stats['max']),
                    'mean': float(global_stats['mean']),
                    'std': float(global_stats['std'])
                }
                
                # Process image for prediction with global statistics
                processed_image, original_shape = preprocess_image(
                    image_data, 
                    global_stats=global_stats
                )
                
                # Add batch dimension
                processed_image = np.expand_dims(processed_image, axis=0)
                
                # Make prediction
                logger.info("Making prediction on full image...")
                predictions = model.predict(processed_image, verbose=0)
                
                # Remove batch dimension
                predictions = predictions[0]
                logger.info(f"Predictions shape: {predictions.shape}")
                
                # Process predictions
                segmentation_mask = postprocess_prediction(predictions, original_shape)
                logger.info(f"Segmentation mask shape: {segmentation_mask.shape}")
                
                # Extract oil coordinates
                oil_coordinates = extract_oil_coordinates(segmentation_mask, transform)
                
                # Save coordinates in global state
                app.state.oil_coordinates = oil_coordinates
                
                # Convert to base64
                original_base64 = array_to_base64(image_data[0])
                segmentation_base64 = mask_to_base64(segmentation_mask)
                
                # Convert bounds to Python native types
                bounds_dict = {
                    "south": float(bounds.bottom),
                    "west": float(bounds.left),
                    "north": float(bounds.top),
                    "east": float(bounds.right)
                }
                
                return {
                    "success": True,
                    "bounds": bounds_dict,
                    "original_image": original_base64,
                    "segmentation_image": segmentation_base64,
                    "oil_coordinates": oil_coordinates,
                    "global_stats": global_stats
                }
                
    except Exception as e:
        logger.error(f"Error in raster processing: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        # Cleanup
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)

def array_to_base64(array):
    try:
        # Pastikan array adalah 2D
        if len(array.shape) == 3:
            array = array[0]
        
        # Konversi ke float32 dan tangani nilai tidak valid
        array = array.astype(np.float32)
        
        # Handle int8 specific normalization
        if array.dtype == np.int8:
            # Convert from int8 range (-128 to 127) to 0-255 range
            array = (array + 128).astype(np.float32)
            # Then normalize to 0-1 range
            array = array / 255.0
        else:
            array[~np.isfinite(array)] = 0

        # Normalize to 0-255 range
        min_val = np.min(array)
        max_val = np.max(array)
        if max_val > min_val:
            normalized = (array - min_val) / (max_val - min_val)
        else:
            normalized = array

        img_uint8 = (normalized * 255).astype(np.uint8)

        # Buat gambar grayscale
        img = Image.fromarray(img_uint8, mode='L')

        # Simpan dengan kualitas maksimum
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', optimize=False, quality=100)
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        logger.error(f"Error dalam array_to_base64: {str(e)}")
        raise

def mask_to_base64(mask):
    try:
        # Create RGB image
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        # Define colors (RGB) dengan warna yang lebih kontras
        colors = {
            0: [0, 0, 0],        # Background/Water (Hitam)
            1: [0, 255, 255],    # Oil (Cyan)
            2: [255, 0, 0],      # Lookalikes/Current (Merah)
            3: [153, 76, 0],     # Antena Parabola (Coklat)
            4: [0, 153, 0]       # Land (Hijau)
        }
        
        # Terapkan warna dengan transparansi
        for class_idx, color in colors.items():
            mask_area = mask == class_idx
            colored_mask[mask_area] = color
            
        # Log distribusi kelas
        unique, counts = np.unique(mask, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Distribusi kelas: {class_dist}")
        
        # Konversi ke PIL Image dengan alpha channel
        img = Image.fromarray(colored_mask, 'RGB')
        
        # Simpan dengan kualitas maksimum
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', quality=100)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        logger.error(f"Error dalam mask_to_base64: {str(e)}")
        raise

# Mount static files - update the directory path and add html response
app.mount("/static", StaticFiles(directory="c:/Users/alvito/Documents/00. Capstone/webgis/static", html=True), name="static")

# Add root redirect
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

# Tambahkan endpoint baru untuk OpenOil
@app.post("/openoil_simulation")
async def openoil_simulation(
    use_detected_oil: bool = Form(False),
    latitude: float = Form(None),
    longitude: float = Form(None),
    oil_type: str = Form("SUMATRAN LIGHT"),
    radius: int = Form(3000),
    num_particles: int = Form(1000),
    duration_hours: int = Form(24),
    start_time: str = Form(None),
    oil_coordinates: str = Form(None)  # Parameter for detected oil coordinates
):
    try:
        # Initialize oil coordinates list
        oil_coordinates_list = []
        
        # Handle detected oil coordinates
        if use_detected_oil and oil_coordinates:
            try:
                # Log received coordinates for debugging
                logger.info(f"Received oil coordinates: {oil_coordinates[:100]}...")
                
                # Parse coordinates from JSON string
                parsed_coordinates = json.loads(oil_coordinates)
                
                # Log the number of coordinates received
                logger.info(f"Received {len(parsed_coordinates)} coordinates from frontend")
                
                # Validate and convert coordinates
                for coord in parsed_coordinates:
                    try:
                        if isinstance(coord, dict):
                            lat = float(coord.get('latitude'))
                            lon = float(coord.get('longitude'))
                        elif isinstance(coord, (list, tuple)) and len(coord) == 2:
                            lat = float(coord[0])
                            lon = float(coord[1])
                        else:
                            logger.warning(f"Invalid coordinate format: {coord}")
                            continue
                        
                        # Validate coordinate ranges
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            oil_coordinates_list.append({
                                'latitude': lat,
                                'longitude': lon
                            })
                        else:
                            logger.warning(f"Coordinates out of range: lat={lat}, lon={lon}")
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(f"Error processing coordinate {coord}: {e}")
                        continue
                
                # Log detailed information about the processed coordinates
                logger.info(f"Successfully parsed {len(oil_coordinates_list)} valid coordinates")
                if oil_coordinates_list:
                    logger.info(f"Sample coordinates (first 5): {oil_coordinates_list[:5]}")
                    logger.info(f"Coordinate range - Lat: [{min(c['latitude'] for c in oil_coordinates_list):.6f}, {max(c['latitude'] for c in oil_coordinates_list):.6f}], Lon: [{min(c['longitude'] for c in oil_coordinates_list):.6f}, {max(c['longitude'] for c in oil_coordinates_list):.6f}]")
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse oil coordinates JSON: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid oil coordinates format"
                )
        
        # If no valid coordinates from form, try app.state
        if use_detected_oil and not oil_coordinates_list and hasattr(app.state, 'oil_coordinates'):
            try:
                state_coordinates = app.state.oil_coordinates
                for coord in state_coordinates:
                    try:
                        if isinstance(coord, dict):
                            lat = float(coord.get('latitude'))
                            lon = float(coord.get('longitude'))
                        elif isinstance(coord, (list, tuple)) and len(coord) == 2:
                            lat = float(coord[0])
                            lon = float(coord[1])
                        else:
                            continue
                        
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            oil_coordinates_list.append({
                                'latitude': lat,
                                'longitude': lon
                            })
                    except (ValueError, TypeError, KeyError):
                        continue
                
                logger.info(f"Retrieved {len(oil_coordinates_list)} valid coordinates from app.state")
                
            except Exception as e:
                logger.error(f"Error processing app.state coordinates: {str(e)}")
        
        # If still no coordinates, use manual coordinates
        if not oil_coordinates_list:
            if latitude is None or longitude is None:
                raise HTTPException(
                    status_code=400,
                    detail="Latitude and longitude are required if no detected oil coordinates are available"
                )
            
            try:
                lat = float(latitude)
                lon = float(longitude)
                
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise HTTPException(
                        status_code=400,
                        detail="Coordinates out of valid range"
                    )
                
                oil_coordinates_list = [{
                    'latitude': lat,
                    'longitude': lon
                }]
                
            except (ValueError, TypeError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid coordinates: {str(e)}"
                )
        
        # Prepare request data for OpenOil service
        request_data = {
            'oil_type': oil_type,
            'radius': radius,
            'num_particles': num_particles,
            'duration_hours': duration_hours,
            'start_time': start_time # Explicitly include start_time even if None
        }
        
        # Add coordinates to request
        if oil_coordinates_list:
            request_data['oil_coordinates'] = oil_coordinates_list
        else:
            request_data['latitude'] = oil_coordinates_list[0]['latitude']
            request_data['longitude'] = oil_coordinates_list[0]['longitude']
        
        # Send request to OpenOil service
        openoil_service_url = "http://localhost:5001/simulate_openoil"
        response = requests.post(
            openoil_service_url,
            json=request_data,
            timeout=3600  # 60 minutes timeout
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenOil service error: {response.text}"
            )
        
        # Get result from OpenOil service
        result = response.json()
        
        # Add bounds for animation if available
        if "plots" in result and "animation" in result["plots"]:
            if "trajectory" in result:
                lats = []
                lons = []
                for time_point in result["trajectory"]:
                    for point in time_point["points"]:
                        lats.append(point["latitude"])
                        lons.append(point["longitude"])
                
                if lats and lons:
                    # Add 10% margin
                    lat_range = max(lats) - min(lats)
                    lon_range = max(lons) - min(lons)
                    margin_lat = lat_range * 0.1
                    margin_lon = lon_range * 0.1
                    
                    result["bounds"] = {
                        "north": max(lats) + margin_lat,
                        "south": min(lats) - margin_lat,
                        "east": max(lons) + margin_lon,
                        "west": min(lons) - margin_lon
                    }
        
        return result
        
    except requests.RequestException as e:
        logger.error(f"Error communicating with OpenOil service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to OpenOil service: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in OpenOil simulation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in OpenOil simulation: {str(e)}"
        )

@app.post("/mados_openoil_simulation")
async def mados_openoil_simulation(
    oil_type: str = Form("SUMATRAN LIGHT"),
    radius: int = Form(3000),
    num_particles: int = Form(1000),
    duration_hours: int = Form(24),
    start_time: str = Form(None),
    oil_coordinates: str = Form(None)  # JSON string of oil coordinates from MADOS
):
    """Run OpenOil simulation using oil coordinates from MADOS detection"""
    try:
        # Validate input
        if not oil_coordinates:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No oil coordinates provided"}
            )

        try:
            # Parse oil coordinates from JSON string
            oil_coords = json.loads(oil_coordinates)
            if not isinstance(oil_coords, list) or len(oil_coords) == 0:
                raise ValueError("Invalid oil coordinates format")
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid oil coordinates JSON format"}
            )

        # Prepare request data for OpenOil service
        request_data = {
            "oil_type": oil_type,
            "radius": radius,
            "num_particles": num_particles,
            "duration_hours": duration_hours,
            "start_time": start_time, # Explicitly include start_time even if None
            "oil_coordinates": oil_coords  # Pass the parsed coordinates
        }

        # Call OpenOil service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:5001/simulate_openoil",
                json=request_data,
                timeout=300.0  # 5 minutes timeout
            )

        if response.status_code != 200:
            return JSONResponse(
                status_code=response.status_code,
                content={"success": False, "error": f"OpenOil service error: {response.text}"}
            )

        result = response.json()
        
        if not result.get("success", False):
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": result.get("error", "Unknown error from OpenOil service")}
            )

        # Add metadata about the simulation
        result["metadata"] = {
            "source": "MADOS",
            "oil_type": oil_type,
            "num_particles": num_particles,
            "duration_hours": duration_hours,
            "num_oil_points": len(oil_coords)
        }

        return JSONResponse(content={"success": True, **result})

    except Exception as e:
        logger.error(f"Error in MADOS OpenOil simulation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error running MADOS OpenOil simulation: {str(e)}"}
        )

# Tambahkan endpoint untuk mendapatkan jenis minyak yang tersedia
@app.get("/available_oil_types")
async def get_available_oil_types():
    try:
        # Kirim request ke layanan OpenOil
        openoil_service_url = "http://localhost:5001/available_oil_types"
        response = requests.get(openoil_service_url, timeout=30)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenOil service error: {response.text}"
            )
        
        # Kembalikan hasil dari layanan OpenOil
        return response.json()
        
    except requests.RequestException as e:
        logger.error(f"Error communicating with OpenOil service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to OpenOil service: {str(e)}"
        )

# Replace the existing restart_mados_service function with our new implementation
def restart_mados_service():
    """Restart the MADOS service"""
    global mados_process
    
    # Terminate current process if it exists
    if mados_process:
        try:
            mados_process.terminate()
            mados_process.wait(timeout=5)
        except Exception:
            try:
                mados_process.kill()
            except Exception:
                pass
    
    # Kill any other instances
    kill_python_process_by_script(MADOS_SCRIPT)
    
    # Make sure port is free
    if is_port_in_use(MADOS_PORT):
        kill_process_on_port(MADOS_PORT)
    
    # Clean up
    clean_temp_dirs()
    clear_gpu_memory()
    
    # Start service
    return start_mados_service()

@app.post("/predict_mados")
async def predict_mados_proxy(
    files: List[UploadFile] = File(None), 
    file: UploadFile = File(None), 
    request: Request = None,
    preview: bool = Form(False),
    sequential_mode: str = Form(None),
    job_id: str = Form(None)
):
    """
    Proxy endpoint untuk meneruskan permintaan ke layanan MADOS
    
    Mendukung mode pemrosesan:
    - Standard: Upload semua band sekaligus (file zip atau 11 band TIFFs)
    - Preview: Mode cepat dengan downsampling untuk preview
    - Sequential: Upload dan proses band satu per satu
    """
    try:
        logger.info(f"Received MADOS request with preview={preview}, sequential_mode={sequential_mode}, job_id={job_id}")
        
        # Check if MADOS service is running
        if not is_port_in_use(MADOS_PORT):
            raise HTTPException(
                status_code=503,
                detail="MADOS service is not running. Please start the service manually."
            )
        
        # Use standard requests instead of httpx for more reliable file uploads
        import requests
        
        # Prepare form data
        form_data = {}
        
        # Add sequential processing parameters if provided
        if sequential_mode:
            form_data['sequential_mode'] = sequential_mode
            if job_id:
                form_data['job_id'] = job_id
                
        # Add preview flag if requested
        if preview:
            form_data['preview'] = 'true'
            
        # Create request to MADOS service
        mados_url = f"http://localhost:{MADOS_PORT}/predict_mados"
        
        # Configure timeouts based on mode
        if preview:
            timeout = 60.0  # 1 minute for preview
        elif sequential_mode == 'finish':
            timeout = 300.0  # 5 minutes for final processing
        else:
            timeout = 180.0  # 3 minutes for standard operations
            
        # Configure retry settings
        max_retries = 3
        retry_delay = 5  # seconds
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if file:
                    # Single file processing
                    file_content = await file.read()
                    files_dict = {'file': (file.filename, file_content, file.content_type)}
                    
                    response = requests.post(
                        mados_url,
                        files=files_dict,
                        data=form_data,
                        timeout=timeout
                    )
                elif files:
                    # Multiple files processing
                    files_dict = {}
                    for i, f in enumerate(files):
                        file_content = await f.read()
                        # Use 'files' as parameter name (not a list index)
                        files_dict[f'files'] = (f.filename, file_content, f.content_type)
                        
                    response = requests.post(
                        mados_url,
                        files=files_dict,
                        data=form_data,
                        timeout=timeout
                    )
                else:
                    # For finish mode, no files needed
                    if sequential_mode == 'finish' and job_id:
                        response = requests.post(
                            mados_url,
                            data=form_data,
                            timeout=timeout
                        )
                    else:
                        return {"status": "error", "message": "No files provided"}

                # Check response
                if response.status_code == 200:
                    try:
                        result = response.json()
                        logger.info(f"MADOS service response: {result.get('status', 'unknown')}")
                        return result
                    except Exception as e:
                        logger.error(f"Error parsing MADOS response: {str(e)}")
                        return {"status": "error", "message": f"Error parsing MADOS response: {str(e)}"}
                elif response.status_code == 503:
                    raise HTTPException(
                        status_code=503,
                        detail="MADOS service is not running. Please start the service manually."
                    )
                else:
                    # Other status codes
                    logger.error(f"MADOS service error: {response.status_code} - {response.text}")
                    return {"status": "error", "message": f"MADOS service error: {response.text}"}
                    
            except requests.Timeout:
                logger.error(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {"status": "error", "message": "Request timed out after multiple attempts"}
            except requests.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {"status": "error", "message": f"Request failed: {str(e)}"}
                
        # If we get here, all retries failed
        return {"status": "error", "message": f"All retry attempts failed. Last error: {last_error}"}
                
    except Exception as e:
        logger.error(f"MADOS proxy error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/mados_job_status/{job_id}")
async def get_mados_job_status(job_id: str):
    """Proxy to check MADOS job status"""
    try:
        # Check if MADOS service is running
        if not is_port_in_use(MADOS_PORT):
            raise HTTPException(
                status_code=503,
                detail="MADOS service is not running. Please start the service manually."
            )
        
        import requests
        
        # Forward to MADOS service
        response = requests.get(f"http://localhost:{MADOS_PORT}/mados_job_status/{job_id}", timeout=10)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to get job status: {response.text}"
            )
        
        return response.json()
        
    except requests.RequestException as e:
        logger.error(f"Error communicating with MADOS service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to MADOS service: {str(e)}"
        )

@app.get("/mados_job_result/{job_id}")
async def get_mados_job_result(job_id: str):
    """Proxy to get MADOS job result"""
    try:
        # Check if MADOS service is running
        if not is_port_in_use(MADOS_PORT):
            raise HTTPException(
                status_code=503,
                detail="MADOS service is not running. Please start the service manually."
            )
        
        import requests
        
        # Forward to MADOS service
        response = requests.get(f"http://localhost:{MADOS_PORT}/mados_job_result/{job_id}", timeout=30)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to get job result: {response.text}"
            )
        
        return response.json()
        
    except requests.RequestException as e:
        logger.error(f"Error communicating with MADOS service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to MADOS service: {str(e)}"
        )

@app.get("/clean_memory")
async def clean_memory():
    """Force memory cleanup on the MADOS service"""
    try:
        # Check if MADOS service is running
        if not is_port_in_use(MADOS_PORT):
            raise HTTPException(
                status_code=503,
                detail="MADOS service is not running. Please start the service manually."
            )
        
        import requests
        
        # Call MADOS clean memory endpoint
        response = requests.get(f"http://localhost:{MADOS_PORT}/clean_memory", timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"Failed to clean memory on MADOS service: {response.status_code}")
        else:
            logger.info("Successfully cleaned memory on MADOS service")
        
        return {"status": "success", "message": "Memory cleanup requested"}
        
    except Exception as e:
        logger.warning(f"Error requesting memory cleanup: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/receive_log")
async def receive_log(log: dict):
    level = log.get("level", "info")
    message = log.get("message", "")
    getattr(logger, level, logger.info)(f"[MADOS] {message}")
    return {"status": "ok"}