from flask import Flask, request, jsonify
import traceback
import warnings
import os
import sys
import io
import base64
import zipfile
import gc  # Add garbage collection
import uuid
import json
import threading
import tempfile
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import time
import platform
import logging
import requests
from datetime import datetime
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import hflip
from rasterio.warp import transform_bounds, transform as rio_transform
import pyproj

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mados_service.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("mados_service")

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Check if we're on Unix-like system where resource module is available
is_unix = platform.system() != "Windows"
if is_unix:
    import resource

# Global variables for model and configuration
model = None
bands_mean = None
bands_std = None
torch = None
np = None
rasterio = None
device = None  # Store device (CPU/GPU)

# Configuration
PATCH_SIZE = 240  # Fixed patch size as in predict.py
BATCH_SIZE = 1  # Default batch size for inference
TEST_TIME_AUGMENTATIONS = True  # Enable test time augmentations for better predictions
MODEL_EMA = True  # Use EMA model for better stability
MODEL_EMA_DECAY = 0.999  # EMA decay rate
MODEL_EMA_EVAL = True  # Use EMA for evaluation

# Training parameters (for reference)
WEIGHT_PARAM = 1.03  # Weighting parameter for Loss Function
LABEL_SMOOTHING = 0.0  # Label smoothing parameter
CLIP_GRAD = None  # Gradient clipping
LEARNING_RATE = 2e-4  # Learning rate
WEIGHT_DECAY = 0  # Weight decay
REDUCE_LR_ON_PLATEAU = 0  # Reduce learning rate on plateau
LR_STEPS = [45, 65]  # Learning rate reduction steps

# Evaluation parameters
RESULTS_PERCENTAGE = True  # Show results in percentage
PREDICT_MASKS = True  # Generate prediction masks

# Device configuration
USE_GPU = False  # Disable GPU by default to prevent memory issues
ENABLE_PROGRESSIVE_MODE = True  # Enable progressive processing mode
ENABLE_AGGRESSIVE_MEMORY_MANAGEMENT = True  # Enable even more aggressive memory management

# Data processing
MAX_OIL_COORDS = 2000  # Maximum number of oil coordinates to return

# Model configuration
INPUT_CHANNELS = 11  # Number of input bands as per mados-master standard
OUTPUT_CHANNELS = 15  # Number of output classes as per mados-master standard
MODEL_TYPE = 'MariNext'  # Model architecture type

def send_log_to_fastapi(message, level='info'):
    """Send log to FastAPI service with retry and timeout handling"""
    try:
        # Increase timeout to 5 seconds
        requests.post(
            "http://localhost:8000/receive_log",
            json={"level": level, "message": message},
            timeout=5
        )
    except requests.exceptions.RequestException as e:
        # Just print locally if FastAPI service is not available
        print(f"[MADOS] {level.upper()}: {message}")
        print(f"Failed to send log to FastAPI: {e}")

# Try to set memory limit for the process (works on Linux/Mac)
def limit_memory_usage():
    """Try to limit memory usage to prevent crashes"""
    if not is_unix:
        print("Memory limiting via resource module not available on Windows")
        return

    try:
        # On Linux/Mac - set soft limit to 90% of available RAM
        if hasattr(resource, 'RLIMIT_AS'):
            # Get system memory info
            import psutil
            mem_info = psutil.virtual_memory()
            total_memory = mem_info.total
            
            # Set limit to 90% of total memory
            memory_limit = int(total_memory * 0.9)
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            print(f"Memory limit set to {memory_limit / (1024**3):.2f} GB")
    except Exception as e:
        print(f"Could not set memory limit: {e}")

# Call this at startup
try:
    limit_memory_usage()
except:
    print("Failed to set memory limits")

def import_dependencies():
    """Import dependencies with error handling"""
    global torch, np, rasterio, device
    
    # Import basic dependencies
    try:
        import torch as torch_module
        torch = torch_module
        
        # Set device - use CUDA if available and enabled
        if USE_GPU and torch.cuda.is_available():
            device = torch.device('cuda')
            # Limit GPU memory usage
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(0.3, i)  # Reduce to 30% to prevent OOM errors
                torch.cuda.empty_cache()  # Clear GPU cache
        else:
            device = torch.device('cpu')
            
        print(f"✓ PyTorch imported successfully (using {device})")
    except ImportError as e:
        print(f"✗ Error importing PyTorch: {e}")
        return False
    
    try:
        import numpy as np_module
        np = np_module
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ Error importing NumPy: {e}")
        return False
    
    try:
        import rasterio as rasterio_module
        rasterio = rasterio_module
        print("✓ Rasterio imported successfully")
    except ImportError as e:
        print(f"✗ Error importing Rasterio: {e}")
        print("Note: Rasterio functionality will not be available")
        # We can still run without rasterio for basic testing
    
    return True

def load_model_with_fallback():
    """Load model with fallback for different MMCV versions"""
    global model, bands_mean, bands_std
    
    if torch is None or np is None:
        print("Required dependencies not loaded")
        return False
    
    try:
        # Add marinext to sys.path
        marinext_path = os.path.join(os.path.dirname(__file__), 'mados-master', 'marinext')
        if marinext_path not in sys.path:
            sys.path.insert(0, marinext_path)
        from marinext_wrapper import MariNext
        print("✓ MariNext imported successfully")
        
        # Initialize model
        model = MariNext(in_chans=11, num_classes=15)
        model.eval()  # Set to evaluation mode
        
        # Load weights
        weights_path = os.path.join(os.path.dirname(__file__), 'mados-master', 'marinext', 'trained_models', '2', 'marinext_2.pth')
        if os.path.exists(weights_path):
            try:
                # Use map_location to place tensors on the right device
                state_dict = torch.load(weights_path, map_location=device)
                # Convert all tensors to float32
                state_dict = {k: v.float() for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                # Move model to device and ensure float32
                model = model.to(device).float()
                print(f"✓ Model weights loaded from {weights_path} to {device}")
                
                # Use half precision if on GPU to save memory
                if device.type == 'cuda':
                    model = model.half()
                    print("✓ Using half precision (FP16) for GPU inference")
                
                # Optimize with TorchScript (significant performance improvement)
                dummy_input = torch.zeros((1, 11, PATCH_SIZE, PATCH_SIZE), 
                                       dtype=torch.float32,  # Always use float32 for tracing
                                       device=device)
                
                try:
                    # Trace the model with dummy input for optimization
                    traced_model = torch.jit.trace(model, dummy_input)
                    traced_model = torch.jit.freeze(traced_model)
                    # Replace model with optimized version
                    model = traced_model
                    print("✓ Model optimized with TorchScript")
                except Exception as e:
                    print(f"⚠ TorchScript optimization failed, using standard model: {e}")
                
                # Explicitly run garbage collection after model loading
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"✗ Error loading weights: {e}")
                return False
        else:
            print(f"✗ Model weights not found at {weights_path}")
            return False
            
        # Set normalization values (should match training)
        bands_mean = np.array([0.0582676, 0.05223386, 0.04381474, 0.0357083, 0.03412902, 0.03680401,
        0.03999107, 0.03566642, 0.03965081, 0.0267993, 0.01978944], dtype=np.float32)
        bands_std = np.array([0.03240627, 0.03432253, 0.0354812, 0.0375769, 0.03785412, 0.04992323,
        0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115], dtype=np.float32)
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return False

def preprocess_sentinel2(image_path):
    """Preprocess Sentinel-2 data with error handling"""
    if rasterio is None:
        print("Rasterio not available, creating dummy data")
        # Return dummy tensor for testing
        return torch.zeros((11, 100, 100), dtype=torch.float32)
    
    try:
        bands = []
        
        # Check if image_path is a file or directory pattern
        if isinstance(image_path, str) and '{band}' in image_path:
            # Handle band pattern (e.g., "image_B{band}.tif")
            for band in range(1, 12):  # 11 bands
                band_path = image_path.format(band=band)
                try:
                    with rasterio.open(band_path) as src:
                        band_data = src.read(1)
                        # Upsample if resolution > 10m
                        if src.res[0] > 10:
                            scale = src.res[0] / 10
                            band_data = src.read(
                                1,
                                out_shape=(int(band_data.shape[0] * scale),
                                          int(band_data.shape[1] * scale)),
                                resampling=rasterio.enums.Resampling.nearest
                            )
                        bands.append(band_data)
                except Exception as e:
                    print(f"Error reading band {band}: {e}")
                    # Create dummy band if reading fails
                    bands.append(np.zeros((100, 100), dtype=np.float32))
        else:
            # Handle single file with multiple bands
            try:
                with rasterio.open(image_path) as src:
                    for band in range(1, min(12, src.count + 1)):
                        band_data = src.read(band)
                        bands.append(band_data)
                    # Pad with zeros if less than 11 bands
                    while len(bands) < 11:
                        bands.append(np.zeros_like(bands[0]))
            except Exception as e:
                print(f"Error reading file {image_path}: {e}")
                # Create dummy data
                bands = [np.zeros((100, 100), dtype=np.float32) for _ in range(11)]
        
        # Stack and normalize
        stacked = np.stack(bands)
        
        # Apply normalization if available
        if bands_mean is not None and bands_std is not None:
            normalized = (stacked - bands_mean.reshape(-1, 1, 1)) / bands_std.reshape(-1, 1, 1)
        else:
            # Simple normalization to [0, 1]
            normalized = stacked / (stacked.max() + 1e-8)
        
        return torch.from_numpy(normalized).float()
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        traceback.print_exc()
        # Return dummy tensor
        return torch.zeros((11, 100, 100), dtype=torch.float32)

def pixel_to_coords(x, y, image_path):
    """Convert pixel coordinates to geographic coordinates"""
    if rasterio is None:
        return 0.0, 0.0
    
    try:
        with rasterio.open(image_path) as src:
            # Get the transform from the source image
            transform = src.transform
            # Convert pixel coordinates to geographic coordinates
            lon, lat = rasterio.transform.xy(transform, y, x)
            
            # Log the transformation for debugging
            logger.info(f"Pixel coordinates ({x}, {y}) transformed to geographic coordinates ({lat}, {lon})")
            logger.info(f"Using transform: {transform}")
            logger.info(f"Source CRS: {src.crs}")
            
            return float(lat), float(lon)
    except Exception as e:
        logger.error(f"Error converting coordinates: {e}")
        return 0.0, 0.0

# Initialize dependencies and model
print("=== MADOS Service Initialization ===")
print("Importing dependencies...")
if import_dependencies():
    print("Loading model...")
    if load_model_with_fallback():
        print("✓ Service initialized successfully")
        send_log_to_fastapi("Model loaded successfully", "info")
    else:
        print("✗ Model loading failed, but service will start anyway")
else:
    print("✗ Critical dependencies missing, service may not work properly")

@app.route('/predict_mados', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Monitor memory before processing
        monitor_memory_usage()
        
        import tempfile
        import shutil
        import rasterio
        import numpy as np
        from PIL import Image
        import io
        import base64
        import zipfile
        from rasterio.windows import Window
        import math
        import concurrent.futures
        from threading import Lock
        
        # Free memory before starting large process
        gc.collect()
        if torch is not None and device is not None and device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Create a lock for model inference to prevent race conditions 
        model_lock = Lock()
        
        # --- Band reference ---
        band_order = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        
        temp_dir = None
        image_path = None
        
        # Check for preview mode flag
        use_preview_mode = request.args.get('preview', 'false').lower() == 'true' or request.form.get('preview', 'false').lower() == 'true'
        
        # Check for sequential processing mode
        sequential_mode = request.form.get('sequential_mode', '')
        job_id = request.form.get('job_id', '')
        
        # Handle sequential processing modes
        if sequential_mode:
            if sequential_mode == 'start':
                # Create a new job
                job_id = str(uuid.uuid4())
                temp_dir = tempfile.mkdtemp(prefix=f"mados_job_{job_id}_")
                
                # Initialize a cache directory for bands
                bands_dir = os.path.join(temp_dir, "bands")
                os.makedirs(bands_dir, exist_ok=True)
                
                # Store the first band
                if 'file' in request.files:
                    file = request.files['file']
                    file_path = os.path.join(bands_dir, file.filename)
                    file.save(file_path)
                    
                    # Update job status
                    update_job_status(job_id, {
                        "status": "initialized", 
                        "stage": 1, 
                        "progress": 10,
                        "message": "First band saved",
                        "bands_saved": [file.filename],
                        "sequential_mode": True
                    })
                    
                    return jsonify({
                        'status': 'initialized',
                        'job_id': job_id,
                        'message': 'Sequential processing started, first band saved'
                    })
                else:
                    return jsonify({'status': 'error', 'message': 'No file provided'}), 400
                    
            elif sequential_mode == 'continue' and job_id:
                # Continue adding bands to existing job
                job_dir = os.path.join(tempfile.gettempdir(), f"mados_job_{job_id}")
                if not os.path.exists(job_dir):
                    return jsonify({'status': 'error', 'message': 'Invalid job ID'}), 400
                    
                bands_dir = os.path.join(job_dir, "bands")
                if not os.path.exists(bands_dir):
                    os.makedirs(bands_dir, exist_ok=True)
                
                # Add the next band
                if 'file' in request.files:
                    file = request.files['file']
                    file_path = os.path.join(bands_dir, file.filename)
                    file.save(file_path)
                    
                    # Read current status
                    status_file = os.path.join(job_dir, "status.json")
                    current_status = {}
                    if os.path.exists(status_file):
                        with open(status_file, 'r') as f:
                            current_status = json.load(f)
                    
                    # Update bands list
                    bands_saved = current_status.get('bands_saved', [])
                    bands_saved.append(file.filename)
                    
                    # Update job status
                    update_job_status(job_id, {
                        "status": "processing", 
                        "stage": 2, 
                        "progress": min(80, 10 + len(bands_saved) * 70 // 11),
                        "message": f"Added band {len(bands_saved)}/11",
                        "bands_saved": bands_saved,
                        "sequential_mode": True
                    })
                    
                    return jsonify({
                        'status': 'processing',
                        'job_id': job_id,
                        'bands_saved': len(bands_saved),
                        'message': f'Band {file.filename} added to job'
                    })
                else:
                    return jsonify({'status': 'error', 'message': 'No file provided'}), 400
                    
            elif sequential_mode == 'finish' and job_id:
                # Process all saved bands
                job_dir = os.path.join(tempfile.gettempdir(), f"mados_job_{job_id}")
                if not os.path.exists(job_dir):
                    return jsonify({'status': 'error', 'message': 'Invalid job ID'}), 400
                    
                bands_dir = os.path.join(job_dir, "bands")
                if not os.path.exists(bands_dir):
                    return jsonify({'status': 'error', 'message': 'No bands saved for this job'}), 400
                
                # Get all saved band files
                band_files = [os.path.join(bands_dir, f) for f in os.listdir(bands_dir) 
                             if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
                
                if not band_files:
                    return jsonify({'status': 'error', 'message': 'No band files found'}), 400
                
                # Update status to processing
                update_job_status(job_id, {
                    "status": "processing", 
                    "stage": 3, 
                    "progress": 85,
                    "message": "Processing all bands",
                    "sequential_mode": True
                })
                
                # Start processing in background
                threading.Thread(target=process_files_job, 
                                args=(job_id, band_files, band_order)).start()
                
                return jsonify({
                    'status': 'processing',
                    'job_id': job_id,
                    'message': 'Finalizing sequential processing',
                    'check_url': f'/mados_job_status/{job_id}'
                })
            else:
                return jsonify({'status': 'error', 'message': 'Invalid sequential mode or missing job ID'}), 400
        
        # Check for progressive mode flag
        use_progressive_mode = ENABLE_PROGRESSIVE_MODE and request.args.get('progressive', 'false').lower() == 'true'
        if use_progressive_mode:
            # Generate a job ID and return immediately
            job_id = str(uuid.uuid4())
            temp_dir = tempfile.mkdtemp(prefix=f"mados_job_{job_id}_")
            
            # Register job with watchdog
            if 'job_watchdog' in globals():
                job_watchdog.register_job(job_id)
            
            # Just store files and start processing in background
            if 'files' in request.files and len(request.files.getlist('files')) > 0:
                files = request.files.getlist('files')
                file_paths = []
                for f in files:
                    save_path = os.path.join(temp_dir, f.filename)
                    f.save(save_path)
                    file_paths.append(save_path)
                
                # Start processing in background
                update_job_status(job_id, {"status": "saved", "stage": 1, "progress": 10})
                threading.Thread(target=process_files_job, 
                                args=(job_id, file_paths, band_order)).start()
                
                return jsonify({
                    'status': 'processing',
                    'job_id': job_id,
                    'message': 'Files received, processing started',
                    'check_url': f'/mados_job_status/{job_id}'
                })
            elif 'file' in request.files:
                file = request.files['file']
                file_path = os.path.join(temp_dir, file.filename)
                file.save(file_path)
                
                # Start processing in background
                update_job_status(job_id, {"status": "saved", "stage": 1, "progress": 10})
                threading.Thread(target=process_file_job, 
                                args=(job_id, temp_dir, file_path)).start()
                
                return jsonify({
                    'status': 'processing',
                    'job_id': job_id,
                    'message': 'File received, processing started',
                    'check_url': f'/mados_job_status/{job_id}'
                })
            else:
                shutil.rmtree(temp_dir)
                return jsonify({'status': 'error', 'message': 'No files provided'}), 400
        else:
            # --- Robust multipart handling ---
            if 'files' in request.files and len(request.files.getlist('files')) > 0:
                files = request.files.getlist('files')
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                for f in files:
                    save_path = os.path.join(temp_dir, f.filename)
                    f.save(save_path)
                    file_paths.append(save_path)
                
                # Check if it's a preview request
                if use_preview_mode:
                    # Generate a job ID for preview processing
                    preview_job_id = str(uuid.uuid4())
                    preview_dir = os.path.join(temp_dir, "preview")
                    os.makedirs(preview_dir, exist_ok=True)
                    
                    # Start preview processing in background with aggressive downsampling
                    update_job_status(preview_job_id, {
                        "status": "preprocessing", 
                        "stage": 1, 
                        "progress": 20,
                        "message": "Generating preview",
                        "is_preview": True
                    })
                    
                    threading.Thread(target=process_preview_job, 
                                    args=(preview_job_id, preview_dir, file_paths)).start()
                    
                    return jsonify({
                        'status': 'processing',
                        'job_id': preview_job_id,
                        'message': 'Preview generation started',
                        'check_url': f'/mados_job_status/{preview_job_id}'
                    })
                
                # For regular processing, continue with the normal flow
                tif_dict = {os.path.basename(f).upper(): f for f in file_paths}
                available_bands = [b for b in band_order if b.upper() in tif_dict]
                
                # Process with available bands (even if incomplete)
                if available_bands:
                    image_path = [tif_dict.get(b.upper()) for b in available_bands]
                    image_path = [p for p in image_path if p is not None]
                    
                    # Warn if not all bands available
                    if len(image_path) < len(band_order):
                        print(f"Warning: Only {len(image_path)}/{len(band_order)} bands available")
                else:
                    shutil.rmtree(temp_dir)
                    return jsonify({'status': 'error', 'error_band': 'No valid band files found'}), 400
                    
            elif 'file' in request.files and request.files['file'].filename.lower().endswith('.zip'):
                file = request.files['file']
                temp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                tif_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
                
                # Check if it's a preview request
                if use_preview_mode:
                    # Generate a job ID for preview processing
                    preview_job_id = str(uuid.uuid4())
                    preview_dir = os.path.join(temp_dir, "preview")
                    os.makedirs(preview_dir, exist_ok=True)
                    
                    # Start preview processing in background with aggressive downsampling
                    update_job_status(preview_job_id, {
                        "status": "preprocessing", 
                        "stage": 1, 
                        "progress": 20,
                        "message": "Extracting zip and generating preview",
                        "is_preview": True
                    })
                    
                    threading.Thread(target=process_preview_job, 
                                    args=(preview_job_id, preview_dir, tif_files)).start()
                    
                    return jsonify({
                        'status': 'processing',
                        'job_id': preview_job_id,
                        'message': 'Preview generation started',
                        'check_url': f'/mados_job_status/{preview_job_id}'
                    })
                
                # For regular processing, continue with the normal flow
                tif_dict = {os.path.basename(f).upper(): f for f in tif_files}
                available_bands = [b for b in band_order if b.upper() in tif_dict]
                
                # Process with available bands (even if incomplete)
                if available_bands:
                    image_path = [tif_dict.get(b.upper()) for b in available_bands]
                    image_path = [p for p in image_path if p is not None]
                    
                    # Warn if not all bands available
                    if len(image_path) < len(band_order):
                        print(f"Warning: Only {len(image_path)}/{len(band_order)} bands available")
                else:
                    shutil.rmtree(temp_dir)
                    return jsonify({'status': 'error', 'error_band': 'No valid band files found in ZIP'}), 400
                    
            elif 'file' in request.files and (request.files['file'].filename.lower().endswith('.tif') or request.files['file'].filename.lower().endswith('.tiff')):
                file = request.files['file']
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
                file.save(temp_file.name)
                image_path = temp_file.name
            else:
                return jsonify({'status': 'error', 'message': 'No valid input files found. Please upload band TIFFs, a ZIP containing TIFFs, or a single multiband TIFF.'}), 400
        
        # ... rest of the function remains the same ...
        
        # --- Function to process a single tile ---
        def process_tile(tile_data):
            i, j, bands_tile = tile_data
            
            # Apply normalization
            if bands_mean is not None and bands_std is not None:
                bands_tile = (bands_tile - bands_mean.reshape(-1,1,1)) / bands_std.reshape(-1,1,1)
            
            # Convert to tensor and move to device
            if device.type == 'cuda':
                bands_tile_tensor = torch.from_numpy(bands_tile).unsqueeze(0).half().to(device)
            else:
                bands_tile_tensor = torch.from_numpy(bands_tile).unsqueeze(0).float().to(device)
            
            # Perform inference with the lock to prevent race conditions
            with model_lock:
                with torch.no_grad():
                    # Apply test time augmentations if enabled
                    if TEST_TIME_AUGMENTATIONS:
                        # Apply TTA transformations
                        tta_input = TTA(bands_tile_tensor)
                        logits = model(tta_input)
                        logits = F.upsample(input=logits, size=(PATCH_SIZE, PATCH_SIZE), mode='bilinear')
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        pred = probs.argmax(1)
                        # Reverse TTA transformations
                        pred = TTA(pred, reverse_aggregation=True)
                    else:
                        logits = model(bands_tile_tensor)
                        logits = F.upsample(input=logits, size=(PATCH_SIZE, PATCH_SIZE), mode='bilinear')
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        pred = probs.argmax(1)
                    
                    # Add 1 to match original implementation (classes start from 1)
                    pred = pred + 1
                    pred = pred.cpu().numpy()[0]
            
            # Clean up GPU memory
            if device.type == 'cuda':
                del bands_tile_tensor
                torch.cuda.empty_cache()
            
            return (i, j, pred)
        
        # --- Optimized tiling logic ---
        def tile_and_predict(band_paths):
            try:
                # Load and preprocess bands
                print("Loading and preprocessing bands...")
                send_log_to_fastapi("Loading and preprocessing bands...", "info")
                
                current_image = []
                for i, band in enumerate(band_paths):
                    print(f"Loading band {i+1}/{len(band_paths)}")
                    send_log_to_fastapi(f"Loading band {i+1}/{len(band_paths)}", "info")
                    with rasterio.open(band, mode='r') as src:
                        current_image.append(src.read(1))
                        if len(current_image) == 1:  # Get metadata from first band
                            crs = src.crs
                            transform = src.transform
                            meta = src.meta

                # Stack bands and move axis
                print("Stacking bands...")
                send_log_to_fastapi("Stacking bands...", "info")
                image = np.stack(current_image)
                
                # Ensure we have 11 channels by padding if necessary
                if image.shape[0] < 11:
                    print(f"Padded input from {image.shape[0]} to 11 channels")
                    send_log_to_fastapi(f"Padded input from {image.shape[0]} to 11 channels", "info")
                    padding = np.zeros((11 - image.shape[0], image.shape[1], image.shape[2]), dtype=image.dtype)
                    image = np.vstack([image, padding])
                
                # Convert to float32 to ensure compatibility
                image = image.astype(np.float32)
                
                # Move channels to last dimension
                image = np.moveaxis(image, 0, -1)
                
                # Get image dimensions
                img_height, img_width = image.shape[1:3]
                print(f"Image dimensions: {img_height}x{img_width}")
                send_log_to_fastapi(f"Image dimensions: {img_height}x{img_width}", "info")
                
                # Overlapping tile prediction (50% overlap)
                overlap = 0.0 #adjust overlap here
                stride = int(PATCH_SIZE * (1 - overlap))  # 50% overlap

                # Membuat tiles dengan overlap
                tiles = []
                for i in range(0, img_height, stride):
                    for j in range(0, img_width, stride):
                        y_end = min(i + PATCH_SIZE, img_height)
                        x_end = min(j + PATCH_SIZE, img_width)
                        tile = image[:, i:y_end, j:x_end]  # [C, h, w]
                        tiles.append((i, j, tile))

                # Untuk rekonstruksi dengan voting
                num_classes = 15  # Jumlah kelas MADOS
                prob_sum = np.zeros((num_classes, img_height, img_width), dtype=np.float32)
                count_map = np.zeros((img_height, img_width), dtype=np.uint8)

                # Prediksi per tile dan simpan ke list
                tile_preds = []
                for i, j, tile in tiles:
                    # Normalisasi tile
                    if bands_mean is not None and bands_std is not None:
                        tile = (tile - bands_mean[:, None, None]) / bands_std[:, None, None]
                    # Konversi ke tensor
                    tile_tensor = torch.from_numpy(tile).unsqueeze(0).float().to(device)
                    with torch.no_grad():
                        logits = model(tile_tensor)
                        logits = F.interpolate(logits, size=tile.shape[1:], mode='bilinear')
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [num_classes, h, w]
                    h, w = probs.shape[1:]
                    prob_sum[:, i:i+h, j:j+w] += probs
                    count_map[i:i+h, j:j+w] += 1

                # Rata-rata probabilitas
                prob_avg = prob_sum / np.maximum(count_map, 1)
                full_pred = np.argmax(prob_avg, axis=0) + 1  # kelas MADOS mulai dari 1
                
                print("Prediction completed successfully")
                send_log_to_fastapi("Prediction completed successfully", "info")
                
                return full_pred, transform, crs
                
            except Exception as e:
                print(f"Error in tile_and_predict: {str(e)}")
                send_log_to_fastapi(f"Error in tile_and_predict: {str(e)}", "error")
                raise
        
        # --- Main prediction ---
        if isinstance(image_path, list):
            # Multi-band, tile if needed
            with rasterio.open(image_path[0]) as src:
                img_width = src.width
                img_height = src.height
                
            if img_width > PATCH_SIZE * 2 or img_height > PATCH_SIZE * 2:
                mask, meta = tile_and_predict(image_path)
            else:
                # Single patch
                bands = []
                for band_path in image_path:
                    with rasterio.open(band_path) as src:
                        bands.append(src.read(1))
                bands = np.stack(bands)
                
                if bands_mean is not None and bands_std is not None:
                    bands = (bands - bands_mean.reshape(-1,1,1)) / bands_std.reshape(-1,1,1)
                
                # Convert to appropriate precision based on device
                if device.type == 'cuda':
                    bands_tensor = torch.from_numpy(bands).unsqueeze(0).half().to(device)
                else:
                    bands_tensor = torch.from_numpy(bands).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    pred = model(bands_tensor)
                    mask = torch.argmax(pred, dim=1).cpu().numpy()[0]
                
                with rasterio.open(image_path[0]) as src:
                    meta = src.meta.copy()
                    
                # Clean up
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
        else:
            # Single multi-band TIFF
            with rasterio.open(image_path) as src:
                img_width = src.width
                img_height = src.height
                count = src.count
                meta = src.meta.copy()
                
                if img_width > PATCH_SIZE or img_height > PATCH_SIZE:
                    # Use memory-mapped arrays for large images
                    band_paths = []
                    for b in range(count):
                        temp_band = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
                        # Use windowed reading/writing to save memory
                        with rasterio.open(temp_band.name, 'w', driver='GTiff', 
                                         height=img_height, width=img_width, 
                                         count=1, dtype=src.dtypes[b], 
                                         crs=src.crs, transform=src.transform) as dst:
                            for window in src.block_windows():
                                data = src.read(b+1, window=window)
                                dst.write(data, 1, window=window)
                        band_paths.append(temp_band.name)
                        
                    mask, meta = tile_and_predict(band_paths)
                    
                    # Clean up temporary files
                    for p in band_paths:
                        os.remove(p)
                else:
                    bands = src.read()
                    
                    if bands_mean is not None and bands_std is not None:
                        bands = (bands - bands_mean.reshape(-1,1,1)) / bands_std.reshape(-1,1,1)
                    
                    # Convert to appropriate precision based on device
                    if device.type == 'cuda':
                        bands_tensor = torch.from_numpy(bands).unsqueeze(0).half().to(device)
                    else:
                        bands_tensor = torch.from_numpy(bands).unsqueeze(0).float().to(device)
                    
                    with torch.no_grad():
                        pred = model(bands_tensor)
                        mask = torch.argmax(pred, dim=1).cpu().numpy()[0]
                    
                    # Clean up
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
            # --- Output mask as PNG base64 ---
            # Palette sesuai urutan label MADOS (assets.py)
            palette = [
                (255,0,0),      # 1. Marine Debris (red)
                (0,128,0),      # 2. Dense Sargassum (green)
                (50,205,50),    # 3. Sparse Floating Algae (limegreen)
                (139,69,19),    # 4. Natural Organic Material (brown)
                (255,140,0),    # 5. Ship (orange)
                (216,191,216),  # 6. Oil Spill (thistle)
                (0,0,128),      # 7. Marine Water (navy)
                (255,215,0),    # 8. Sediment-Laden Water (gold)
                (128,0,128),    # 9. Foam (purple)
                (189,183,107),  # 10. Turbid Water (darkkhaki)
                (0,206,209),    # 11. Shallow Water (darkturquoise)
                (255,228,196),  # 12. Waves & Wakes (bisque)
                (105,105,105),  # 13. Oil Platform (dimgrey)
                (255,105,180),  # 14. Jellyfish (hotpink)
                (255,255,0),    # 15. Sea snot (yellow)
            ]
            
            # Create RGB mask
            colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for class_idx, color in enumerate(palette):
                mask_area = mask == (class_idx + 1)  # kelas MADOS mulai dari 1
                colored_mask[mask_area] = color
            
            # Convert mask to base64
            mask_img = Image.fromarray(colored_mask)
            buf = io.BytesIO()
            mask_img.save(buf, format='PNG', optimize=True, compress_level=6)
            segmentation_base64 = base64.b64encode(buf.getvalue()).decode()
            
            # --- Output original image as RGB composite (bands 4,3,2 if available) ---
            if isinstance(image_path, list):
                # Load bands in a memory-efficient way
                rgb_bands = []
                band_indices = [3, 2, 1]  # 4th, 3rd, 2nd bands (0-indexed)
                
                for idx in band_indices:
                    if idx < len(image_path):
                        with rasterio.open(image_path[idx]) as src:
                            rgb_bands.append(src.read(1))
                            
                if len(rgb_bands) == 3:
                    rgb = np.stack(rgb_bands, axis=-1)
                    # Normalize and convert to 8-bit more efficiently
                    min_val = np.percentile(rgb, 2)  # Use percentile for better contrast
                    max_val = np.percentile(rgb, 98)
                    rgb = np.clip((rgb - min_val) / (max_val - min_val + 1e-8) * 255, 0, 255).astype(np.uint8)
                    
                    rgb_img = Image.fromarray(rgb)
                    buf2 = io.BytesIO()
                    rgb_img.save(buf2, format='PNG', optimize=True, compress_level=6)
                    original_base64 = base64.b64encode(buf2.getvalue()).decode()
                else:
                    original_base64 = None
            else:
                with rasterio.open(image_path) as src:
                    if src.count >= 4:
                        rgb_bands = [src.read(4), src.read(3), src.read(2)]
                        rgb = np.stack(rgb_bands, axis=-1)
                        
                        # Normalize and convert to 8-bit more efficiently
                        min_val = np.percentile(rgb, 2)  # Use percentile for better contrast
                        max_val = np.percentile(rgb, 98)
                        rgb = np.clip((rgb - min_val) / (max_val - min_val + 1e-8) * 255, 0, 255).astype(np.uint8)
                        
                        rgb_img = Image.fromarray(rgb)
                        buf2 = io.BytesIO()
                        rgb_img.save(buf2, format='PNG', optimize=True, compress_level=6)
                        original_base64 = base64.b64encode(buf2.getvalue()).decode()
                    else:
                        original_base64 = None
                    
            # --- Oil coordinates (class 6) - more efficient sampling ---
            oil_coords = []
            if 6 in np.unique(mask):
                # Use lower resolution to find pixels for efficiency
                downsample_factor_coords = 4
                downsampled_mask = mask[::downsample_factor_coords, ::downsample_factor_coords]
                y_idx, x_idx = np.where(downsampled_mask == 6)
                
                # Convert back to original resolution
                y_idx = y_idx * downsample_factor_coords
                x_idx = x_idx * downsample_factor_coords
                
                # If we have too many points, use a faster sampling method instead of KMeans
                max_coords = min(MAX_OIL_COORDS, len(y_idx))
                
                if len(y_idx) > max_coords:
                    # Use systematic sampling (much faster than KMeans)
                    # This evenly selects points from the full range
                    indices = np.linspace(0, len(y_idx) - 1, max_coords, dtype=int)
                    y_idx = y_idx[indices]
                    x_idx = x_idx[indices]
                
                # Store coordinates
                for y, x in zip(y_idx, x_idx):
                    oil_coords.append([float(y), float(x)])
                
            # --- PATCH: Tambahkan bounds ke result ---
            try:
                with rasterio.open(band_paths[0]) as src:
                    bounds_obj = src.bounds
                    src_crs = src.crs
                    transform_obj = src.transform

                    # Cek apakah CRS sudah EPSG:4326
                    if src_crs is not None and src_crs.to_string() not in ["EPSG:4326", "WGS84", "+proj=longlat +datum=WGS84 +no_defs"]:
                        # Reproject bounds ke EPSG:4326
                        bounds_wgs84 = transform_bounds(src_crs, 'EPSG:4326',
                            bounds_obj.left, bounds_obj.bottom, bounds_obj.right, bounds_obj.top, densify_pts=21)
                        bounds_dict = {
                            'south': float(bounds_wgs84[1]),
                            'west': float(bounds_wgs84[0]),
                            'north': float(bounds_wgs84[3]),
                            'east': float(bounds_wgs84[2])
                        }
                        # Reproject oil_coords ke EPSG:4326
                        if oil_coords:
                            xs = [c[1] for c in oil_coords]
                            ys = [c[0] for c in oil_coords]
                            lon, lat = rio_transform(src_crs, 'EPSG:4326', xs, ys)
                            oil_coords = [[float(la), float(lo)] for la, lo in zip(lat, lon)]
                    else:
                        bounds_dict = {
                            'south': float(bounds_obj.bottom),
                            'west': float(bounds_obj.left),
                            'north': float(bounds_obj.top),
                            'east': float(bounds_obj.right)
                        }
            except Exception as e:
                bounds_dict = None
                print(f"Failed to get bounds: {e}")
            
            # --- Cleanup temp files ---
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if isinstance(image_path, str) and image_path and os.path.exists(image_path) and image_path.endswith('.tif'):
                os.remove(image_path)
            
            # Final memory cleanup
            gc.collect()
            if torch is not None and device is not None and device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # --- Return response ---
            result = {
                'status': 'success',
                'segmentation_image': 'data:image/png;base64,' + segmentation_base64,
                'original_image': 'data:image/png;base64,' + original_base64 if original_base64 else None,
                'oil_coordinates': oil_coords,
                'bounds': bounds_dict,
                'shape': mask.shape,
                'num_oil_pixels': len(oil_coords),
                'model_type': 'MariNext',
                'device_used': device.type if device else 'unknown',
                'metadata': {
                    'crs': str(meta['crs']) if meta else None,
                    'transform': list(meta['transform']) if meta else None,
                    'test_time_augmentations': TEST_TIME_AUGMENTATIONS,
                    'model_ema': MODEL_EMA,
                    'model_ema_eval': MODEL_EMA_EVAL
                }
            }
            
            # Monitor memory after processing
            cleanup_memory()
            
            return jsonify(result)
    except Exception as e:
        print(f"Error in prediction: {e}")
        traceback.print_exc()
        cleanup_memory()  # Clean up memory even if there's an error
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Include memory stats for GPU if available
    gpu_info = {}
    if torch is not None and device is not None and device.type == 'cuda':
        for i in range(torch.cuda.device_count()):
            gpu_info[f'gpu_{i}'] = {
                'name': torch.cuda.get_device_name(i),
                'memory_allocated': f"{torch.cuda.memory_allocated(i) / 1024**2:.2f} MB",
                'memory_reserved': f"{torch.cuda.memory_reserved(i) / 1024**2:.2f} MB",
                'max_memory_allocated': f"{torch.cuda.max_memory_allocated(i) / 1024**2:.2f} MB"
            }
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device is not None else 'unknown',
        'torch_available': torch is not None,
        'numpy_available': np is not None,
        'rasterio_available': rasterio is not None,
        'gpu_info': gpu_info,
        'message': 'MADOS service is running'
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_loaded': model is not None,
        'model_type': 'MariNext' if model is not None else None,
        'device': str(device) if device is not None else 'unknown',
        'precision': 'float16 (half)' if device is not None and device.type == 'cuda' else 'float32',
        'input_channels': 11,
        'num_classes': 15,
        'bands_mean': bands_mean.tolist() if bands_mean is not None else None,
        'bands_std': bands_std.tolist() if bands_std is not None else None,
        'dependencies': {
            'torch': torch is not None,
            'numpy': np is not None,
            'rasterio': rasterio is not None
        }
    })

@app.route('/clean_memory', methods=['GET'])
def clean_memory():
    """Force cleanup of memory"""
    gc.collect()
    if torch is not None and device is not None and device.type == 'cuda':
        torch.cuda.empty_cache()
    return jsonify({
        'status': 'success',
        'message': 'Memory cleanup requested'
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Test endpoint working',
        'service': 'MADOS'
    })

@app.route('/predict_mados_pipeline', methods=['POST'])
def predict_pipeline():
    """Prediksi dengan pendekatan pipeline untuk mengurangi beban memori"""
    try:
        # TAHAP 1: Terima dan simpan file
        job_id = str(uuid.uuid4())
        temp_dir = tempfile.mkdtemp(prefix=f"mados_job_{job_id}_")
        
        # Simpan file input
        if 'file' in request.files:
            file_path = os.path.join(temp_dir, "input.tif")
            request.files['file'].save(file_path)
            # Tambahkan entry ke database/file status
            with open(os.path.join(temp_dir, "status.json"), "w") as f:
                json.dump({"status": "saved", "stage": 1, "progress": 10}, f)
            
            # Kembalikan response cepat dengan job_id
            return jsonify({
                'job_id': job_id,
                'status': 'processing',
                'message': 'File saved, starting preprocessing'
            })
            
            # Lanjutkan proses di thread terpisah
            threading.Thread(target=process_job, args=(job_id, temp_dir, file_path)).start()
        else:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400
                
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def process_job(job_id, temp_dir, file_path):
    """Proses job di background"""
    try:
        # TAHAP 2: Preprocess (simpan status)
        update_job_status(job_id, {"status": "preprocessing", "stage": 2, "progress": 30})
        
        # Preprocess image ke tile-tile kecil
        # ...
        
        # TAHAP 3: Prediksi per batch (simpan status tiap batch)
        update_job_status(job_id, {"status": "predicting", "stage": 3, "progress": 50})
        
        # Prediksi per batch dan simpan hasil sementara
        # ...
        
        # TAHAP 4: Merge hasil dan generasi output (simpan status)
        update_job_status(job_id, {"status": "finalizing", "stage": 4, "progress": 90})
        
        # Gabungkan hasil prediksi
        # ...
        
        # TAHAP 5: Selesai, simpan hasil akhir
        update_job_status(job_id, {"status": "completed", "stage": 5, "progress": 100})
        
    except Exception as e:
        update_job_status(job_id, {"status": "error", "error": str(e)})
        
# Endpoint untuk cek status
@app.route('/mados_job_status/<job_id>', methods=['GET'])
def check_job_status(job_id):
    # Baca status dari file/database
    status_file = os.path.join(tempfile.gettempdir(), f"mados_job_{job_id}", "status.json")
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            return jsonify(json.load(f))
    return jsonify({'status': 'not_found'}), 404

@app.route('/mados_job_result/<job_id>', methods=['GET'])
def get_job_result(job_id):
    """Get the result of a MADOS job"""
    try:
        # Get job directory
        job_dir = os.path.join(tempfile.gettempdir(), f"mados_job_{job_id}")
        print(f"Checking job result for {job_id} in directory: {job_dir}")
        
        if not os.path.exists(job_dir):
            print(f"Job directory not found: {job_dir}")
            return jsonify({
                'status': 'error',
                'message': 'Job directory not found'
            }), 404

        # Check status file
        status_file = os.path.join(job_dir, "status.json")
        if not os.path.exists(status_file):
            print(f"Status file not found: {status_file}")
            return jsonify({
                'status': 'error',
                'message': 'Status file not found'
            }), 404

        # Read status
        try:
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            print(f"Current job status: {status_data.get('status')}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in status file: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid status file format'
            }), 500

        # If job is still processing, return current status
        if status_data.get('status') == 'processing':
            progress = status_data.get('progress', 0)
            message = status_data.get('message', 'Processing')
            print(f"Job still processing: {message} (progress: {progress}%)")
            return jsonify({
                'status': 'processing',
                'message': message,
                'progress': progress
            }), 202

        # If job has error, return error message
        if status_data.get('status') == 'error':
            error_msg = status_data.get('message', 'Unknown error')
            print(f"Job {job_id} failed: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 500

        # Check result file
        result_file = os.path.join(job_dir, "result.json")
        if not os.path.exists(result_file):
            print(f"Result file not found: {result_file}")
            return jsonify({
                'status': 'error',
                'message': 'Result file not found'
            }), 404

        # Read result file
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            print(f"Successfully loaded result file for job {job_id}")
            return jsonify(result)
            
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in result file: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid result file format'
            }), 500
        except Exception as e:
            print(f"Error reading result file: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error reading result file: {str(e)}'
            }), 500

    except Exception as e:
        print(f"Error getting job result: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Error getting job result: {str(e)}'
        }), 500

def process_chunk(chunk_data, model_lock, device, bands_mean, bands_std):
    """Process a chunk of satellite image data
    
    Args:
        chunk_data: Tuple of (i, j, bands_tile) where i,j are tile coordinates
                   and bands_tile is the data to process
        model_lock: Lock for thread-safe model inference
        device: PyTorch device (CPU/GPU)
        bands_mean: Normalization mean values
        bands_std: Normalization std values
        
    Returns:
        Tuple of (i, j, prediction) for this chunk
    """
    import torch
    import gc
    import numpy as np
    
    i, j, bands_tile = chunk_data
    
    try:
        # Ensure input is float32
        bands_tile = bands_tile.astype(np.float32)
        
        # Apply normalization
        if bands_mean is not None and bands_std is not None:
            bands_mean = bands_mean.astype(np.float32)
            bands_std = bands_std.astype(np.float32)
            bands_tile = (bands_tile - bands_mean.reshape(-1,1,1)) / bands_std.reshape(-1,1,1)
        
        # Convert to tensor and move to device
        if device.type == 'cuda':
            bands_tile_tensor = torch.from_numpy(bands_tile).unsqueeze(0).half().to(device)
        else:
            bands_tile_tensor = torch.from_numpy(bands_tile).unsqueeze(0).float().to(device)
        
        # Perform inference with the lock to prevent race conditions
        with model_lock:
            with torch.no_grad():
                pred = model(bands_tile_tensor)
                pred = torch.argmax(pred, dim=1).cpu().numpy()[0]
        
        # Clean up GPU memory
        del bands_tile_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return (i, j, pred)
    except Exception as e:
        print(f"Error processing chunk {i},{j}: {str(e)}")
        # Return empty result in case of error
        return (i, j, np.zeros((bands_tile.shape[1], bands_tile.shape[2]), dtype=np.uint8))
    finally:
        # Ensure memory cleanup
        gc.collect()

def monitor_memory_usage():
    """Monitor and log memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
        
        if device and device.type == 'cuda':
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB")
    except ImportError:
        print("psutil not installed - memory monitoring disabled")
    except Exception as e:
        print(f"Error monitoring memory: {e}")

def cleanup_memory():
    """Aggressive memory cleanup with error handling"""
    try:
        gc.collect()
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
        monitor_memory_usage()
    except Exception as e:
        print(f"Error during memory cleanup: {e}")

def update_job_status(job_id, status_data):
    """Update job status and log to FastAPI"""
    try:
        # Create job directory if it doesn't exist
        job_dir = os.path.join(tempfile.gettempdir(), f"mados_job_{job_id}")
        os.makedirs(job_dir, exist_ok=True)
        
        # Save status to file
        status_file = os.path.join(job_dir, "status.json")
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
            
        # Log status update
        message = f"Job {job_id}: {status_data.get('status', 'unknown')} - {status_data.get('message', '')}"
        send_log_to_fastapi(message, status_data.get('status', 'info'))
        
    except Exception as e:
        error_msg = f"Error updating job status: {str(e)}"
        print(error_msg)
        send_log_to_fastapi(error_msg, "error")

def process_files_job(job_id, band_paths, band_names=None):
    try:
        print(f"Processing {len(band_paths)} bands...")
        send_log_to_fastapi(f"Processing {len(band_paths)} bands...", "info")
        
        # Create job directory at the start
        job_dir = os.path.join(tempfile.gettempdir(), f"mados_job_{job_id}")
        os.makedirs(job_dir, exist_ok=True)
        
        # Define transformations with correct number of bands
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Define standardization parameters for all bands
        bands_mean = np.array([0.0582676, 0.05223386, 0.04381474, 0.0357083, 0.03412902, 0.03680401,
        0.03999107, 0.03566642, 0.03965081, 0.0267993, 0.01978944], dtype=np.float32)
        bands_std = np.array([0.03240627, 0.03432253, 0.0354812, 0.0375769, 0.03785412, 0.04992323,
        0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115], dtype=np.float32)
        
        # Update initial status
        update_job_status(job_id, {
            "status": "processing",
            "progress": 0,
            "message": "Starting band processing..."
        })
        
        # Load and preprocess bands
        current_image = []
        total_bands = 11  # Always expect 11 bands
        
        for i in range(total_bands):
            progress = int((i / total_bands) * 30)  # First 30% for loading
            update_job_status(job_id, {
                "status": "processing",
                "progress": progress,
                "message": f"Loading band {i+1}/{total_bands}"
            })
            
            if i < len(band_paths):
                with rasterio.open(band_paths[i]) as src:
                    # Ensure float32 data type
                    band_data = src.read(1).astype(np.float32)
                    current_image.append(band_data)
                    if len(current_image) == 1:  # Get metadata from first band
                        crs = src.crs
                        transform = src.transform
                        meta = src.meta
            else:
                # Pad with zeros if band is missing
                if current_image:
                    current_image.append(np.zeros_like(current_image[0], dtype=np.float32))
                else:
                    # If no bands loaded yet, create a dummy band
                    with rasterio.open(band_paths[0]) as src:
                        current_image.append(np.zeros((src.height, src.width), dtype=np.float32))
                        crs = src.crs
                        transform = src.transform
                        meta = src.meta

        # Stack bands and move axis
        update_job_status(job_id, {
            "status": "processing",
            "progress": 30,
            "message": "Stacking bands..."
        })
        
        image = np.stack(current_image)
        
        # Ensure we have exactly 11 bands
        if image.shape[0] != 11:
            print(f"Warning: Input has {image.shape[0]} bands, padding to 11 bands")
            send_log_to_fastapi(f"Warning: Input has {image.shape[0]} bands, padding to 11 bands", "warning")
            
            # Create padded array
            padded_image = np.zeros((11, image.shape[1], image.shape[2]))
            padded_image[:image.shape[0], :, :] = image
            padded_image[image.shape[0]:, :, :] = np.array(bands_mean[image.shape[0]:]).reshape(1, 1, -1)
            image = padded_image

        # Get image dimensions
        img_height, img_width = image.shape[1:3]
        
        # Overlapping tile prediction (50% overlap)
        overlap = 0.5
        stride = int(PATCH_SIZE * (1 - overlap))  # 50% overlap

        # Membuat tiles dengan overlap
        tiles = []
        for i in range(0, img_height, stride):
            for j in range(0, img_width, stride):
                y_end = min(i + PATCH_SIZE, img_height)
                x_end = min(j + PATCH_SIZE, img_width)
                tile = image[:, i:y_end, j:x_end]  # [C, h, w]
                tiles.append((i, j, tile))

        # Untuk rekonstruksi dengan voting
        num_classes = 15  # Jumlah kelas MADOS
        prob_sum = np.zeros((num_classes, img_height, img_width), dtype=np.float32)
        count_map = np.zeros((img_height, img_width), dtype=np.uint8)

        # Prediksi per tile dan simpan ke list
        tile_preds = []
        for i, j, tile in tiles:
            # Normalisasi tile
            if bands_mean is not None and bands_std is not None:
                tile = (tile - bands_mean[:, None, None]) / bands_std[:, None, None]
            # Konversi ke tensor
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = model(tile_tensor)
                logits = F.interpolate(logits, size=tile.shape[1:], mode='bilinear')
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [num_classes, h, w]
            h, w = probs.shape[1:]
            prob_sum[:, i:i+h, j:j+w] += probs
            count_map[i:i+h, j:j+w] += 1

        # Rata-rata probabilitas
        prob_avg = prob_sum / np.maximum(count_map, 1)
        full_pred = np.argmax(prob_avg, axis=0) + 1  # kelas MADOS mulai dari 1
        
        # Save results
        update_job_status(job_id, {
            "status": "processing",
            "progress": 95,
            "message": "Saving results..."
        })
        
        # Convert prediction to RGB mask
        # Palette sesuai urutan label MADOS (assets.py)
        palette = [
            (255,0,0),      # 1. Marine Debris (red)
            (0,128,0),      # 2. Dense Sargassum (green)
            (50,205,50),    # 3. Sparse Floating Algae (limegreen)
            (139,69,19),    # 4. Natural Organic Material (brown)
            (255,140,0),    # 5. Ship (orange)
            (216,191,216),  # 6. Oil Spill (thistle)
            (0,0,128),      # 7. Marine Water (navy)
            (255,215,0),    # 8. Sediment-Laden Water (gold)
            (128,0,128),    # 9. Foam (purple)
            (189,183,107),  # 10. Turbid Water (darkkhaki)
            (0,206,209),    # 11. Shallow Water (darkturquoise)
            (255,228,196),  # 12. Waves & Wakes (bisque)
            (105,105,105),  # 13. Oil Platform (dimgrey)
            (255,105,180),  # 14. Jellyfish (hotpink)
            (255,255,0),    # 15. Sea snot (yellow)
        ]
        
        # Create RGB mask
        colored_mask = np.zeros((*full_pred.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(palette):
            mask_area = full_pred == (class_idx + 1)  # kelas MADOS mulai dari 1
            colored_mask[mask_area] = color
        
        # Convert to base64
        img = Image.fromarray(colored_mask, 'RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        segmentation_base64 = base64.b64encode(buf.getvalue()).decode()
        
        # Create RGB composite from original image
        rgb_bands = []
        band_indices = [3, 2, 1]  # 4th, 3rd, 2nd bands (0-indexed)
        
        for idx in band_indices:
            if idx < len(band_paths):
                with rasterio.open(band_paths[idx]) as src:
                    rgb_bands.append(src.read(1))
        
        if len(rgb_bands) == 3:
            rgb = np.stack(rgb_bands, axis=-1)
            min_val = np.percentile(rgb, 2)
            max_val = np.percentile(rgb, 98)
            rgb = np.clip((rgb - min_val) / (max_val - min_val + 1e-8) * 255, 0, 255).astype(np.uint8)
            
            rgb_img = Image.fromarray(rgb)
            buf2 = io.BytesIO()
            rgb_img.save(buf2, format='PNG')
            original_base64 = base64.b64encode(buf2.getvalue()).decode()
        else:
            original_base64 = None
        
        # Extract oil coordinates
        oil_coords = []
        if 6 in np.unique(full_pred):  # Assuming 6 is the oil class
            downsample_factor = 4
            downsampled_mask = full_pred[::downsample_factor, ::downsample_factor]
            y_idx, x_idx = np.where(downsampled_mask == 6)
            
            # Convert back to original resolution
            y_idx = y_idx * downsample_factor
            x_idx = x_idx * downsample_factor
            
            # Sample points if too many
            max_coords = min(2000, len(y_idx))
            if len(y_idx) > max_coords:
                indices = np.linspace(0, len(y_idx) - 1, max_coords, dtype=int)
                y_idx = y_idx[indices]
                x_idx = x_idx[indices]
            
            # Convert pixel coordinates to geographic coordinates using the first band's transform
            try:
                with rasterio.open(band_paths[0]) as src:
                    transform = src.transform
                    crs = src.crs
                    
                    # Log the transformation details
                    logger.info(f"Using transform from first band: {transform}")
                    logger.info(f"Source CRS: {crs}")
                    
                    # Create transformer for coordinate conversion
                    transformer = None
                    if crs is not None and crs.to_string() not in ["EPSG:4326", "WGS84", "+proj=longlat +datum=WGS84 +no_defs"]:
                        transformer = pyproj.Transformer.from_crs(crs.to_string(), "EPSG:4326", always_xy=True)
                    
                    for y, x in zip(y_idx, x_idx):
                        try:
                            # Convert pixel to geographic coordinates
                            lon, lat = rasterio.transform.xy(transform, y, x)
                            
                            # If CRS is not WGS84/EPSG:4326, transform coordinates
                            if transformer is not None:
                                lon, lat = transformer.transform(lon, lat)
                            
                            # Log each coordinate transformation
                            logger.info(f"Pixel ({x}, {y}) -> Geographic ({lat}, {lon})")
                            
                            # Append coordinates in [latitude, longitude] format
                            oil_coords.append([float(lat), float(lon)])
                        except Exception as e:
                            logger.error(f"Error converting coordinate ({x}, {y}): {e}")
                            continue
            except Exception as e:
                logger.error(f"Error accessing band file for coordinate transformation: {e}")
        
        # --- PATCH: Tambahkan bounds ke result ---
        try:
            with rasterio.open(band_paths[0]) as src:
                bounds_obj = src.bounds
                src_crs = src.crs
                transform_obj = src.transform

                # Cek apakah CRS sudah EPSG:4326
                if src_crs is not None and src_crs.to_string() not in ["EPSG:4326", "WGS84", "+proj=longlat +datum=WGS84 +no_defs"]:
                    # Reproject bounds ke EPSG:4326
                    bounds_wgs84 = transform_bounds(src_crs, 'EPSG:4326',
                        bounds_obj.left, bounds_obj.bottom, bounds_obj.right, bounds_obj.top, densify_pts=21)
                    bounds_dict = {
                        'south': float(bounds_wgs84[1]),
                        'west': float(bounds_wgs84[0]),
                        'north': float(bounds_wgs84[3]),
                        'east': float(bounds_wgs84[2])
                    }
                else:
                    bounds_dict = {
                        'south': float(bounds_obj.bottom),
                        'west': float(bounds_obj.left),
                        'north': float(bounds_obj.top),
                        'east': float(bounds_obj.right)
                    }
        except Exception as e:
            bounds_dict = None
            print(f"Failed to get bounds: {e}")
        
        # Save results to JSON
        result = {
            'status': 'success',
            'segmentation_image': 'data:image/png;base64,' + segmentation_base64,
            'original_image': 'data:image/png;base64,' + original_base64 if original_base64 else None,
            'oil_coordinates': oil_coords,
            'bounds': bounds_dict,
            'shape': full_pred.shape,
            'num_oil_pixels': len(oil_coords),
            'model_type': 'MariNext',
            'device_used': device.type if device else 'unknown',
            'metadata': {
                'crs': str(crs) if crs else None,
                'transform': list(transform) if transform else None,
                'test_time_augmentations': TEST_TIME_AUGMENTATIONS,
                'model_ema': MODEL_EMA,
                'model_ema_eval': MODEL_EMA_EVAL
            }
        }
        
        # Save to result.json with error handling
        try:
            result_file = os.path.join(job_dir, "result.json")
            with open(result_file, 'w') as f:
                json.dump(result, f)
            print(f"Successfully saved result to {result_file}")
            send_log_to_fastapi("Results saved successfully", "info")
        except Exception as e:
            error_msg = f"Error saving result file: {str(e)}"
            print(error_msg)
            send_log_to_fastapi(error_msg, "error")
            raise
        
        # Update final status
        update_job_status(job_id, {
            "status": "completed",
            "progress": 100,
            "message": "Prediction completed successfully"
        })
        
        return full_pred, transform, crs
        
    except Exception as e:
        error_msg = f"Error in process_files_job: {str(e)}"
        print(error_msg)
        send_log_to_fastapi(error_msg, "error")
        update_job_status(job_id, {
            "status": "error",
            "progress": 0,
            "message": error_msg
        })
        raise

def TTA(img, reverse_aggregation = False):
    """Test Time Augmentation implementation from MADOS
    
    Args:
        img: Input tensor
        reverse_aggregation: Whether to reverse the augmentation
        
    Returns:
        Augmented tensor or aggregated predictions
    """
    im_list = []
    
    if not reverse_aggregation:
        for k in [0,1,2,3]:
            im = torch.rot90(img, k=k, dims=[-2, -1])
            im_list.append(im)
            
            im = hflip(im)
            im_list.append(im)
            
        img = torch.cat(im_list)
        
    else:
        for k in [3,2,1,0]:
            im = hflip(img[k*2 + 1,:,:])
            im = torch.rot90(im, k=-k, dims=[-2, -1])
            im_list.append(im)
            
            im = torch.rot90(img[k*2,:,:], k=-k, dims=[-2, -1])
            im_list.append(im)

        img = torch.stack(im_list)
        img = torch.mode(img, dim=0, keepdim=True)[0]
    
    return img

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=5000, debug=True)

# Add signal handlers for graceful shutdown
import signal

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\nShutting down MADOS service...")
    cleanup_memory()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)