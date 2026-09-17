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
            # --- PATCH: Always add status: success if not present ---
            if 'status' not in result or result['status'] != 'success':
                result['status'] = 'success'
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
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Error getting job result: {str(e)}'
        }), 500 

def process_files_job(job_id, band_paths, band_names=None):
    try:
        import numpy as np
        import rasterio
        from PIL import Image
        import io, base64, gc
        # ... existing code for loading bands ...
        # After stacking bands:
        image = np.stack(current_image)
        image = np.moveaxis(image, (0, 1, 2), (2, 0, 1))
        img_height, img_width, _ = image.shape
        # ... prediction code ...
        # full_pred = ... (uint8 mask)
        # --- Generate RGB composite from 11 bands (optical) ---
        # Use Sentinel-2 convention: B04 (red, idx=3), B03 (green, idx=2), B02 (blue, idx=1)
        rgb_indices = [3, 2, 1]
        rgb_bands = []
        for idx in rgb_indices:
            if idx < image.shape[2]:
                rgb_bands.append(image[:,:,idx])
        if len(rgb_bands) == 3:
            rgb = np.stack(rgb_bands, axis=-1)
            # Stretch to 0-255
            min_val = np.percentile(rgb, 2)
            max_val = np.percentile(rgb, 98)
            rgb = np.clip((rgb - min_val) / (max_val - min_val + 1e-8) * 255, 0, 255).astype(np.uint8)
            rgb_img = Image.fromarray(rgb)
            buf2 = io.BytesIO()
            rgb_img.save(buf2, format='PNG')
            original_base64 = base64.b64encode(buf2.getvalue()).decode()
        else:
            original_base64 = None
        # --- Generate segmentation mask RGB ---
        # MADOS palette (from frontend legend)
        palette = [
            (0,0,128),      # Marine Water (navy)
            (255,0,0),      # Marine Debris (red)
            (0,128,0),      # Dense Sargassum (green)
            (50,205,50),    # Sparse Floating Algae (limegreen)
            (139,69,19),    # Natural Organic Material (brown)
            (255,140,0),    # Ship (orange)
            (216,191,216),  # Oil Spill (thistle)
            (255,215,0),    # Sediment-Laden Water (gold)
            (128,0,128),    # Foam (purple)
            (189,183,107),  # Turbid Water (darkkhaki)
            (0,206,209),    # Shallow Water (darkturquoise)
            (255,228,196),  # Waves & Wakes (bisque)
            (105,105,105),  # Oil Platform (dimgrey)
            (255,105,180),  # Jellyfish (hotpink)
            (255,255,0),    # Sea snot (yellow)
        ]
        colored_mask = np.zeros((*full_pred.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(palette):
            colored_mask[full_pred == class_idx] = color
        img = Image.fromarray(colored_mask, 'RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        segmentation_base64 = base64.b64encode(buf.getvalue()).decode()
        # --- Calculate bounds from transform ---
        # Use transform from first band
        with rasterio.open(band_paths[0]) as src:
            bounds = src.bounds
            transform = src.transform
        bounds_dict = {
            'south': float(bounds.bottom),
            'west': float(bounds.left),
            'north': float(bounds.top),
            'east': float(bounds.right)
        }
        # --- Oil coordinates (class 6 = Oil Spill) ---
        oil_coords = []
        oil_class_idx = 6
        if oil_class_idx in np.unique(full_pred):
            downsample_factor = 4
            downsampled_mask = full_pred[::downsample_factor, ::downsample_factor]
            y_idx, x_idx = np.where(downsampled_mask == oil_class_idx)
            y_idx = y_idx * downsample_factor
            x_idx = x_idx * downsample_factor
            max_coords = min(500, len(y_idx))
            if len(y_idx) > max_coords:
                indices = np.linspace(0, len(y_idx) - 1, max_coords, dtype=int)
                y_idx = y_idx[indices]
                x_idx = x_idx[indices]
            for y, x in zip(y_idx, x_idx):
                lon, lat = rasterio.transform.xy(transform, y, x)
                oil_coords.append([float(lat), float(lon)])
        # --- Save results to JSON ---
        result = {
            'status': 'success',
            'original_image': 'data:image/png;base64,' + original_base64 if original_base64 else None,
            'segmentation_image': 'data:image/png;base64,' + segmentation_base64,
            'oil_coordinates': oil_coords,
            'bounds': bounds_dict,
            'shape': full_pred.shape,
            'num_oil_pixels': len(oil_coords),
            'model_type': 'MariNext',
            'device_used': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        # Save to result.json
        import os, json
        job_dir = os.path.join(tempfile.gettempdir(), f"mados_job_{job_id}")
        result_file = os.path.join(job_dir, "result.json")
        with open(result_file, 'w') as f:
            json.dump(result, f)
        # Update final status
        update_job_status(job_id, {
            "status": "completed",
            "progress": 100,
            "message": "Prediction completed successfully"
        })
        return full_pred, transform, None
    except Exception as e:
        # ... existing error handling ...
        raise 