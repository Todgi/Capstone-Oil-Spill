from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import json
import numpy as np
import os
import tempfile
import base64
from io import BytesIO
from PIL import Image
import logging
import time
import requests
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openoil_service.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("openoil_service")

# Import OpenDrift libraries
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.openoil import OpenOil

app = Flask(__name__)

# Rate limiting configuration
RATE_LIMIT = 10  # requests per minute
last_request_time = {}

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        current_time = time.time()
        if f.__name__ in last_request_time:
            time_diff = current_time - last_request_time[f.__name__]
            if time_diff < 60/RATE_LIMIT:  # 60 seconds / RATE_LIMIT
                time.sleep(60/RATE_LIMIT - time_diff)
        last_request_time[f.__name__] = current_time
        return f(*args, **kwargs)
    return decorated_function

# Helper function to convert numpy arrays to lists for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

# Custom reader class with proper headers
class CustomReader(reader_netCDF_CF_generic.Reader):
    def __init__(self, url, **kwargs):
        self.headers = {
            'User-Agent': 'OpenOil-Simulation/1.0',
            'Accept': 'application/x-netcdf',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        super().__init__(url, **kwargs)
    
    def _get_dataset(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error accessing {url}: {str(e)}")
            raise

@app.route('/available_oil_types', methods=['GET'])
def get_oil_types():
    """Return available oil types"""
    o = OpenOil(loglevel=20, location='Indonesia')
    return jsonify({
        'success': True,
        'oil_types': list(o.oiltypes)
    })

@app.route('/simulate_openoil', methods=['POST'])
@rate_limit
def simulate_openoil():
    """Run OpenOil simulation with parameters from request"""
    try:
        data = request.json
        logger.info(f"Received OpenOil request data: {data}")
        
        # Extract parameters
        oil_type = data.get('oil_type', 'SUMATRAN LIGHT')
        radius = data.get('radius', 3000)
        num_particles = data.get('num_particles', 1000)
        duration_hours = data.get('duration_hours', 24)
        start_time = data.get('start_time')
        
        # Initialize OpenOil
        o = OpenOil(loglevel=20, location='Indonesia')
        
        # Add readers with proper error handling
        try:
            reader_ncep = CustomReader('https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/ncep_global/NCEP_Global_Atmospheric_Model_best.ncd')
        except Exception as e:
            logger.warning(f"Error with primary NCEP URL: {e}")
            try:
                reader_ncep = CustomReader('https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ncep_global')
            except Exception as e2:
                logger.warning(f"Error with alternative NCEP URL 1: {e2}")
                reader_ncep = CustomReader('https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/FMRC/GLBy0.08_930_FMRC_best.ncd')
        
        try:
            reader_hycom = CustomReader('https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd')
        except Exception as e:
            logger.error(f"Error accessing HYCOM data: {e}")
            return jsonify({
                'success': False,
                'error': "Unable to access ocean current data. Please try again later."
            }), 503
        
        o.add_reader([reader_hycom, reader_ncep])
        
        # Configure
        o.set_config('processes:evaporation', True)
        o.set_config('processes:emulsification', True)
        o.set_config('drift:vertical_mixing', True)
        o.set_config('vertical_mixing:timestep', 5)
        
        # Set start time
        if not start_time:
            return jsonify({
                'success': False,
                'error': "Start time is required for the simulation"
            }), 400

        try:
            # Try parsing various common formats
            date_formats_to_try = [
                '%Y-%m-%dT%H:%M:%SZ', # ISO 8601 with Z
                '%Y-%m-%dT%H:%M:%S',  # ISO 8601 without Z
                '%Y-%m-%dT%H:%M',    # YYYY-MM-DDTHH:MM (from datetime-local input)
                '%Y-%m-%d %H:%M:%S',  # YYYY-MM-DD HH:MM:SS
                '%m/%d/%Y %H:%M',    # MM/DD/YYYY HH:MM
                '%d/%m/%Y %H:%M',    # DD/MM/YYYY HH:MM
                '%Y-%m-%d',          # YYYY-MM-DD (Date only)
                '%m/%d/%Y',          # MM/DD/YYYY (Date only)
                '%d/%m/%Y'           # DD/MM/YYYY (Date only)
            ]
            
            parsed_time = None
            for fmt in date_formats_to_try:
                try:
                    parsed_time = datetime.strptime(start_time, fmt)
                    logger.info(f"Successfully parsed start time '{start_time}' using format: {fmt}")
                    break # Stop after successful parse
                except ValueError:
                    continue # Try next format
            
            if parsed_time is None:
                return jsonify({
                    'success': False,
                    'error': f"Could not parse start time '{start_time}'. Please use a valid date format."
                }), 400
            
            # Ensure the datetime object is timezone-naive for consistency with OpenDrift
            if parsed_time.tzinfo is not None:
                parsed_time = parsed_time.replace(tzinfo=None)

            # Validate that the start time is not in the future
            current_time = datetime.now()
            if parsed_time > current_time:
                return jsonify({
                    'success': False,
                    'error': f"Start time '{start_time}' cannot be in the future"
                }), 400

            time = parsed_time

        except Exception as e:
            logger.error(f"Error parsing start time '{start_time}': {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Invalid start time format: {str(e)}"
            }), 400
        
        # Handle oil coordinates
        if 'oil_coordinates' in data and data['oil_coordinates']:
            oil_coordinates = data['oil_coordinates']
            
            if not oil_coordinates:
                logger.error("Received empty oil_coordinates list")
                return jsonify({
                    'success': False,
                    'error': "No valid detected oil coordinates provided."
                }), 400

            valid_coordinates = []
            
            # Validate and convert coordinates
            for coord in oil_coordinates:
                try:
                    # Handle different coordinate formats
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
                        valid_coordinates.append((lat, lon))
                    else:
                        logger.warning(f"Coordinates out of range: lat={lat}, lon={lon}")
                        continue

                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Error processing coordinate {coord}: {e}")
                    continue
            
            if not valid_coordinates:
                logger.error("No valid coordinates found after processing the list.")
                raise ValueError("No valid coordinates found in the input data")
            
            # Distribute particles among valid coordinates
            particles_per_point = max(1, num_particles // len(valid_coordinates))
            remaining_particles = num_particles
            
            for lat, lon in valid_coordinates:
                # Calculate particles for this point
                point_particles = min(particles_per_point, remaining_particles)
                remaining_particles -= point_particles
                
                if point_particles > 0:
                    # Seed particles at this point
                    o.seed_elements(
                        lon=lon, 
                        lat=lat, 
                        radius=radius // 2,  # Smaller radius for multiple points
                        number=point_particles,
                        time=time, 
                        z=0, 
                        oil_type=oil_type
                    )
                
                if remaining_particles <= 0:
                    break
        else:
            # Use single coordinate
            try:
                latitude = float(data.get('latitude', -5.082163))
                longitude = float(data.get('longitude', 106.192765))
                
                # Validate coordinate ranges
                if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                    raise ValueError("Coordinates out of valid range")
                
                o.seed_elements(
                    lon=longitude, 
                    lat=latitude, 
                    radius=radius, 
                    number=num_particles,
                    time=time, 
                    z=0, 
                    oil_type=oil_type
                )
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid coordinates: {str(e)}")
        
        # Run simulation
        time_step_seconds = 900  # 15 minutes
        time_step_output = 900  # 15 minutes
        steps = int((duration_hours * 3600) / time_step_seconds)
        o.run(steps=steps, time_step=time_step_seconds, time_step_output=time_step_output)
        
        # Extract trajectory data
        lons = o.get_property('lon')[0]
        lats = o.get_property('lat')[0]
        depths = o.get_property('z')[0]
        status = o.get_property('status')[0]
        
        # Convert to numpy arrays and ensure float type
        lons = np.array(lons, dtype=float)
        lats = np.array(lats, dtype=float)
        
        # Create trajectory points
        trajectory_points = []
        for i in range(len(lons)):
            points_at_time = []
            for j in range(len(lons[i])):
                if status[i][j] == 0:  # Only include active particles
                    try:
                        lon = float(lons[i][j])
                        lat = float(lats[i][j])
                        depth = float(depths[i][j])
                        
                        # Validate coordinates
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            points_at_time.append({
                                'longitude': lon,
                                'latitude': lat,
                                'depth': depth
                            })
                    except (ValueError, TypeError):
                        continue
            
            if points_at_time:  # Only add time points that have valid coordinates
                trajectory_points.append({
                    'time': (time + timedelta(seconds=i*time_step_output)).isoformat(),
                    'points': points_at_time
                })
        
        # Generate plot images
        plot_images = {}
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Oil budget plot
            budget_file = os.path.join(tmpdirname, 'oil_budget.png')
            o.plot_oil_budget(filename=budget_file)
            with open(budget_file, 'rb') as f:
                plot_images['oil_budget'] = base64.b64encode(f.read()).decode('utf-8')
            
            # Trajectory plot
            trajectory_file = os.path.join(tmpdirname, 'trajectory.png')
            o.plot(filename=trajectory_file)
            with open(trajectory_file, 'rb') as f:
                plot_images['trajectory'] = base64.b64encode(f.read()).decode('utf-8')
            
            # Animation GIF
            animation_file = os.path.join(tmpdirname, 'animation.gif')
            o.animation(filename=animation_file, fps=10, color='status', time_step_output=1800)
            with open(animation_file, 'rb') as f:
                plot_images['animation'] = base64.b64encode(f.read()).decode('utf-8')
            
            # Calculate bounds from valid coordinates
            valid_lons = lons[~np.isnan(lons)]
            valid_lats = lats[~np.isnan(lats)]
            
            if len(valid_lons) > 0 and len(valid_lats) > 0:
                lon_min = float(np.nanmin(valid_lons))
                lon_max = float(np.nanmax(valid_lons))
                lat_min = float(np.nanmin(valid_lats))
                lat_max = float(np.nanmax(valid_lats))
                
                # Add margins
                lon_margin = (lon_max - lon_min) * 0.1
                lat_margin = (lat_max - lat_min) * 0.1
                
                plot_images['animation_bounds'] = {
                    'south': lat_min - lat_margin,
                    'north': lat_max + lat_margin,
                    'west': lon_min - lon_margin,
                    'east': lon_max + lon_margin
                }
        
        return jsonify({
            'success': True,
            'trajectory': trajectory_points,
            'plots': plot_images
        })
        
    except Exception as e:
        logger.error(f"OpenOil simulation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)