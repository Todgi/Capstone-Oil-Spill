import xarray as xr
import netCDF4
import sys
import requests
from datetime import datetime, timedelta
import os
import subprocess
import tempfile
import configparser
import logging
import urllib.parse
from bs4 import BeautifulSoup
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hycom_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("hycom_test")

def create_session_with_retries():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_hycom_dataset_urls():
    """Get available HYCOM dataset URLs from the THREDDS catalog"""
    try:
        session = create_session_with_retries()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }

        # List of catalog URLs to try
        catalog_urls = [
            'https://tds.hycom.org/thredds/catalog/GLBy0.08/expt_93.0/catalog.html',
            'https://tds.hycom.org/thredds/catalog/GLBu0.08/expt_93.0/catalog.html'
        ]

        dataset_urls = []
        
        # Try each catalog URL with delay between attempts
        for catalog_url in catalog_urls:
            try:
                print(f"Trying catalog: {catalog_url}")
                # Add delay between requests to avoid rate limiting
                time.sleep(2)
                
                response = session.get(catalog_url, headers=headers, timeout=30)
                if response.status_code == 200:
                    print(f"Successfully accessed catalog: {catalog_url}")
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for links containing dataset information
                    for link in soup.find_all('a'):
                        href = link.get('href', '')
                        if 'best.ncd' in href:
                            # Convert to OPeNDAP URL
                            dods_url = href.replace('catalog.html', 'dodsC/')
                            if dods_url.startswith('/'):
                                dods_url = 'https://tds.hycom.org' + dods_url
                            dataset_urls.append(dods_url)
                            print(f"Found dataset: {dods_url}")
                    
                    if dataset_urls:
                        break
            except Exception as e:
                print(f"Error accessing catalog {catalog_url}: {str(e)}")
                continue

        if not dataset_urls:
            print("No datasets found in catalogs, trying direct URLs...")
            # Use specific URLs for the latest data
            dataset_urls = [
                'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/GLBy0.08_930_best.ncd',
                'https://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_93.0/GLBu0.08_930_best.ncd'
            ]
        
        return dataset_urls
    except Exception as e:
        print(f"Error getting dataset URLs: {str(e)}")
        return []

def test_hycom_connection():
    """Test connection to HYCOM server"""
    print("\nTesting connection to HYCOM server...")
    try:
        # Get available dataset URLs
        print("Getting available HYCOM datasets...")
        hycom_urls = get_hycom_dataset_urls()
        
        if not hycom_urls:
            print("No HYCOM datasets found")
            return False
            
        print(f"Found {len(hycom_urls)} datasets")
        
        # Test coordinates (Java Sea region)
        test_lon = 105.0
        test_lat = -5.0
        test_time = datetime.utcnow()
        
        # Set up headers for OPeNDAP access
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/x-netcdf',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        session = create_session_with_retries()
        success = False
        
        for url in hycom_urls:
            try:
                print(f"\nTrying HYCOM URL: {url}")
                
                # Add delay between requests to avoid rate limiting
                time.sleep(2)
                
                # First check if the dataset is accessible
                response = session.head(url, headers=headers, timeout=30)
                if response.status_code != 200:
                    print(f"Dataset not accessible: {response.status_code}")
                    continue
                
                print("Dataset is accessible, trying to open...")
                
                # Try to open the dataset directly with xarray
                ds = xr.open_dataset(
                    url,
                    engine='netcdf4',
                    decode_times=True,
                    decode_coords=True,
                    mask_and_scale=True,
                    chunks={'time': 1}  # Use chunking for better memory management
                )
                print("Successfully opened dataset")
                
                # Check if required variables exist
                required_vars = ['water_u', 'water_v']
                missing_vars = [var for var in required_vars if var not in ds.variables]
                
                if missing_vars:
                    print(f"Missing required variables: {missing_vars}")
                    continue
                
                # Try to get a small subset of data
                print("Testing data access...")
                data = ds.sel(
                    lon=test_lon,
                    lat=test_lat,
                    time=test_time,
                    method='nearest'
                )
                
                # Check if we got valid data
                if data is not None and len(data) > 0:
                    print("Successfully retrieved data")
                    print("\nSample data:")
                    print(f"U velocity: {data.water_u.values}")
                    print(f"V velocity: {data.water_v.values}")
                    success = True
                    break
                else:
                    print("No data returned")
                    
            except Exception as e:
                print(f"Error with URL {url}: {str(e)}")
                continue
        
        if success:
            print("\nHYCOM connection test successful!")
            return True
        else:
            print("\nAll HYCOM URLs failed")
            return False
            
    except Exception as e:
        print(f"Error testing HYCOM connection: {str(e)}")
        return False

def test_hycom_data_download():
    """Test downloading a small subset of HYCOM data"""
    print("\nTesting HYCOM data download...")
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, 'hycom_test.nc')
        
        # Test coordinates (Java Sea region)
        lon_min, lon_max = 104, 106
        lat_min, lat_max = -6, -4
        time = datetime.utcnow()
        
        # Get available dataset URLs
        hycom_urls = get_hycom_dataset_urls()
        if not hycom_urls:
            raise Exception("No HYCOM datasets found")
            
        # Use the first available URL
        url = hycom_urls[0]
        
        print(f"Downloading data from: {url}")
        
        # Set up headers for OPeNDAP access
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/x-netcdf',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        # Try to open the dataset directly with xarray
        ds = xr.open_dataset(
            url,
            engine='netcdf4',
            decode_times=True,
            decode_coords=True,
            mask_and_scale=True,
            chunks={'time': 1}  # Use chunking for better memory management
        )
        
        # Select small subset
        data = ds.sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max),
            time=time,
            method='nearest'
        )
        
        # Save to file
        data.to_netcdf(output_file)
        print(f"Successfully downloaded and saved data to: {output_file}")
        
        # Try to read the saved file
        test_ds = xr.open_dataset(output_file)
        print("\nVerifying downloaded data:")
        print(f"Variables: {list(test_ds.variables)}")
        print(f"Dimensions: {dict(test_ds.dims)}")
        
        return True
        
    except Exception as e:
        print(f"Error testing HYCOM download: {str(e)}")
        return False
    finally:
        # Clean up temporary directory
        if 'temp_dir' in locals():
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass

if __name__ == "__main__":
    print("HYCOM Connection Test")
    print("====================")
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    print(f"xarray version: {xr.__version__}")
    print(f"netCDF4 version: {netCDF4.__version__}")
    
    # Test connection
    if not test_hycom_connection():
        print("\nConnection test failed. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Test data download
    if not test_hycom_data_download():
        print("\nDownload test failed. Please check your internet connection and try again.")
        sys.exit(1)
    
    print("\nAll tests passed! HYCOM is accessible and working correctly.")
