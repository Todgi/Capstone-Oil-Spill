import os
import sys
from datetime import datetime, timedelta
import subprocess
import importlib.util
import pkg_resources

def check_cmems_credentials():
    """Check if CMEMS credentials are properly configured"""
    username = os.getenv('CMEMS_USERNAME')
    password = os.getenv('CMEMS_PASSWORD')
    
    print("Checking CMEMS credentials...")
    print(f"Username found: {'Yes' if username else 'No'}")
    print(f"Password found: {'Yes' if password else 'No'}")
    
    if not username or not password:
        print("\nCMEMS credentials not found. Please set them using:")
        print("Command Prompt (cmd.exe):")
        print("set CMEMS_USERNAME=your_username")
        print("set CMEMS_PASSWORD=your_password")
        print("\nPowerShell:")
        print("$env:CMEMS_USERNAME='your_username'")
        print("$env:CMEMS_PASSWORD='your_password'")
        return False
    return True

def test_motu_installation():
    """Test if motuclient is properly installed"""
    print("\nTesting motuclient installation...")
    try:
        # Check if motuclient is installed
        spec = importlib.util.find_spec("motuclient")
        if spec is None:
            print("motuclient not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "motuclient"])
        
        # Get version using pkg_resources
        version = pkg_resources.get_distribution("motuclient").version
        print(f"motuclient version: {version}")
        
        # Try importing motuclient
        import motuclient
        print("motuclient module imported successfully")
        return True
    except Exception as e:
        print(f"Error testing motuclient: {str(e)}")
        print("\nPlease install motuclient using:")
        print("pip install motuclient")
        return False

def test_cmems_connection():
    """Test connection to CMEMS server"""
    print("\nTesting connection to CMEMS server...")
    try:
        # Try to get dataset information
        motu_command = [
            'motu-client',
            '--quiet',
            '--user', os.getenv('CMEMS_USERNAME'),
            '--pwd', os.getenv('CMEMS_PASSWORD'),
            '--motu', 'https://my.cmems-du.eu/motu-web/Motu',
            '--service-id', 'GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS',
            '--product-id', 'global-analysis-forecast-phy-001-024',
            '--longitude-min', '105',
            '--longitude-max', '108',
            '--latitude-min', '-8',
            '--latitude-max', '-5',
            '--date-min', (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'),
            '--date-max', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            '--depth-min', '0.493',
            '--depth-max', '0.4942',
            '--variable', 'uo',
            '--variable', 'vo',
            '--out-dir', '.',
            '--out-name', 'test_cmems.nc'
        ]
        
        print("Executing command:", ' '.join(motu_command))
        print("\nDownloading data... This may take a few minutes.")
        print("Progress: [", end='', flush=True)
        
        # Start the process
        process = subprocess.Popen(
            motu_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait for the process with timeout
        try:
            stdout, stderr = process.communicate(timeout=300)  # 5 minutes timeout
            
            if process.returncode == 0:
                print("] Done!")
                print("Successfully connected to CMEMS and downloaded test data")
                return True
            else:
                print("] Failed!")
                print(f"Error connecting to CMEMS: {stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            process.kill()
            print("] Timeout!")
            print("The download took too long. Please try again.")
            return False
            
    except Exception as e:
        print(f"Error testing CMEMS connection: {str(e)}")
        return False

if __name__ == "__main__":
    print("CMEMS Setup Verification Script")
    print("===============================")
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check credentials
    if not check_cmems_credentials():
        sys.exit(1)
    
    # Test motuclient installation
    if not test_motu_installation():
        sys.exit(1)
    
    # Test CMEMS connection
    if not test_cmems_connection():
        sys.exit(1)
    
    print("\nAll tests passed! CMEMS is properly configured.")
    print("You can now use the testHYCOM.py script to download ocean current data.") 