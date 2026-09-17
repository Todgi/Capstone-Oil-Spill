import os
import sys
import time
import subprocess
import psutil
import requests
import shutil
import tempfile

def clean_temp_dirs():
    """Clean temporary directories"""
    print("Cleaning temporary directories...")
    temp_dir = tempfile.gettempdir()
    print(f"Temp directory: {temp_dir}")
    
    count = 0
    for item in os.listdir(temp_dir):
        if item.startswith("mados_job_"):
            item_path = os.path.join(temp_dir, item)
            if os.path.isdir(item_path):
                try:
                    print(f"Removing: {item_path}")
                    shutil.rmtree(item_path)
                    count += 1
                except Exception as e:
                    print(f"Error removing {item_path}: {e}")
    
    print(f"Removed {count} temporary directories")

def clear_gpu_memory():
    """Clear GPU memory if available"""
    try:
        import torch
        if torch.cuda.is_available():
            print("Clearing CUDA memory...")
            # Get initial memory usage
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                print(f"GPU {i} before cleanup: allocated={allocated:.2f}MB, reserved={reserved:.2f}MB")
            
            # Clear memory
            torch.cuda.empty_cache()
            
            # Check memory after cleanup
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                print(f"GPU {i} after cleanup: allocated={allocated:.2f}MB, reserved={reserved:.2f}MB")
        else:
            print("CUDA not available")
    except ImportError:
        print("PyTorch not installed")

def kill_port_processes(port):
    """Kill processes using the specified port"""
    print(f"Checking for processes using port {port}...")
    
    try:
        # For Windows
        if os.name == 'nt':
            output = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()
            if output:
                for line in output.split('\n'):
                    if f':{port}' in line and 'LISTENING' in line:
                        pid = line.strip().split()[-1]
                        print(f"Killing process with PID {pid}")
                        subprocess.run(f"taskkill /F /PID {pid}", shell=True)
        # For Linux/Mac
        else:
            output = subprocess.check_output(f'lsof -i :{port} -t', shell=True).decode()
            if output:
                for pid in output.strip().split('\n'):
                    print(f"Killing process with PID {pid}")
                    subprocess.run(f"kill -9 {pid}", shell=True)
    except Exception as e:
        print(f"Error checking port processes: {e}")

def kill_mados_processes():
    """Kill any MADOS processes"""
    print("Checking for MADOS processes...")
    
    count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] in ['python.exe', 'python', 'python3']:
                cmdline = " ".join(proc.info['cmdline'] if proc.info['cmdline'] else [])
                if 'mados_service.py' in cmdline:
                    print(f"Killing MADOS process: PID {proc.pid}, cmdline: {cmdline[:50]}...")
                    proc.kill()
                    count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            print(f"Error accessing process: {e}")
    
    print(f"Killed {count} MADOS processes")

def start_mados_service():
    """Start the MADOS service"""
    print("Starting MADOS service...")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Only use first GPU
        env['OMP_NUM_THREADS'] = '2'  # Limit OpenMP threads
        env['PYTHONUNBUFFERED'] = '1'  # Unbuffered output
        
        if os.name == 'nt':  # Windows
            process = subprocess.Popen(
                ["start", "python", "mados_service.py"],
                shell=True,
                env=env,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:  # Linux/Mac
            process = subprocess.Popen(
                ["python", "mados_service.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
        
        print("MADOS service started")
        
        # Wait for service to start
        max_retries = 20
        for i in range(max_retries):
            time.sleep(1)
            try:
                response = requests.get("http://localhost:5000/health", timeout=2)
                if response.status_code == 200:
                    print(f"MADOS service is running! Response: {response.json()}")
                    return True
            except:
                print(f"Waiting for MADOS service to start ({i+1}/{max_retries})...")
        
        print(f"MADOS service didn't respond after {max_retries} seconds")
        return False
    except Exception as e:
        print(f"Error starting MADOS service: {e}")
        return False

def main():
    """Main test function"""
    print("=== MADOS Service Test ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # 1. Clean up existing processes and temp files
    kill_mados_processes()
    kill_port_processes(5000)
    clean_temp_dirs()
    clear_gpu_memory()
    
    # 2. Start the MADOS service
    print("\n=== Starting MADOS Service ===")
    start_result = start_mados_service()
    
    # 3. Test the health endpoint
    if start_result:
        print("\n=== Testing MADOS Health Endpoint ===")
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            print(f"Health check response: {response.status_code}")
            print(f"Health data: {response.json()}")
        except Exception as e:
            print(f"Error checking health: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main() 