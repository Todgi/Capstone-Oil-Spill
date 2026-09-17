import os
import sys
import time
import subprocess
import requests
import logging
import signal
import psutil
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mados_monitor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("mados_monitor")

# Configuration
MADOS_PORT = 5000
MADOS_HOST = "localhost"
MADOS_URL = f"http://{MADOS_HOST}:{MADOS_PORT}"
CHECK_INTERVAL = 20  # seconds
MAX_MEMORY_PERCENT = 85  # Restart if memory usage exceeds this percentage
MADOS_SCRIPT = "mados_service.py"
MAX_CONSECUTIVE_FAILURES = 3

def clear_temp_dirs():
    """Clear temporary mados job directories that might be causing issues"""
    try:
        temp_dir = os.path.join(os.environ.get('TEMP', os.environ.get('TMP', '/tmp')))
        logger.info(f"Cleaning up temp directory: {temp_dir}")
        
        # Look for mados job directories
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
        logger.error(f"Error cleaning temp directories: {e}")

def reset_gpu_memory():
    """Reset GPU memory in case of leaks"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Clearing CUDA memory")
            torch.cuda.empty_cache()
            
            # More aggressive approach
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    # Reset memory stats
                    torch.cuda.reset_peak_memory_stats()
            
            logger.info("CUDA memory cleared")
    except Exception as e:
        logger.warning(f"Failed to reset GPU memory: {e}")

def is_service_running():
    """Check if the MADOS service is running and responsive"""
    try:
        response = requests.get(f"{MADOS_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def find_mados_process():
    """Find the MADOS process ID if it's running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if Python is running the MADOS script
            if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                cmdline = " ".join(proc.info['cmdline']) if proc.info['cmdline'] else ""
                if MADOS_SCRIPT in cmdline:
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

def kill_mados_process():
    """Try to kill any existing MADOS processes"""
    proc = find_mados_process()
    if proc:
        logger.info(f"Terminating existing MADOS process (PID: {proc.pid})")
        try:
            proc.terminate()
            # Wait for process to terminate gracefully
            gone, alive = psutil.wait_procs([proc], timeout=10)
            if alive:
                # Force kill if still alive
                for p in alive:
                    p.kill()
            return True
        except:
            logger.exception("Failed to terminate MADOS process")
            
            # More aggressive approach - kill all Python processes with MADOS
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if (proc.info['name'] == 'python.exe' or proc.info['name'] == 'python') and \
                           MADOS_SCRIPT in " ".join(proc.info['cmdline']):
                            proc.kill()
                    except:
                        pass
            except:
                pass
    return False

def start_mados_service():
    """Start the MADOS service"""
    logger.info("Starting MADOS service...")
    
    # First clear temp dirs and reset GPU memory
    clear_temp_dirs()
    reset_gpu_memory()
    
    try:
        # Determine the full path to the MADOS script
        script_path = os.path.join(os.getcwd(), MADOS_SCRIPT)
        if not os.path.exists(script_path):
            script_path = MADOS_SCRIPT  # Try without path
            
        # Set environment variables for performance
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Only use first GPU
        env['OMP_NUM_THREADS'] = '2'  # Limit OpenMP threads
        env['PYTHONUNBUFFERED'] = '1'  # Unbuffered output
        
        # Start the process in a new command window (Windows)
        if os.name == 'nt':
            process = subprocess.Popen(
                ["start", "python", script_path],
                shell=True,
                env=env,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            # For Linux/Mac
            process = subprocess.Popen(
                ["python", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            
        logger.info(f"Started MADOS service with PID: {process.pid}")
        return True
    except Exception as e:
        logger.exception(f"Failed to start MADOS service: {str(e)}")
        return False

def restart_service():
    """Restart the MADOS service"""
    logger.info("Restarting MADOS service...")
    
    # Kill any existing process
    kill_mados_process()
    
    # Wait for the port to be released
    timeout = 30
    for i in range(timeout):
        if not is_port_in_use(MADOS_PORT):
            break
        time.sleep(1)
    
    if is_port_in_use(MADOS_PORT):
        logger.warning(f"Port {MADOS_PORT} is still in use after {timeout}s")
        # Try to forcibly kill any process using port 5000
        kill_process_on_port(MADOS_PORT)
        time.sleep(2)
    
    # Start the service
    return start_mados_service()

def is_port_in_use(port):
    """Check if a port is in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """Forcibly kill any process using the specified port"""
    try:
        if os.name == 'nt':  # Windows
            # Using netstat to find process using the port
            output = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()
            if output:
                # Extract PID (last column)
                for line in output.split('\n'):
                    if f':{port}' in line and 'LISTENING' in line:
                        pid = line.strip().split()[-1]
                        try:
                            # Kill the process
                            os.kill(int(pid), signal.SIGTERM)
                            logger.info(f"Killed process {pid} using port {port}")
                            return True
                        except:
                            logger.error(f"Failed to kill process {pid}")
        else:  # Linux/Mac
            output = subprocess.check_output(f'lsof -i :{port} -t', shell=True).decode()
            if output:
                pid = output.strip()
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    logger.info(f"Killed process {pid} using port {port}")
                    return True
                except:
                    logger.error(f"Failed to kill process {pid}")
        return False
    except:
        return False

def check_memory_usage():
    """Check if MADOS process is using too much memory"""
    proc = find_mados_process()
    if proc:
        try:
            # Check both system memory and GPU memory if applicable
            memory_percent = proc.memory_percent()
            logger.info(f"MADOS memory usage: {memory_percent:.1f}%")
            
            # Check for GPU memory if CUDA is available
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024 * 1024 * 1024)  # GB
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024 * 1024 * 1024)  # GB
                        logger.info(f"GPU {i} memory: allocated={memory_allocated:.2f}GB, reserved={memory_reserved:.2f}GB")
                        
                        # Check if memory is over threshold
                        if memory_allocated > 4.0:  # 4GB threshold
                            logger.warning(f"GPU {i} memory usage too high: {memory_allocated:.2f}GB")
                            return True
            except:
                pass
                
            if memory_percent > MAX_MEMORY_PERCENT:
                logger.warning(f"MADOS using too much memory ({memory_percent:.1f}%), restarting...")
                return True
        except:
            pass
    return False

def main():
    logger.info("Starting MADOS service monitor")
    consecutive_failures = 0
    
    # Initial startup
    if not is_service_running():
        restart_service()
    
    last_restart = time.time()
    
    try:
        while True:
            if not is_service_running() or check_memory_usage():
                consecutive_failures += 1
                logger.warning(f"MADOS service failed or using too much memory (failure #{consecutive_failures})")
                
                # Don't restart too frequently
                if time.time() - last_restart > 60:  # At least 1 minute between restarts
                    restart_success = restart_service()
                    last_restart = time.time()
                    
                    if not restart_success:
                        logger.error("Failed to restart MADOS service")
                    
                    # If we've had too many consecutive failures, take more drastic measures
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.warning(f"Too many consecutive failures ({consecutive_failures}), clearing all temp files")
                        clear_temp_dirs()  # Clear all temp files
                        reset_gpu_memory()  # Reset GPU memory
                        consecutive_failures = 0  # Reset counter
            else:
                consecutive_failures = 0  # Reset counter on successful checks
            
            # Check every CHECK_INTERVAL seconds
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
    except Exception as e:
        logger.exception(f"Monitor error: {str(e)}")

if __name__ == "__main__":
    main()