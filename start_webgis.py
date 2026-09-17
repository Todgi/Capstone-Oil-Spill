import os
import subprocess
import sys
import time
import atexit
import signal
import psutil
import logging
import shutil
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("webgis_startup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("webgis_startup")

# Configuration
APP_SCRIPT = "app.py"
MADOS_SCRIPT = "mados_service.py"
MONITOR_SCRIPT = "restart_mados.py"
APP_PORT = 8000
MADOS_PORT = 5000
USE_GPU = False  # Ubah jadi False

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

def start_service(script_name, new_console=True, wait_port=None, retry_count=3):
    """Start a service and wait for it to be ready"""
    logger.info(f"Starting service: {script_name}")
    
    # Clean up before starting
    if script_name == MADOS_SCRIPT:
        clean_temp_dirs()
        clear_gpu_memory()
        
    # Check if already running
    if wait_port and is_port_in_use(wait_port):
        logger.info(f"Service appears to be already running on port {wait_port}")
        return None
    
    # Set environment variables
    env = os.environ.copy()
    if script_name == MADOS_SCRIPT:
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Only use first GPU
        env['OMP_NUM_THREADS'] = '2'  # Limit OpenMP threads
    env['PYTHONUNBUFFERED'] = '1'  # Unbuffered output
    
    # Create the service process
    try:
        if os.name == 'nt':  # Windows
            if new_console:
                # Start in new console window
                process = subprocess.Popen(
                    ["start", "python", script_name],
                    shell=True,
                    env=env,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # Start in same console 
                process = subprocess.Popen(
                    ["python", script_name],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
        else:  # Linux/Mac
            process = subprocess.Popen(
                ["python", script_name],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
        logger.info(f"Started {script_name} process")
        
        # Wait for service to be ready
        if wait_port:
            retry = 0
            while retry < retry_count:
                for attempt in range(20):  # Wait up to 20 seconds
                    time.sleep(1)
                    if is_port_in_use(wait_port):
                        logger.info(f"Service is running on port {wait_port}")
                        return process
                    if attempt % 5 == 0:
                        logger.info(f"Waiting for service to start on port {wait_port} (attempt {attempt+1}/20)...")
                
                # Service didn't start, try again
                retry += 1
                if retry < retry_count:
                    logger.warning(f"Service failed to start, retrying ({retry}/{retry_count})...")
                    kill_python_process_by_script(script_name)
                    time.sleep(2)
                    
                    # Try to start again
                    if os.name == 'nt' and new_console:
                        process = subprocess.Popen(
                            ["start", "python", script_name],
                            shell=True,
                            env=env,
                            creationflags=subprocess.CREATE_NEW_CONSOLE
                        )
                    else:
                        process = subprocess.Popen(
                            ["python", script_name],
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
            
            logger.error(f"Service failed to start after {retry_count} attempts")
            return None
        return process
    except Exception as e:
        logger.error(f"Error starting service {script_name}: {e}")
        return None

def clean_up(processes):
    """Kill all running processes"""
    logger.info("Cleaning up processes...")
    for name, process in processes.items():
        if process and process.poll() is None:
            try:
                logger.info(f"Terminating {name} process")
                process.terminate()
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, Exception) as e:
                logger.warning(f"Error terminating {name} process: {e}")
                try:
                    process.kill()
                except Exception:
                    pass
    
    # Extra cleanup
    kill_python_process_by_script(APP_SCRIPT)
    kill_python_process_by_script(MADOS_SCRIPT)
    kill_python_process_by_script(MONITOR_SCRIPT)
    
    # Ensure ports are free
    if is_port_in_use(APP_PORT):
        kill_process_on_port(APP_PORT)
    if is_port_in_use(MADOS_PORT):
        kill_process_on_port(MADOS_PORT)

def start_mados_monitor():
    """Start the MADOS monitor service"""
    if not os.path.exists(MONITOR_SCRIPT):
        logger.warning(f"MADOS monitor script {MONITOR_SCRIPT} not found, skipping")
        return None
    
    logger.info("Starting MADOS monitor service")
    try:
        # Start in the background
        if os.name == 'nt':  # Windows
            process = subprocess.Popen(
                ["start", "python", MONITOR_SCRIPT],
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:  # Linux/Mac
            process = subprocess.Popen(
                ["python", MONITOR_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        logger.info("MADOS monitor service started")
        return process
    except Exception as e:
        logger.error(f"Error starting MADOS monitor service: {e}")
        return None

def main():
    logger.info("==== Starting WebGIS Oil Spill Detection Application ====")
    
    # Register cleanup handler
    processes = {}
    atexit.register(clean_up, processes)
    
    # Handle Ctrl+C (SIGINT)
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down...")
        clean_up(processes)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Ensure ports are free first
    if is_port_in_use(APP_PORT):
        logger.warning(f"Port {APP_PORT} is already in use, attempting to free it")
        kill_process_on_port(APP_PORT)
        time.sleep(1)
    if is_port_in_use(MADOS_PORT):
        logger.warning(f"Port {MADOS_PORT} is already in use, attempting to free it")
        kill_process_on_port(MADOS_PORT)
        time.sleep(1)
    
    # Clean up any previous processes and temporary files
    kill_python_process_by_script(APP_SCRIPT)
    kill_python_process_by_script(MADOS_SCRIPT)
    kill_python_process_by_script(MONITOR_SCRIPT)
    clean_temp_dirs()
    clear_gpu_memory()
    
    # Start MADOS service first
    logger.info("Starting MADOS service...")
    processes['mados'] = start_service(MADOS_SCRIPT, new_console=True, wait_port=MADOS_PORT)
    
    # Start the main application
    logger.info("Starting main application...")
    processes['app'] = start_service(APP_SCRIPT, new_console=True, wait_port=APP_PORT)
    
    # Start the MADOS monitor (wait a bit for MADOS to initialize)
    time.sleep(5)
    processes['monitor'] = start_mados_monitor()
    
    logger.info("All services started!")
    logger.info("Access the application at: http://localhost:8000")
    logger.info("Press Ctrl+C to stop all services and exit")
    
    # Keep main thread running to handle signals
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        clean_up(processes)
        sys.exit(0)

if __name__ == "__main__":
    main() 