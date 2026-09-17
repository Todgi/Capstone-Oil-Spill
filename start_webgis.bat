@echo off
echo.
echo ===== WebGIS Oil Spill Detection Application =====
echo.

REM Check for required Python packages
echo Checking for required packages...
pip install psutil requests -q
if %ERRORLEVEL% NEQ 0 (
    echo Error installing required packages! Please check your Python installation.
    goto :error
)

REM First, clear temp directories that might be causing issues
python -c "import tempfile, os, shutil; temp_dir = tempfile.gettempdir(); [shutil.rmtree(os.path.join(temp_dir, d)) for d in os.listdir(temp_dir) if d.startswith('mados_job_') and os.path.isdir(os.path.join(temp_dir, d))]"

REM Kill any existing processes
echo Stopping any existing services...
taskkill /F /FI "WINDOWTITLE eq mados_service.py" /T >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq app.py" /T >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq restart_mados.py" /T >nul 2>&1

REM Clear ports 
python -c "import psutil, os; [os.system(f'taskkill /F /PID {conn.pid}') for proc in psutil.process_iter(['pid', 'connections']) for conn in proc.connections() if hasattr(conn, 'laddr') and conn.laddr.port in [5000, 8000]]" >nul 2>&1

REM Start the application
echo Starting WebGIS Oil Spill Detection Application...
python start_webgis.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error starting the application! 
    echo Please check the logs or try running manually:
    echo   python start_webgis.py
    goto :error
)

goto :end

:error
echo.
echo Application startup failed. Press any key to exit...
pause >nul
exit /b 1

:end
echo.
echo Application is running. Close this window to stop all services.
echo Or press Ctrl+C to stop the application. 