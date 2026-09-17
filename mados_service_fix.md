# MADOS Service Restart Fix

## Problem

The WebGIS Oil Spill Detection application was experiencing crashes and connection issues with the MADOS service, resulting in errors like:

```
Error processing with MADOS: MADOS service error: {"detail":"Error in MADOS proxy: 503: MADOS service is down and restart attempt failed"}
```

This was occurring due to several issues:
1. The MADOS service crashing due to memory leaks
2. Restart mechanism failing to properly clean up resources
3. Temporary files being left behind causing conflicts
4. GPU memory not being properly released

## Solution Implemented

### 1. Robust Service Monitor (`restart_mados.py`)

Created a dedicated monitoring service that:
- Monitors MADOS service health
- Detects when memory usage gets too high
- Cleans temporary directories
- Properly releases GPU memory
- Forcibly kills stuck processes
- Properly handles port conflicts
- Implements retry logic with backoff

### 2. Improved App Restart Logic (`app.py`)

Updated the restart mechanism in the main application to:
- Clean temporary directories before restart
- Release GPU memory properly
- Kill any zombie processes
- Handle port conflicts
- Use proper environment variables for better performance
- Implement progressive polling with timeout

### 3. Enhanced Frontend Error Handling (`script.js`)

Added better error handling in the frontend:
- Retry logic for failed connections
- Proper handling of service restarts
- Informative progress messages
- Detection of connection issues
- State recovery after service restart

### 4. Improved Application Startup (`start_webgis.py`)

Enhanced the main startup script:
- Proper cleanup of resources before starting
- Monitoring of all services
- Graceful shutdown procedures
- Better logging
- Proper error handling and reporting

### 5. Testing Script (`test_mados.py`)

Added a testing utility to:
- Verify MADOS service can start properly
- Clean up temporary directories
- Test GPU memory management
- Check for port conflicts
- Safely kill problematic processes

## How to Use

To fix MADOS service issues:

1. **If the service crashes during processing:**
   - The application will automatically attempt to restart it
   - Progress will be maintained where possible
   - The frontend will display proper error messages

2. **If persistent issues occur:**
   - Run `python restart_mados.py` to start the monitoring service
   - It will automatically restart the MADOS service when needed

3. **For a clean restart of everything:**
   - Close all application windows
   - Run `start_webgis.bat` to start with all the optimizations

## Technical Details

1. **Memory Management**
   - Temporary files are now stored with proper cleanup
   - GPU memory is properly released between processing tasks
   - Memory mapping is used for large datasets
   - Explicit garbage collection is performed

2. **Process Management**
   - Proper process termination with timeouts
   - Port availability checking
   - Killing of zombie processes
   - Recovery from partial failures

3. **Error Handling**
   - Comprehensive error detection
   - Proper status reporting
   - Recovery mechanisms for various failure modes
   - Graceful degradation under heavy load 