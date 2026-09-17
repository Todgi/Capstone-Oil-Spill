@echo off

set "VIRTUAL_ENV=%~dp0.."
set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"
set "PROMPT=(drift-env) %PROMPT%"

:: Set CMEMS credentials if they exist in a config file
if exist "%VIRTUAL_ENV%\cmems_config.txt" (
    for /f "tokens=1,2 delims==" %%a in (%VIRTUAL_ENV%\cmems_config.txt) do (
        set "%%a=%%b"
    )
)

echo Drift Environment activated with CMEMS support
echo To set your CMEMS credentials, create a file named 'cmems_config.txt' in the drift-env folder
echo with the following content:
echo CMEMS_USERNAME=your_username
echo CMEMS_PASSWORD=your_password 