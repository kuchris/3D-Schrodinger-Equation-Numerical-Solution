@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    python -m venv "%~dp0.venv"
    if errorlevel 1 goto failed
)
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto failed
echo.
echo Setup complete. Edit settings.toml and custom_potential.py, then run plot.cmd.
pause
exit /b 0

:failed
echo.
echo Setup failed. Check the error above. Python 3.10 or later is required on PATH.
pause
exit /b 1
