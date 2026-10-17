@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -B -m schrodinger3d.plotting %*
) else if exist "%~dp0.audit-venv\Scripts\python.exe" (
    "%~dp0.audit-venv\Scripts\python.exe" -B -m schrodinger3d.plotting %*
) else (
    python -B -m schrodinger3d.plotting %*
)
set "plot_exit=%errorlevel%"
if not "%plot_exit%"=="0" (
    echo.
    echo Plot failed. Check the error above and config\settings.toml.
) else (
    echo.
    echo Done. Images are saved in outputs\plots unless configured otherwise.
)
pause
exit /b %plot_exit%
