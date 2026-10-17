@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0plot_states.py" %*
) else if exist "%~dp0.audit-venv\Scripts\python.exe" (
    "%~dp0.audit-venv\Scripts\python.exe" "%~dp0plot_states.py" %*
) else (
    python "%~dp0plot_states.py" %*
)
set "plot_exit=%errorlevel%"
if not "%plot_exit%"=="0" (
    echo.
    echo Plot failed. Check the error above and settings.toml.
) else (
    echo.
    echo Done. Images are saved in the plots folder unless configured otherwise.
)
pause
exit /b %plot_exit%
