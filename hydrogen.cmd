@echo off
call "%~dp0plot.cmd" --config "%~dp0hydrogen_settings.toml"
exit /b %errorlevel%
