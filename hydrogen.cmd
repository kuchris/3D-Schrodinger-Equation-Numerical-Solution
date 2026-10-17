@echo off
call "%~dp0plot.cmd" --config "%~dp0config\hydrogen.toml"
exit /b %errorlevel%
