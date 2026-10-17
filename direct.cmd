@echo off
call "%~dp0plot.cmd" --config "%~dp0config\direct.toml"
exit /b %errorlevel%
