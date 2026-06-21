@echo off
setlocal
call "%~dp0..\..\run_dashboard.cmd"
exit /b %ERRORLEVEL%
