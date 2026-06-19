@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\ayuan\anaconda3\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Anaconda Python was not found at:
    echo %PYTHON%
    pause
    exit /b 1
)

echo Checking dashboard dependencies...
"%PYTHON%" -c "import streamlit, fastapi, pydantic, mlflow, pandas, matplotlib"
if errorlevel 1 (
    echo.
    echo A dashboard dependency is missing.
    echo Run this command first:
    echo "%PYTHON%" -m pip install -r "%~dp0requirements-dashboard.txt"
    pause
    exit /b 1
)

echo Starting Home Credit Decision Lab...
"%PYTHON%" -m streamlit run "%~dp0monitoring_dashboard\monitoring_dashboard.py"

endlocal
