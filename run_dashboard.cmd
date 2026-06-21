@echo off
setlocal
cd /d "%~dp0"

set "UV=uv"
where uv >nul 2>&1
if errorlevel 1 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV=%USERPROFILE%\.local\bin\uv.exe"
    ) else (
        echo ERROR: uv is required but was not found.
        echo Install it from https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
)

echo Starting Home Credit Decision Lab with the locked project environment...
"%UV%" run --group dashboard streamlit run "%~dp0INSY684_Group-Extend\INSY684_Group-Extend\monitoring_dashboard\monitoring_dashboard.py"
endlocal
