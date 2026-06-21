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

echo Starting the prediction API at http://localhost:8000 ...
"%UV%" run uvicorn homecredit_service.main:app --host 0.0.0.0 --port 8000
endlocal
