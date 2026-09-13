@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Setting up Realtime Person Counter for the first time, this may take a minute...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Failed to create the virtual environment. Make sure Python is installed
        echo and available on PATH, then try again.
        pause
        exit /b 1
    )

    ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Failed to install dependencies. Check your internet connection and try again.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" "src\main.py"
if errorlevel 1 (
    echo.
    echo The app exited with an error ^(see above^).
    pause
)
