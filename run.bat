@echo off
REM Run this from inside the depth_calib_tool folder (double-click it, or run
REM from a command prompt). Creates the venv only if it doesn't exist yet,
REM installs/updates dependencies, then launches the app.

cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing/updating dependencies...
pip install -r requirements.txt

echo Starting app...
python app.py

pause
