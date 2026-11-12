@echo off
cd /d "%~dp0"
py -3.12 -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo Install Python 3.12 from https://www.python.org/downloads/ and run this file again.
  exit /b 1
)
if not exist .venv\Scripts\python.exe py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python scripts\ensure_runtime.py
if errorlevel 1 exit /b 1
.venv\Scripts\streamlit run app.py
