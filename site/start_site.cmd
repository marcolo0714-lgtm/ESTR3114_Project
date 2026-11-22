@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the environment first using the instructions in README.md.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop.
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
