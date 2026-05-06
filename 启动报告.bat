@echo off
echo Starting server...
cd /d "%~dp0"
start "" "http://localhost:8765/report.html"
python -m http.server 8765
