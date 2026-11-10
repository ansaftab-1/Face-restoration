@echo off
title FaceFusion 3.9.2 - Scalable FastAPI REST Server (Port 8000)
cd /d "%~dp0"
echo Starting FaceFusion Scalable FastAPI Server on http://localhost:8000 ...
echo Interactive Swagger API Docs: http://localhost:8000/docs
".venv\Scripts\python.exe" api_server.py
pause

