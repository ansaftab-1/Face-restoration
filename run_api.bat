@echo off
title Photo Enhancer & Face Restoration Server (Port 8000)
cd /d "%~dp0"
set "PATH=%~dp0.venv\Scripts;%PATH%"
echo Starting Photo Enhancer Server on http://localhost:8000 ...
echo Opening http://localhost:8000 in your browser...
start http://localhost:8000
uv run python api_server.py
pause

