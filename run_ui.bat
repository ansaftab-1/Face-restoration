@echo off
title FaceFusion 3.9.2 - Local GPU Web UI (DirectML)
cd /d "%~dp0"
set "PATH=%~dp0.venv\Scripts;%PATH%"
echo Starting FaceFusion Web UI on your NVIDIA GPU (DirectML)...
uv run python facefusion.py run --execution-providers directml cpu --video-memory-strategy strict --open-browser
pause

