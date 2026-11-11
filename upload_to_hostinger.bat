@echo off
title Deploy FaceFusion via Docker to Hostinger VPS (187.7.79.207)
cd /d "%~dp0"

echo ==========================================================
echo 🐳 Deploying FaceFusion via DOCKER to Hostinger (187.7.79.207)
echo ==========================================================
echo.
echo Please type your Hostinger root password when prompted.
echo.

echo [1/2] Uploading project files, models, and 18+ moderation code...
ssh root@187.7.79.207 "mkdir -p /root/facefusion"

scp -r Dockerfile docker-compose.yml requirements.txt api_server.py facefusion.py facefusion.ini deploy_docker_hostinger.sh API_ENDPOINTS_DOCUMENTATION.md facefusion static .assets root@187.7.79.207:/root/facefusion/

echo.
echo [2/2] Building and running Docker containers on Hostinger...
ssh root@187.7.79.207 "cd /root/facefusion && chmod +x deploy_docker_hostinger.sh && bash deploy_docker_hostinger.sh"

echo.
echo ==========================================================
echo 🎉 All Done! Your Docker containers are running on Hostinger.
echo ==========================================================
pause
