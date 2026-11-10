#!/bin/bash
set -e

echo "=========================================================="
echo "🐳 Deploying FaceFusion via Docker on Hostinger (187.7.79.207)"
echo "=========================================================="

# 1. Install Docker & Docker Compose if not already installed
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    apt-get update -y
    apt-get install -y curl ca-certificates gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# 2. Go to directory
cd /root/facefusion

# 3. Open Ports 8000 & 7860 on Firewall
echo "🛡️ Opening Firewall Ports 8000 and 7860..."
ufw allow 8000/tcp || true
ufw allow 7860/tcp || true

# 4. Build and Start Containers
echo "🚀 Building and starting Docker containers in the background..."
docker compose down || true
docker compose up -d --build

# 5. Show Running Containers
echo "=========================================================="
echo "✅ DOCKER CONTAINERS ARE RUNNING!"
echo "=========================================================="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🎉 ACCESS YOUR SERVICES HERE:"
echo "FastAPI Docs: http://187.7.79.207:8000/docs"
echo "Web UI:       http://187.7.79.207:7860"
echo "Health Check: http://187.7.79.207:8000/api/v1/health"
echo "Logs:         docker compose logs -f"
echo "=========================================================="

