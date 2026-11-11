#!/bin/bash
set -e

# ==============================================================================
# FaceFusion 3.9.2 + Scalable FastAPI Server Automated Hostinger Deployer
# Server IP: 187.7.79.207
# ==============================================================================

echo "=========================================================="
echo "🚀 Starting FaceFusion 3.9.2 Installation on Hostinger VPS"
echo "=========================================================="

# 1. Update OS & Install System Dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    curl wget git ca-certificates \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    python3 python3-pip python3-venv python3-dev \
    build-essential ufw

# 2. Setup Application Directory
INSTALL_DIR="/root/facefusion"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# 3. Download FaceFusion 3.9.2 Core Code if not present
if [ ! -f "facefusion.py" ]; then
    echo "📦 Downloading FaceFusion 3.9.2 release..."
    wget -q https://github.com/facefusion/facefusion/archive/refs/tags/3.9.2.tar.gz -O /tmp/facefusion-3.9.2.tar.gz
    tar -xzf /tmp/facefusion-3.9.2.tar.gz --strip-components=1 -C "$INSTALL_DIR"
    rm -f /tmp/facefusion-3.9.2.tar.gz
fi

# 4. Create Python Virtual Environment
if [ ! -d ".venv" ]; then
    echo "🐍 Setting up Python Virtual Environment (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# 5. Install Dependencies (Detecting GPU vs CPU)
echo "📦 Installing Python dependencies..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected! Installing onnxruntime-gpu..."
    pip install onnxruntime-gpu==1.20.1
else
    echo "⚙️ CPU Server detected! Installing onnxruntime..."
    pip install onnxruntime==1.20.1
fi

pip install \
    gradio==5.50.0 \
    fastapi==0.115.6 \
    uvicorn==0.34.0 \
    opencv-python-headless==4.10.0.84 \
    pillow \
    numpy \
    scipy \
    tqdm \
    psutil \
    pydantic \
    python-multipart \
    httpx

# 6. Create Models Directory & Download Core Assets
MODELS_DIR="$INSTALL_DIR/.assets/models"
mkdir -p "$MODELS_DIR"

echo "📥 Downloading core ONNX models from facefusion-assets..."
MODELS=(
    "inswapper_128_fp16.onnx|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.onnx"
    "inswapper_128_fp16.hash|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.hash"
    "gfpgan_1.4.onnx|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx"
    "gfpgan_1.4.hash|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.hash"
    "yoloface_8n.onnx|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx"
    "yoloface_8n.hash|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.hash"
    "2dfan4.onnx|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/2dfan4.onnx"
    "2dfan4.hash|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/2dfan4.hash"
    "arcface_w600k_r50.onnx|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.onnx"
    "arcface_w600k_r50.hash|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.hash"
    "xseg_1.onnx|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/xseg_1.onnx"
    "xseg_1.hash|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/xseg_1.hash"
    "bisenet_resnet_34.onnx|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx"
    "bisenet_resnet_34.hash|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.hash"
    "nsfw_1.onnx|https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_1.onnx"
    "nsfw_1.hash|https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_1.hash"
    "nsfw_2.onnx|https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_2.onnx"
    "nsfw_2.hash|https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_2.hash"
    "nsfw_3.onnx|https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_3.onnx"
    "nsfw_3.hash|https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_3.hash"
    "ddcolor.onnx|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ddcolor.onnx"
    "ddcolor.hash|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ddcolor.hash"
    "span_kendata_x4.onnx|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/span_kendata_x4.onnx"
    "span_kendata_x4.hash|https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/span_kendata_x4.hash"
)

for item in "${MODELS[@]}"; do
    FILE="${item%%|*}"
    URL="${item##*|}"
    if [ ! -f "$MODELS_DIR/$FILE" ]; then
        echo "Downloading $FILE..."
        wget -q -c "$URL" -O "$MODELS_DIR/$FILE" || true
    fi
done

# 7. Configure 18+ Content Moderation
cat << 'EOF' > "$INSTALL_DIR/facefusion/content_moderation.py"
import os
from typing import Any, Dict, List, Optional
import numpy
from facefusion import content_analyser, logger
from facefusion.filesystem import filter_image_paths, is_image, is_video
from facefusion.vision import detect_video_fps, is_vision_frame, read_image, restrict_trim_frame

REJECTION_MESSAGE = (
    "[18+ Content Restricted] Adult / NSFW (18+) content is strictly prohibited. "
    "Your photo/video has been rejected and cannot be uploaded or processed. "
    "(18+ content upload ya run karna mana hai - photo reject kar di gayi hai.)"
)

def inspect_frame_scores(vision_frame: Any) -> Dict[str, Any]:
    content_analyser.pre_check()
    frame_1 = content_analyser.prepare_detect_frame(vision_frame, "nsfw_1")
    det_1 = content_analyser.forward_nsfw(frame_1, "nsfw_1")
    score_1 = float(numpy.max(numpy.amax(det_1[:, 4:], axis=1)))
    flag_1 = bool(score_1 > 0.2)

    frame_2 = content_analyser.prepare_detect_frame(vision_frame, "nsfw_2")
    det_2 = content_analyser.forward_nsfw(frame_2, "nsfw_2")
    score_2 = float(det_2[0] - det_2[1])
    flag_2 = bool(score_2 > 0.25)

    frame_3 = content_analyser.prepare_detect_frame(vision_frame, "nsfw_3")
    det_3 = content_analyser.forward_nsfw(frame_3, "nsfw_3")
    score_3 = float((det_3[2] + det_3[3]) - (det_3[0] + det_3[1]))
    flag_3 = bool(score_3 > 10.5)

    is_18_plus = bool((flag_1 and flag_2) or (flag_1 and flag_3) or (flag_2 and flag_3))
    return {
        "is_18_plus": is_18_plus,
        "allowed": not is_18_plus,
        "models": {
            "nsfw_1_erax": {"flagged": flag_1, "score": round(score_1, 4), "threshold": 0.2},
            "nsfw_2_marqo": {"flagged": flag_2, "score": round(score_2, 4), "threshold": 0.25},
            "nsfw_3_freepik": {"flagged": flag_3, "score": round(score_3, 4), "threshold": 10.5},
        },
        "message": REJECTION_MESSAGE if is_18_plus else "Safe content (SFW) - allowed.",
    }

def check_media_18_plus(file_path: str, trim_frame_start: Optional[int] = None, trim_frame_end: Optional[int] = None, delete_on_reject: bool = False) -> Dict[str, Any]:
    if not file_path or not os.path.isfile(file_path):
        return {"is_18_plus": False, "allowed": True, "media_type": "unknown", "message": "File not found or non-visual file."}
    content_analyser.pre_check()
    if is_image(file_path):
        vision_frame = read_image(file_path)
        if not is_vision_frame(vision_frame):
            return {"is_18_plus": False, "allowed": True, "media_type": "image", "message": "Unable to decode image frame."}
        details = inspect_frame_scores(vision_frame)
        details["media_type"] = "image"
        if details["is_18_plus"]:
            logger.error(REJECTION_MESSAGE, __name__)
            if delete_on_reject:
                try: os.remove(file_path)
                except OSError: pass
        return details
    if is_video(file_path):
        start_f, end_f = restrict_trim_frame(file_path, trim_frame_start, trim_frame_end)
        flagged = content_analyser.analyse_video(file_path, start_f, end_f)
        if flagged:
            logger.error(REJECTION_MESSAGE, __name__)
            if delete_on_reject:
                try: os.remove(file_path)
                except OSError: pass
        return {"is_18_plus": bool(flagged), "allowed": not bool(flagged), "media_type": "video", "message": REJECTION_MESSAGE if flagged else "Safe video content (SFW) - allowed."}
    return {"is_18_plus": False, "allowed": True, "media_type": "other", "message": "Safe content - allowed."}

def check_source_and_target(source_paths: Optional[List[str]], target_path: Optional[str], trim_frame_start: Optional[int] = None, trim_frame_end: Optional[int] = None, delete_on_reject: bool = False) -> Dict[str, Any]:
    for src_path in filter_image_paths(source_paths or []):
        res = check_media_18_plus(src_path, delete_on_reject=delete_on_reject)
        if res["is_18_plus"]:
            if delete_on_reject and target_path and os.path.isfile(target_path):
                try: os.remove(target_path)
                except OSError: pass
            res["rejected_input"] = "source"
            res["rejected_file"] = os.path.basename(src_path)
            return res
    if target_path:
        res = check_media_18_plus(target_path, trim_frame_start=trim_frame_start, trim_frame_end=trim_frame_end, delete_on_reject=delete_on_reject)
        if res["is_18_plus"]:
            if delete_on_reject:
                for src_path in source_paths or []:
                    if src_path and os.path.isfile(src_path):
                        try: os.remove(src_path)
                        except OSError: pass
            res["rejected_input"] = "target"
            res["rejected_file"] = os.path.basename(target_path)
            return res
    return {"is_18_plus": False, "allowed": True, "rejected_input": None, "message": "Both source and target passed 18+ content moderation."}
EOF

# 8. Setup Systemd Service for 24/7 Background Running
echo "⚙️ Creating systemd service for FastAPI server (Port 8000)..."
cat << EOF > /etc/systemd/system/facefusion-api.service
[Unit]
Description=FaceFusion Scalable FastAPI REST API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/.venv/bin/python api_server.py
Restart=always
RestartSec=5
Environment=HOST=0.0.0.0
Environment=PORT=8000
Environment=FACEFUSION_MEMORY_STRATEGY=tolerant

[Install]
WantedBy=multi-user.target
EOF

# 9. Configure Firewall (Open Ports 8000 and 7860)
echo "🛡️ Configuring Firewall..."
ufw allow 8000/tcp || true
ufw allow 7860/tcp || true

# 10. Enable & Start Service
systemctl daemon-reload
systemctl enable facefusion-api
systemctl restart facefusion-api

echo "=========================================================="
echo "🎉 DEPLOYMENT COMPLETE! FaceFusion is Live on Hostinger!"
echo "=========================================================="
echo "Swagger API Docs: http://187.7.79.207:8000/docs"
echo "ReDoc Reference:  http://187.7.79.207:8000/redoc"
echo "Health Check:     http://187.7.79.207:8000/api/v1/health"
echo "Service Status:   systemctl status facefusion-api"
echo "View Live Logs:   journalctl -u facefusion-api -f"
echo "=========================================================="

