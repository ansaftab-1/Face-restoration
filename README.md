# 🎨 Face Restoration & Photo Enhancer Platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ansaftab-1/Face-restoration/blob/main/Photo_Restoration_Colab.ipynb)
![License](https://img.shields.io/badge/license-OpenRAIL--AS-green)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-teal)

A state-of-the-art **AI Photo Restoration & Enhancement** platform built on **FaceFusion**, **GFPGAN**, and **DDColor**. Designed for restoring vintage, scratched, blurry, or low-resolution portrait photographs with an ultra-clean, minimal 1-click web interface and a scalable REST API.

---

## 🌟 Key Features

* **🧑‍🦰 AI Face Restoration:** Uses **GFPGAN 1.4** & **CodeFormer** to repair facial tears, scratches, blurriness, and restore clear eyes, teeth, and natural skin texture.
* **🎨 Old Photo Colorization:** Automatically predicts and applies natural colors to vintage black-and-white photos using **DDColor** deep learning prior.
* **✨ Color Vibrance Pop:** CLAHE dynamic contrast and selective vibrance boost to give faded vintage pictures deep, rich color tones.
* **🌸 Skin & Beauty Retouching:** Subtle bilateral filter softly smooths skin grain, blemishes, and cracks while preserving sharp eyes and facial contours.
* **🔍 4x Ultra HD Super-Resolution:** 4x spatial upscaling using **SPAN** deep learning super-resolution.
* **⚡ 1-Click Minimal Web Interface:** Clean, focused UI without confusing controls — drag-and-drop before/after slider comparison.
* **🚀 Fast Warm Inference REST API:** FastAPI backend keeps ONNX models warm in GPU/CPU memory for sub-second / 1-2s response times.
* **☁️ Google Colab 1-Click Execution:** Run directly in the cloud on free NVIDIA T4 GPUs with zero local setup required.

---

## ⚡ Quick Start

### 1. Run in Google Colab (Free Cloud GPU)
Click the badge below to open the interactive notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ansaftab-1/Face-restoration/blob/main/Photo_Restoration_Colab.ipynb)

---

### 2. Run Locally on Windows

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ansaftab-1/Face-restoration.git
   cd Face-restoration
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Start the Photo Enhancer Web UI:**
   Double-click `run_api.bat` or run:
   ```bash
   uv run python api_server.py
   ```
   Open **`http://localhost:8000`** in your browser.

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Minimal Photo Enhancer Web Interface |
| `POST` | `/api/v1/enhance/photo` | Instant Photo Restoration & Colorization |
| `POST` | `/api/v1/swap/image` | Instant Face Swapping |
| `GET` | `/api/v1/health` | Server Health & Warm GPU Models Status |
| `GET` | `/docs` | Interactive Swagger API Documentation |

### Example cURL Request:
```bash
curl -X POST "http://localhost:8000/api/v1/enhance/photo" \
     -H "X-API-Key: your_optional_key" \
     -F "file=@old_photo.jpg" \
     -F "colorize=true"
```

---

## 🏗️ Architecture & Model Pipeline

```
Input Photo (Damaged / B&W)
          │
          ▼
1. Face Detection (YOLOv8-Face)  ──► Locates facial bounding boxes
          │
          ▼
2. Facial Alignment (2DFAN-4)    ──► Normalizes geometry & keypoints
          │
          ▼
3. Face Restoration (GFPGAN 1.4) ──► Reconstructs realistic skin, eyes, teeth
          │
          ▼
4. Colorization (DDColor)        ──► Predicts & applies realistic colors
          │
          ▼
5. Blending & Export             ──► Smooth alpha-mask paste-back to original
```

---

## 📄 License
This project is licensed under the [OpenRAIL-AS License](LICENSE.md). Underlying models (GFPGAN, DDColor) are subject to their respective open-source licenses.
