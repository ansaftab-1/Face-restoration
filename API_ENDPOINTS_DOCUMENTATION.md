# FaceFusion FastAPI — Complete URL & Endpoint Documentation

This document lists **every URL and REST API endpoint** provided by [`api_server.py`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/api_server.py), what each endpoint does, what parameters to send, and exact request/response examples.

---

## 1. Base URLs (Where to Hit the API)

| Environment | Base URL | Interactive Swagger UI |
| :--- | :--- | :--- |
| **Local PC** (via [`run_api.bat`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/run_api.bat)) | `http://localhost:8000` (or `http://127.0.0.1:8000`) | `http://localhost:8000/docs` |
| **Hostinger VPS (Direct IP)** | `http://<YOUR_HOSTINGER_VPS_IP>:8000` | `http://<YOUR_HOSTINGER_VPS_IP>:8000/docs` |
| **Hostinger Domain (HTTPS)** | `https://api.yourdomain.com` | `https://api.yourdomain.com/docs` |

### Optional Authentication Header
If you set the environment variable `FACEFUSION_API_KEY=your_secret_key` in [`docker-compose.yml`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/docker-compose.yml) or your shell, include this HTTP header on `POST` and `DELETE` requests:
```http
X-API-Key: your_secret_key
```
*(By default, `FACEFUSION_API_KEY` is empty so you can test locally without headers.)*

---

## 2. Quick Summary Table of All Endpoints

| # | HTTP Method | Endpoint Path | Full Local URL to Hit | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `GET` | `/docs` | `http://localhost:8000/docs` | Interactive **Swagger UI** to test endpoints in browser |
| **2** | `GET` | `/redoc` | `http://localhost:8000/redoc` | **ReDoc** API reference documentation |
| **3** | `GET` | `/openapi.json` | `http://localhost:8000/openapi.json` | Raw OpenAPI 3.1 JSON schema (for Postman / codegen) |
| **4** | `GET` | `/api/v1/health` | `http://localhost:8000/api/v1/health` | Check server health, GPU providers, queue size & warm models |
| **5** | `GET` | `/api/v1/models` | `http://localhost:8000/api/v1/models` | List downloaded `.onnx` models & all supported model names |
| **6** | `POST` | `/api/v1/moderation/check` | `http://localhost:8000/api/v1/moderation/check` | **18+ / NSFW Upload Pre-Check** (returns `200 OK` if safe or `403 Forbidden` if 18+) |
| **7** | `POST` | `/api/v1/swap/image` | `http://localhost:8000/api/v1/swap/image` | **Instant Image Face Swap** (blocks 18+ uploads & returns swapped image in ~1.21s warm) |
| **8** | `POST` | `/api/v1/jobs` | `http://localhost:8000/api/v1/jobs` | **Async Job Queue** for Images & Videos (blocks 18+ uploads at submission) |
| **9** | `GET` | `/api/v1/jobs` | `http://localhost:8000/api/v1/jobs` | List recent queued, processing, and completed jobs |
| **10** | `GET` | `/api/v1/jobs/{job_id}` | `http://localhost:8000/api/v1/jobs/{job_id}` | Check status, progress, and download link of a specific job |
| **11** | `GET` | `/api/v1/jobs/{job_id}/result` | `http://localhost:8000/api/v1/jobs/{job_id}/result` | Download the finished swapped image or video file |
| **12** | `DELETE` | `/api/v1/jobs/{job_id}` | `http://localhost:8000/api/v1/jobs/{job_id}` | Cancel/delete a job and remove its files from disk |
| **13** | `GET` | `/outputs/{filename}` | `http://localhost:8000/outputs/{filename}` | Direct static URL to view/download generated output files |

---

## 3. Detailed Endpoint Reference

### Endpoint 1: Health & GPU Status
- **Method & Path**: `GET /api/v1/health`
- **URL to Hit**: [`health_check`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/api_server.py#L376-L398) -> `http://localhost:8000/api/v1/health`
- **What It Does**: Verifies the API server is online, shows active GPU/CPU execution providers (`DmlExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`), how many jobs are currently queued or processing, and which ONNX models are warm in memory.

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**Example Response (`200 OK`):**
```json
{
  "status": "ok",
  "version": "3.9.2",
  "onnxruntime_version": "1.24.4",
  "onnxruntime_providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
  "active_execution_providers": ["directml", "cpu"],
  "video_memory_strategy": "tolerant",
  "max_concurrent_jobs": 1,
  "queued_jobs": 0,
  "processing_jobs": 0,
  "warm_loaded_models": [
    "facefusion.face_detector.yolo_face.0.directml.cpu",
    "facefusion.processors.modules.face_swapper.core.inswapper_128_fp16.0.directml.cpu"
  ],
  "downloaded_models_count": 13
}
```

---

### Endpoint 2: List Available & Downloaded Models
- **Method & Path**: `GET /api/v1/models`
- **URL to Hit**: [`list_models`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/api_server.py#L401-L422) -> `http://localhost:8000/api/v1/models`
- **What It Does**: Lists all `.onnx` model files already downloaded in `.assets/models/` along with all supported `face_swapper_model` and `face_enhancer_model` choices.

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/models"
```

**Example Response (`200 OK`):**
```json
{
  "downloaded_models": [
    { "name": "inswapper_128_fp16.onnx", "size_mb": 264.8 },
    { "name": "gfpgan_1.4.onnx", "size_mb": 324.5 }
  ],
  "available_face_swapper_models": [
    "alphaface_256",
    "blendswap_256",
    "ghost_1_256",
    "hyperswap_1a_256",
    "inswapper_128",
    "inswapper_128_fp16",
    "simswap_256",
    "uniface_256"
  ],
  "available_face_enhancer_models": [
    "codeformer",
    "gfpgan_1.2",
    "gfpgan_1.3",
    "gfpgan_1.4",
    "gpen_bfr_256",
    "gpen_bfr_512",
    "restoreformer_plus_plus"
  ],
  "recommended_for_4gb_gpu": {
    "face_swapper_model": "inswapper_128_fp16",
    "face_enhancer_model": "gfpgan_1.4"
  }
}
```



---

### Endpoint 2.5: 18+ / NSFW Content Moderation Pre-Check
- **Method & Path**: `POST /api/v1/moderation/check`
- **URL to Hit**: [`check_content_moderation`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/api_server.py#L446-L491) -> `http://localhost:8000/api/v1/moderation/check`
- **Content-Type**: `multipart/form-data`
- **What It Does**: Uses [`facefusion/content_moderation.py`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/facefusion/content_moderation.py) and the 3-model ONNX ensemble (`nsfw_1.onnx` EraX, `nsfw_2.onnx` Marqo, `nsfw_3.onnx` Freepik) to scan any uploaded photo or video. Also runs automatically inside `POST /api/v1/swap/image`, `POST /api/v1/jobs`, and the Gradio Web UI upload boxes ([`source.py`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/facefusion/uis/components/source.py) & [`target.py`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/facefusion/uis/components/target.py)).

**Example Request (`cURL`):**
```bash
curl -X POST "http://localhost:8000/api/v1/moderation/check" \
  -F "file=@.assets/examples/source.jpg"
```

**Example Response when Safe (`200 OK`):**
```json
{
  "status": "approved",
  "restricted_18_plus": false,
  "allowed": true,
  "media_type": "image",
  "message": "Safe content (SFW) — allowed.",
  "models": {
    "nsfw_1_erax": { "flagged": false, "score": 0.0012, "threshold": 0.2 },
    "nsfw_2_marqo": { "flagged": false, "score": -4.812, "threshold": 0.25 },
    "nsfw_3_freepik": { "flagged": false, "score": -12.44, "threshold": 10.5 }
  }
}
```

**Example Response when 18+ / NSFW Detected (`403 Forbidden` — Also Returned by `/api/v1/swap/image` & `/api/v1/jobs`):**
```json
{
  "status": "rejected",
  "restricted_18_plus": true,
  "rejected_input": "source",
  "rejected_file": "img_92483f8b_src.jpg",
  "message": "🚫 18+ Content Restricted: Adult / NSFW (18+) content is strictly prohibited. Your photo/video has been rejected and cannot be uploaded or processed. (18+ content upload ya run karna mana hai — photo reject kar di gayi hai.)",
  "models": {
    "nsfw_1_erax": { "flagged": true, "score": 0.912, "threshold": 0.2 },
    "nsfw_2_marqo": { "flagged": true, "score": 0.845, "threshold": 0.25 },
    "nsfw_3_freepik": { "flagged": true, "score": 14.21, "threshold": 10.5 }
  }
}
```

---

### Endpoint 3: Instant Synchronous Image Face Swap
- **Method & Path**: `POST /api/v1/swap/image`
- **URL to Hit**: [`swap_image_instant`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/api_server.py#L425-L505) -> `http://localhost:8000/api/v1/swap/image`
- **Content-Type**: `multipart/form-data`
- **What It Does**: Immediately swaps the face from `source` onto `target` using warm GPU models (~1.21s on warm requests) and returns the result in the same HTTP response.

#### Form Data Parameters (`multipart/form-data`)

| Field Name | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `source` | `File` | **Yes** | — | Source face image (`.jpg`, `.jpeg`, `.png`, `.webp`) |
| `target` | `File` | **Yes** | — | Target image to put the face onto (`.jpg`, `.jpeg`, `.png`, `.webp`) |
| `face_swapper_model` | `string` | No | `inswapper_128_fp16` | Model from `/api/v1/models` (`inswapper_128_fp16`, `hyperswap_1a_256`, etc.) |
| `enhance_face` | `boolean` | No | `false` | Set `true` to run 512×512 face restoration (`gfpgan_1.4`) after swapping |
| `face_enhancer_model` | `string` | No | `gfpgan_1.4` | Face enhancer model (`gfpgan_1.4`, `codeformer`, etc.) |
| `face_enhancer_blend` | `integer` | No | `80` | Blend strength of face enhancer (`0` to `100`) |
| `occlusion_mask` | `boolean` | No | `true` | Uses `xseg_1` occlusion mask so hair/glasses/hands in front of face are preserved |
| `response_format` | `string` | No | `json` | Choose `"json"` (URL + metadata), `"file"` (direct image binary), or `"base64"` |

**Example Request (`cURL`):**
```bash
curl -X POST "http://localhost:8000/api/v1/swap/image" \
  -F "source=@.assets/examples/source.jpg" \
  -F "target=@.assets/examples/target-1080p.jpg" \
  -F "face_swapper_model=inswapper_128_fp16" \
  -F "enhance_face=false" \
  -F "occlusion_mask=true" \
  -F "response_format=json"
```

**Example Response (`200 OK` when `response_format=json`):**
```json
{
  "job_id": "img_92483f8baa72",
  "status": "completed",
  "processing_time_seconds": 1.21,
  "download_url": "http://localhost:8000/outputs/img_92483f8baa72_out.jpg",
  "relative_url": "/outputs/img_92483f8baa72_out.jpg",
  "options": {
    "face_swapper_model": "inswapper_128_fp16",
    "enhance_face": false,
    "face_enhancer_model": null,
    "occlusion_mask": true
  }
}
```

---

### Endpoint 4: Submit Scalable Async Job (Images & Videos)
- **Method & Path**: `POST /api/v1/jobs`
- **URL to Hit**: [`submit_job`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/api_server.py#L508-L586) -> `http://localhost:8000/api/v1/jobs`
- **Content-Type**: `multipart/form-data`
- **What It Does**: Places an **image or video** face swap job into the scalable background queue and immediately returns `202 Accepted` with a `job_id`. Best for **video files (`.mp4`, `.mov`, `.webm`)** or when many users send requests at the same time.

#### Form Data Parameters (`multipart/form-data`)

| Field Name | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `source` | `File` | **Yes** | — | Source face image (`.jpg`, `.png`, `.webp`) |
| `target` | `File` | **Yes** | — | Target **image** (`.jpg`, `.png`) OR **video** (`.mp4`, `.mov`, `.webm`, `.avi`, `.mkv`) |
| `face_swapper_model` | `string` | No | `inswapper_128_fp16` | Face swapper model name |
| `enhance_face` | `boolean` | No | `false` | Enable GFPGAN face restoration |
| `face_enhancer_model` | `string` | No | `gfpgan_1.4` | Face enhancer model name |
| `face_enhancer_blend` | `integer` | No | `80` | Face enhancer blend (`0`–`100`) |
| `occlusion_mask` | `boolean` | No | `true` | Enable `xseg_1` occlusion mask |
| `trim_frame_start` | `integer` | No | `null` | Optional start frame number for video trimming |
| `trim_frame_end` | `integer` | No | `null` | Optional end frame number for video trimming |
| `webhook_url` | `string` | No | `null` | Optional URL where the server will `POST` the completed job JSON automatically |

**Example Request (`cURL`):**
```bash
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -F "source=@.assets/examples/source.jpg" \
  -F "target=@.assets/examples/target-240p.mp4" \
  -F "face_swapper_model=inswapper_128_fp16" \
  -F "trim_frame_start=0" \
  -F "trim_frame_end=50"
```

**Example Response (`202 Accepted`):**
```json
{
  "job_id": "job_af76fac64a16",
  "status": "queued",
  "workflow_type": "video",
  "created_at": "2026-11-04T18:20:48.557195+00:00",
  "started_at": null,
  "completed_at": null,
  "processing_time_seconds": null,
  "queue_position": 1,
  "status_url": "/api/v1/jobs/job_af76fac64a16",
  "result_url": null,
  "download_url": null,
  "error": null,
  "options": {
    "face_swapper_model": "inswapper_128_fp16",
    "enhance_face": false,
    "face_enhancer_model": "gfpgan_1.4",
    "occlusion_mask": true
  }
}
```

---

### Endpoint 5: Check Job Status
- **Method & Path**: `GET /api/v1/jobs/{job_id}`
- **URL to Hit**: [`get_job_status`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/api_server.py#L596-L602) -> `http://localhost:8000/api/v1/jobs/{job_id}`
- **What It Does**: Returns the live status (`queued`, `processing`, `completed`, `failed`), execution time, and download URLs for `{job_id}`.

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/jobs/job_af76fac64a16"
```

**Example Response (`200 OK` when completed):**
```json
{
  "job_id": "job_af76fac64a16",
  "status": "completed",
  "workflow_type": "video",
  "created_at": "2026-11-04T18:20:48.557195+00:00",
  "started_at": "2026-11-04T18:20:48.989944+00:00",
  "completed_at": "2026-11-04T18:21:14.717553+00:00",
  "processing_time_seconds": 25.73,
  "queue_position": null,
  "status_url": "/api/v1/jobs/job_af76fac64a16",
  "result_url": "/api/v1/jobs/job_af76fac64a16/result",
  "download_url": "/outputs/job_af76fac64a16_out.mp4",
  "error": null,
  "options": {
    "face_swapper_model": "inswapper_128_fp16",
    "enhance_face": false,
    "face_enhancer_model": "gfpgan_1.4",
    "occlusion_mask": true
  }
}
```

---

### Endpoint 6: Download Job Result File
- **Method & Path**: `GET /api/v1/jobs/{job_id}/result`
- **URL to Hit**: [`download_job_result`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/api_server.py#L605-L622) -> `http://localhost:8000/api/v1/jobs/{job_id}/result`
- **What It Does**: Streams/downloads the finished `.jpg`, `.png`, or `.mp4` file directly. (Returns `409 Conflict` if the job is still `queued` or `processing`.)

**Example Request:**
```bash
curl -L "http://localhost:8000/api/v1/jobs/job_af76fac64a16/result" -o swapped_result.mp4
```

---

### Endpoint 7: List All Recent Jobs
- **Method & Path**: `GET /api/v1/jobs?limit=20`
- **URL to Hit**: [`list_jobs`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/api_server.py#L589-L593) -> `http://localhost:8000/api/v1/jobs?limit=20`
- **What It Does**: Returns a list of the most recent jobs (up to `limit`, default `20`, max `100`).

---

### Endpoint 8: Cancel / Delete Job & Clean Up Files
- **Method & Path**: `DELETE /api/v1/jobs/{job_id}`
- **URL to Hit**: [`delete_job`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/api_server.py#L625-L642) -> `http://localhost:8000/api/v1/jobs/{job_id}`
- **What It Does**: Cancels a queued job or deletes a finished job and removes its uploaded source, target, and output files from `api_storage/`.

**Example Request:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/jobs/job_af76fac64a16"
```

---

### Endpoint 9: Direct Static Output File Access
- **Method & Path**: `GET /outputs/{filename}`
- **URL to Hit**: `http://localhost:8000/outputs/{filename}`
- **What It Does**: Directly serves any generated image or video from [`api_storage/outputs/`](file:///c:/Users/M4%20Tech/Desktop/New%20folder/api_storage/outputs) so you can embed it directly in an HTML `<img>` or `<video>` tag on your website.

---

## 4. Integration Examples (JavaScript / PHP for Hostinger Websites)

### JavaScript / React / Frontend (`fetch`)
```javascript
async function swapFaceInstant(sourceFile, targetFile) {
  const formData = new FormData();
  formData.append("source", sourceFile);
  formData.append("target", targetFile);
  formData.append("face_swapper_model", "inswapper_128_fp16");
  formData.append("enhance_face", "false");
  formData.append("occlusion_mask", "true");
  formData.append("response_format", "json");

  const response = await fetch("http://localhost:8000/api/v1/swap/image", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  console.log("Swapped in", data.processing_time_seconds, "s ->", data.download_url);
  return data.download_url;
}
```

### PHP (`cURL` — for Hostinger WordPress / PHP Websites)
```php
<?php
$apiUrl = "http://YOUR_SERVER_IP:8000/api/v1/swap/image";

$ch = curl_init($apiUrl);
$postFields = [
    'source' => new CURLFile($_FILES['source']['tmp_name'], $_FILES['source']['type'], $_FILES['source']['name']),
    'target' => new CURLFile($_FILES['target']['tmp_name'], $_FILES['target']['type'], $_FILES['target']['name']),
    'face_swapper_model' => 'inswapper_128_fp16',
    'enhance_face' => 'false',
    'occlusion_mask' => 'true',
    'response_format' => 'json',
];

curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => $postFields,
    CURLOPT_RETURNTRANSFER => true,
]);

$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true);
echo "Result Image URL: " . $result['download_url'];
?>
```
