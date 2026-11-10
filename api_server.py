#!/usr/bin/env python3
"""
Scalable FastAPI REST API Server for FaceFusion + facefusion-assets
Supports:
- Warm In-Process GPU/CPU Inference Engine (keeps ONNX models hot in memory for 1-2s image swaps)
- Async Job Queue & Worker Pool (prevents GPU VRAM OOM under concurrent load)
- Synchronous Image Swap endpoint (/api/v1/swap/image)
- Asynchronous Image & Video Job Queue endpoints (/api/v1/jobs, /api/v1/jobs/{job_id}, /api/v1/jobs/{job_id}/result)
- Optional API Key Authentication (set FACEFUSION_API_KEY env var)
"""

import asyncio
import base64
import os
import shutil
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
import onnxruntime
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure .venv/Scripts (including bundled ffmpeg.exe & ffprobe.exe) is on PATH
os.environ["OMP_NUM_THREADS"] = "1"
_venv_scripts = os.path.dirname(os.path.abspath(sys.executable))
if _venv_scripts not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_scripts + os.pathsep + os.environ.get("PATH", "")

from facefusion import content_analyser, content_moderation, core, inference_manager, logger, state_manager
from facefusion.execution import get_available_execution_providers
from facefusion.face_store import clear_faces
from facefusion.vision import read_static_image, read_static_video_frame

# ------------------------------------------------------------------------------
# Configuration & Directories
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
API_STORAGE_DIR = BASE_DIR / "api_storage"
UPLOADS_DIR = API_STORAGE_DIR / "uploads"
OUTPUTS_DIR = API_STORAGE_DIR / "outputs"
MODELS_DIR = BASE_DIR / ".assets" / "models"
STATIC_DIR = BASE_DIR / "static"

for d in (UPLOADS_DIR, OUTPUTS_DIR, MODELS_DIR, STATIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("FACEFUSION_API_KEY", "").strip()
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
DEFAULT_MEMORY_STRATEGY = os.getenv("FACEFUSION_MEMORY_STRATEGY", "tolerant")

# Detect best default execution providers automatically
AVAILABLE_PROVIDERS = get_available_execution_providers()
if "cuda" in AVAILABLE_PROVIDERS:
    DEFAULT_PROVIDERS = ["cuda", "cpu"]
elif "directml" in AVAILABLE_PROVIDERS:
    DEFAULT_PROVIDERS = ["directml", "cpu"]
else:
    DEFAULT_PROVIDERS = ["cpu"]

# ------------------------------------------------------------------------------
# In-Memory Job Store & Async Worker Queue
# ------------------------------------------------------------------------------
JOBS: Dict[str, Dict[str, Any]] = {}
JOB_QUEUE: asyncio.Queue[str] = asyncio.Queue()
INFERENCE_LOCK = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
WORKER_TASKS: List[asyncio.Task] = []
VALIDATED_MODEL_COMBOS: set[str] = set()


def init_facefusion_engine() -> None:
    """Initialize FaceFusion's state_manager directly in <1ms without spawning subprocesses."""
    import tempfile
    import facefusion.choices

    defaults = {
        "command": "headless-run",
        "config_path": "facefusion.ini",
        "temp_path": tempfile.gettempdir(),
        "jobs_path": ".jobs",
        "source_paths": [],
        "target_path": None,
        "output_path": None,
        "download_providers": ["github", "huggingface"],
        "download_scope": "lite",
        "face_detector_model": "yolo_face",
        "face_detector_size": "640x640",
        "face_detector_margin": [0, 0, 0, 0],
        "face_detector_angles": [0],
        "face_detector_score": 0.5,
        "face_landmarker_model": "2dfan4",
        "face_landmarker_score": 0.5,
        "face_selector_mode": "reference",
        "face_selector_order": "large-small",
        "face_selector_age_start": None,
        "face_selector_age_end": None,
        "face_selector_gender": None,
        "face_selector_race": None,
        "reference_face_position": 0,
        "reference_face_distance": 0.3,
        "reference_frame_number": 0,
        "face_tracker_score": 0.0,
        "face_occluder_model": "xseg_1",
        "face_parser_model": "bisenet_resnet_34",
        "face_mask_types": ["box", "occlusion"],
        "face_mask_areas": list(facefusion.choices.face_mask_areas),
        "face_mask_regions": list(facefusion.choices.face_mask_regions),
        "face_mask_blur": 0.3,
        "face_mask_padding": [0, 0, 0, 0],
        "voice_extractor_model": "kim_vocal_2",
        "trim_frame_start": None,
        "trim_frame_end": None,
        "temp_frame_format": "png",
        "temp_pixel_format": "bgr24",
        "target_frame_amount": 2,
        "output_image_quality": 80,
        "output_image_scale": 1.0,
        "output_audio_encoder": "aac",
        "output_audio_quality": 80,
        "output_audio_volume": 100,
        "output_video_encoder": "libx264",
        "output_video_preset": "veryfast",
        "output_video_quality": 80,
        "output_video_scale": 1.0,
        "output_video_fps": 25.0,
        "workflow_mode": "auto",
        "workflow_strategy": "memory",
        "processors": ["face_swapper"],
        "face_swapper_model": "inswapper_128_fp16",
        "face_swapper_pixel_boost": "128x128",
        "face_swapper_weight": 0.5,
        "face_enhancer_model": "gfpgan_1.4",
        "face_enhancer_blend": 80,
        "face_enhancer_weight": 0.5,
        "execution_device_ids": [0],
        "execution_providers": DEFAULT_PROVIDERS,
        "execution_thread_count": 4,
        "video_memory_strategy": DEFAULT_MEMORY_STRATEGY,
        "log_level": "info",
        "halt_on_error": False,
    }
    for key, val in defaults.items():
        state_manager.init_item(key, val)
    logger.init("info")


def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """Optional API Key security check if FACEFUSION_API_KEY environment variable is set."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


def execute_facefusion_job_sync(job_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a FaceFusion image or video job synchronously inside the worker thread.
    Keeps ONNX models warm in `inference_manager.INFERENCE_POOLS` when memory_strategy='tolerant'
    while clearing per-request face caches so RAM never leaks.
    """
    start_time = time.perf_counter()

    processors = job_params.get("processors")
    if not processors:
        processors = ["face_swapper"]
        if job_params.get("enhance_face"):
            processors.append("face_enhancer")

    face_mask_types = ["box", "occlusion"] if job_params.get("occlusion_mask", True) else ["box"]
    source_paths = [job_params["source_path"]] if job_params.get("source_path") else []

    # Configure state_manager for this job
    state_manager.set_item("source_paths", source_paths)
    state_manager.set_item("target_path", job_params["target_path"])
    state_manager.set_item("output_path", job_params["output_path"])
    state_manager.set_item("processors", processors)
    state_manager.set_item("face_selector_mode", job_params.get("face_selector_mode", "many" if not source_paths else "reference"))
    if "face_swapper" in processors:
        state_manager.set_item("face_swapper_model", job_params.get("face_swapper_model", "inswapper_128_fp16"))
        state_manager.set_item("face_swapper_pixel_boost", job_params.get("face_swapper_pixel_boost", "128x128"))
    if "face_enhancer" in processors:
        state_manager.set_item("face_enhancer_model", job_params.get("face_enhancer_model", "gfpgan_1.4"))
        state_manager.set_item("face_enhancer_blend", int(job_params.get("face_enhancer_blend", 80)))
    if "frame_colorizer" in processors:
        state_manager.set_item("frame_colorizer_model", job_params.get("frame_colorizer_model", "ddcolor"))
        state_manager.set_item("frame_colorizer_blend", int(job_params.get("frame_colorizer_blend", 100)))
        state_manager.set_item("frame_colorizer_size", job_params.get("frame_colorizer_size", "256x256"))
    if "frame_enhancer" in processors:
        state_manager.set_item("frame_enhancer_model", job_params.get("frame_enhancer_model", "span_kendata_x4"))
        state_manager.set_item("frame_enhancer_blend", int(job_params.get("frame_enhancer_blend", 100)))

    state_manager.set_item("face_mask_types", face_mask_types)
    state_manager.set_item("workflow_mode", "auto")
    state_manager.set_item("execution_providers", job_params.get("execution_providers") or DEFAULT_PROVIDERS)
    state_manager.set_item(
        "video_memory_strategy",
        job_params.get("video_memory_strategy") or DEFAULT_MEMORY_STRATEGY,
    )
    state_manager.set_item("trim_frame_start", job_params.get("trim_frame_start"))
    state_manager.set_item("trim_frame_end", job_params.get("trim_frame_end"))

    try:
        # Clear previous frame/face caches (keeps ONNX sessions warm!)
        clear_faces()
        read_static_image.cache_clear()
        read_static_video_frame.cache_clear()
        content_analyser.analyse_image.cache_clear()
        content_analyser.analyse_video.cache_clear()

        combo_key = f"{','.join(processors)}:{state_manager.get_item('face_swapper_model')}:{state_manager.get_item('face_enhancer_model')}"
        if combo_key not in VALIDATED_MODEL_COMBOS:
            if not core.common_pre_check() or not core.processors_pre_check():
                return {
                    "success": False,
                    "error_code": 2,
                    "error": "Model pre-check or download failed.",
                    "processing_time_seconds": round(time.perf_counter() - start_time, 2),
                }
            VALIDATED_MODEL_COMBOS.add(combo_key)

        error_code = core.conditional_process()
        elapsed = round(time.perf_counter() - start_time, 2)

        if error_code == 0 and os.path.isfile(job_params["output_path"]):
            return {
                "success": True,
                "error_code": 0,
                "processing_time_seconds": elapsed,
            }
        else:
            error_messages = {
                1: "General processing error.",
                2: "Pre-process validation failed (e.g. no face detected in source/target).",
                3: content_moderation.REJECTION_MESSAGE,
                4: "Processing was interrupted.",
            }
            return {
                "success": False,
                "error_code": error_code,
                "error": error_messages.get(error_code, f"Processing exited with code {error_code}"),
                "processing_time_seconds": elapsed,
            }
    finally:
        clear_faces()
        read_static_image.cache_clear()
        read_static_video_frame.cache_clear()


async def queue_worker(worker_id: int) -> None:
    """Background worker that processes queued jobs sequentially/scalable via semaphore."""
    while True:
        job_id = await JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if not job or job["status"] == "cancelled":
            JOB_QUEUE.task_done()
            continue

        job["status"] = "processing"
        job["started_at"] = datetime.now(timezone.utc).isoformat()

        async with INFERENCE_LOCK:
            try:
                result = await asyncio.to_thread(execute_facefusion_job_sync, job["params"])
                if result["success"]:
                    job["status"] = "completed"
                    job["processing_time_seconds"] = result["processing_time_seconds"]
                    job["completed_at"] = datetime.now(timezone.utc).isoformat()
                else:
                    job["status"] = "failed"
                    job["error"] = result.get("error", "Unknown error")
                    job["error_code"] = result.get("error_code", 1)
                    job["processing_time_seconds"] = result.get("processing_time_seconds")
                    job["completed_at"] = datetime.now(timezone.utc).isoformat()
            except Exception as exc:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Trigger optional webhook callback if provided
        webhook_url = job["params"].get("webhook_url")
        if webhook_url:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(webhook_url, json=serialize_job(job))
            except Exception:
                pass

        JOB_QUEUE.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_facefusion_engine()
    for i in range(MAX_CONCURRENT_JOBS):
        task = asyncio.create_task(queue_worker(i))
        WORKER_TASKS.append(task)
    yield
    for task in WORKER_TASKS:
        task.cancel()


# ------------------------------------------------------------------------------
# FastAPI App Initialization
# ------------------------------------------------------------------------------
app = FastAPI(
    title="FaceFusion Scalable REST API",
    description=(
        "Production-ready FastAPI server for FaceFusion (`facefusion-assets`). "
        "Supports instant synchronous image face swapping (`/api/v1/swap/image`) "
        "and a scalable async job queue for images & videos (`/api/v1/jobs`)."
    ),
    version="3.9.2",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ------------------------------------------------------------------------------
# Pydantic Response Models
# ------------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "3.9.2"
    onnxruntime_version: str
    onnxruntime_providers: List[str]
    active_execution_providers: List[str]
    video_memory_strategy: str
    max_concurrent_jobs: int
    queued_jobs: int
    processing_jobs: int
    warm_loaded_models: List[str]
    downloaded_models_count: int


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed", "cancelled"]
    workflow_type: Literal["image", "video"]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    queue_position: Optional[int] = None
    status_url: str
    result_url: Optional[str] = None
    download_url: Optional[str] = None
    error: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


def serialize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = job["job_id"]
    is_done = job["status"] == "completed"
    output_filename = os.path.basename(job["params"]["output_path"])
    return {
        "job_id": job_id,
        "status": job["status"],
        "workflow_type": job["workflow_type"],
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "processing_time_seconds": job.get("processing_time_seconds"),
        "queue_position": job.get("queue_position", 0) if job["status"] == "queued" else None,
        "status_url": f"/api/v1/jobs/{job_id}",
        "result_url": f"/api/v1/jobs/{job_id}/result" if is_done else None,
        "download_url": f"/outputs/{output_filename}" if is_done else None,
        "error": job.get("error"),
        "options": {
            "face_swapper_model": job["params"].get("face_swapper_model"),
            "enhance_face": job["params"].get("enhance_face"),
            "face_enhancer_model": job["params"].get("face_enhancer_model"),
            "occlusion_mask": job["params"].get("occlusion_mask"),
        },
    }


def save_upload_file(upload: UploadFile, prefix: str) -> str:
    ext = Path(upload.filename or "file.jpg").suffix.lower() or ".jpg"
    file_path = UPLOADS_DIR / f"{prefix}_{uuid.uuid4().hex[:10]}{ext}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return str(file_path)


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return RedirectResponse(url="/docs")


@app.get("/facefusion.ico", include_in_schema=False)
async def favicon():
    ico_path = BASE_DIR / "facefusion.ico"
    if ico_path.is_file():
        return FileResponse(str(ico_path), media_type="image/x-icon")
    raise HTTPException(status_code=404)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health, GPU execution providers, warm models in memory, and queue metrics."""
    downloaded = sorted(f.name for f in MODELS_DIR.glob("*.onnx"))
    warm_models = list(inference_manager.INFERENCE_POOL_SET.get("cli", {}).keys()) + list(
        inference_manager.INFERENCE_POOL_SET.get("ui", {}).keys()
    )
    queued_count = sum(1 for j in JOBS.values() if j["status"] == "queued")
    processing_count = sum(1 for j in JOBS.values() if j["status"] == "processing")

    return HealthResponse(
        status="ok",
        version="3.9.2",
        onnxruntime_version=onnxruntime.__version__,
        onnxruntime_providers=onnxruntime.get_available_providers(),
        active_execution_providers=DEFAULT_PROVIDERS,
        video_memory_strategy=DEFAULT_MEMORY_STRATEGY,
        max_concurrent_jobs=MAX_CONCURRENT_JOBS,
        queued_jobs=queued_count,
        processing_jobs=processing_count,
        warm_loaded_models=warm_models,
        downloaded_models_count=len(downloaded),
    )


@app.get("/api/v1/models", tags=["System"])
async def list_models():
    """List downloaded `facefusion-assets` ONNX models on disk and supported model choices."""
    from facefusion.processors.modules.face_enhancer import choices as enhancer_choices
    from facefusion.processors.modules.face_swapper import choices as swapper_choices

    downloaded = [
        {
            "name": f.name,
            "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
        }
        for f in sorted(MODELS_DIR.glob("*.onnx"))
    ]
    return {
        "downloaded_models": downloaded,
        "available_face_swapper_models": swapper_choices.face_swapper_models,
        "available_face_enhancer_models": enhancer_choices.face_enhancer_models,
        "recommended_for_4gb_gpu": {
            "face_swapper_model": "inswapper_128_fp16",
            "face_enhancer_model": "gfpgan_1.4",
        },
    }


@app.post("/api/v1/moderation/check", tags=["18+ Content Moderation"], dependencies=[Depends(verify_api_key)])
async def check_content_moderation(
    file: UploadFile = File(..., description="Image or video file to check for 18+ / NSFW content"),
):
    """
    **18+ / NSFW Content Moderation Endpoint.**
    Scans an uploaded image or video with the 3-model ONNX NSFW ensemble (`nsfw_1` EraX, `nsfw_2` Marqo, `nsfw_3` Freepik).
    Returns `200 OK` if safe, or `403 Forbidden` with the 18+ rejection message if adult content is detected.
    """
    temp_id = f"mod_{uuid.uuid4().hex[:10]}"
    temp_path = save_upload_file(file, temp_id)
    try:
        async with INFERENCE_LOCK:
            result = await asyncio.to_thread(
                content_moderation.check_media_18_plus,
                temp_path,
                None,
                None,
                True,
            )
    finally:
        if os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if result["is_18_plus"]:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "status": "rejected",
                "restricted_18_plus": True,
                "allowed": False,
                "media_type": result.get("media_type"),
                "message": content_moderation.REJECTION_MESSAGE,
                "models": result.get("models"),
            },
        )

    return {
        "status": "approved",
        "restricted_18_plus": False,
        "allowed": True,
        "media_type": result.get("media_type"),
        "message": result.get("message"),
        "models": result.get("models"),
    }


@app.post("/api/v1/enhance/photo", tags=["Photo Enhancement"])
async def enhance_photo_instant(
    request: Request,
    file: UploadFile = File(..., description="Photo to restore and enhance (.jpg, .jpeg, .png, .webp, .bmp)"),
    api_key: Optional[str] = Form(None, description="API Key (can also be passed in X-API-Key header)"),
    face_enhancer_model: str = Form("gfpgan_1.4", description="Face restoration model"),
    face_enhancer_blend: int = Form(80, description="Enhancement blend percentage (0-100)"),
    colorize: bool = Form(False, description="Enable photo colorization for B&W photos (DDColor)"),
    upscale: bool = Form(False, description="Enable whole-image super-resolution (SPAN x4)"),
    response_format: Literal["json", "file", "base64"] = Form("json"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """
    **Instant Photo Restoration & Enhancement Endpoint.**
    Upload an old, blurry, or low-resolution photo.
    Restores facial details (GFPGAN), optionally colorizes B&W photos, and upscales resolution.
    """
    provided_key = (x_api_key or api_key or "").strip()
    if API_KEY and provided_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

    job_id = f"photo_{uuid.uuid4().hex[:12]}"
    target_path = save_upload_file(file, f"{job_id}_raw")
    target_ext = Path(target_path).suffix.lower()

    if target_ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        if os.path.isfile(target_path):
            os.remove(target_path)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{target_ext}'. Only .jpg, .png, .webp, .bmp are supported.",
        )

    # 18+ / NSFW Upload Gate
    async with INFERENCE_LOCK:
        mod_check = await asyncio.to_thread(
            content_moderation.check_media_18_plus,
            target_path,
            None,
            None,
            True,
        )

    if mod_check["is_18_plus"]:
        if os.path.isfile(target_path):
            os.remove(target_path)
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "status": "rejected",
                "restricted_18_plus": True,
                "message": content_moderation.REJECTION_MESSAGE,
            },
        )

    output_filename = f"{job_id}_enhanced{target_ext}"
    output_path = str(OUTPUTS_DIR / output_filename)

    processors = ["face_enhancer"]
    if colorize:
        processors.append("frame_colorizer")
    if upscale:
        processors.append("frame_enhancer")

    job_params = {
        "source_path": None,
        "target_path": target_path,
        "output_path": output_path,
        "processors": processors,
        "face_selector_mode": "many",
        "face_enhancer_model": face_enhancer_model,
        "face_enhancer_blend": face_enhancer_blend,
        "frame_colorizer_model": "ddcolor",
        "frame_enhancer_model": "span_kendata_x4",
    }

    async with INFERENCE_LOCK:
        result = await asyncio.to_thread(execute_facefusion_job_sync, job_params)

    if not result["success"]:
        status_code = status.HTTP_403_FORBIDDEN if result.get("error_code") == 3 else status.HTTP_422_UNPROCESSABLE_ENTITY
        raise HTTPException(status_code=status_code, detail=result)

    if response_format == "file":
        media_type = "image/png" if target_ext == ".png" else "image/jpeg"
        return FileResponse(output_path, media_type=media_type, filename=output_filename)

    base_url = str(request.base_url).rstrip("/")
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "status": "completed",
        "processing_time_seconds": result["processing_time_seconds"],
        "download_url": f"{base_url}/outputs/{output_filename}",
        "relative_url": f"/outputs/{output_filename}",
        "original_url": f"{base_url}/uploads/{Path(target_path).name}",
        "processors": processors,
    }

    if response_format == "base64":
        with open(output_path, "rb") as f:
            payload["image_base64"] = base64.b64encode(f.read()).decode("utf-8")

    return JSONResponse(content=payload)


@app.post("/api/v1/swap/image", tags=["Instant Face Swap"], dependencies=[Depends(verify_api_key)])
async def swap_image_instant(
    request: Request,
    source: UploadFile = File(..., description="Source image containing the face to swap from (.jpg, .png, .webp)"),
    target: UploadFile = File(..., description="Target image to swap the face onto (.jpg, .png, .webp)"),
    face_swapper_model: str = Form("inswapper_128_fp16", description="Face swapper model name"),
    enhance_face: bool = Form(False, description="Enable GFPGAN/CodeFormer face restoration (512x512)"),
    face_enhancer_model: str = Form("gfpgan_1.4", description="Face enhancer model name"),
    face_enhancer_blend: int = Form(80, description="Face enhancer blend percentage (0-100)"),
    occlusion_mask: bool = Form(True, description="Enable XSeg occlusion mask (preserves hair, glasses, hands)"),
    response_format: Literal["json", "file", "base64"] = Form(
        "json",
        description="Return JSON metadata + URL ('json'), raw image file ('file'), or JSON with base64 image ('base64')",
    ),
):
    """
    **Synchronous / Instant Image Face Swap Endpoint.**
    First verifies that neither `source` nor `target` contains 18+ / NSFW content (deletes & rejects with `403 Forbidden` if detected), then processes the swap using warm GPU/CPU models.
    """
    job_id = f"img_{uuid.uuid4().hex[:12]}"
    source_path = save_upload_file(source, f"{job_id}_src")
    target_path = save_upload_file(target, f"{job_id}_tgt")

    target_ext = Path(target_path).suffix.lower()
    if target_ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        for p in (source_path, target_path):
            if os.path.isfile(p):
                os.remove(p)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target image extension '{target_ext}'. For videos, use POST /api/v1/jobs.",
        )

    # 18+ / NSFW Upload Gate: block & delete immediately if either source or target is 18+
    async with INFERENCE_LOCK:
        mod_check = await asyncio.to_thread(
            content_moderation.check_source_and_target,
            [source_path],
            target_path,
            None,
            None,
            True,
        )

    if mod_check["is_18_plus"]:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "status": "rejected",
                "restricted_18_plus": True,
                "rejected_input": mod_check.get("rejected_input"),
                "rejected_file": mod_check.get("rejected_file"),
                "message": content_moderation.REJECTION_MESSAGE,
                "models": mod_check.get("models"),
            },
        )

    output_filename = f"{job_id}_out{target_ext}"
    output_path = str(OUTPUTS_DIR / output_filename)

    job_params = {
        "source_path": source_path,
        "target_path": target_path,
        "output_path": output_path,
        "face_swapper_model": face_swapper_model,
        "face_swapper_pixel_boost": "128x128" if "128" in face_swapper_model else "256x256",
        "enhance_face": enhance_face,
        "face_enhancer_model": face_enhancer_model,
        "face_enhancer_blend": face_enhancer_blend,
        "occlusion_mask": occlusion_mask,
    }

    async with INFERENCE_LOCK:
        result = await asyncio.to_thread(execute_facefusion_job_sync, job_params)

    if not result["success"]:
        status_code = status.HTTP_403_FORBIDDEN if result.get("error_code") == 3 else status.HTTP_422_UNPROCESSABLE_ENTITY
        raise HTTPException(status_code=status_code, detail=result)

    if response_format == "file":
        media_type = "image/png" if target_ext == ".png" else "image/jpeg"
        return FileResponse(output_path, media_type=media_type, filename=output_filename)

    base_url = str(request.base_url).rstrip("/")
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "status": "completed",
        "processing_time_seconds": result["processing_time_seconds"],
        "download_url": f"{base_url}/outputs/{output_filename}",
        "relative_url": f"/outputs/{output_filename}",
        "options": {
            "face_swapper_model": face_swapper_model,
            "enhance_face": enhance_face,
            "face_enhancer_model": face_enhancer_model if enhance_face else None,
            "occlusion_mask": occlusion_mask,
        },
    }

    if response_format == "base64":
        with open(output_path, "rb") as f:
            payload["image_base64"] = base64.b64encode(f.read()).decode("utf-8")

    return JSONResponse(content=payload)


@app.post(
    "/api/v1/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Scalable Async Job Queue"],
    dependencies=[Depends(verify_api_key)],
)
async def submit_job(
    source: UploadFile = File(..., description="Source face image (.jpg, .png, .webp)"),
    target: UploadFile = File(..., description="Target image (.jpg, .png) OR target video (.mp4, .mov, .webm)"),
    face_swapper_model: str = Form("inswapper_128_fp16", description="Face swapper model name"),
    enhance_face: bool = Form(False, description="Enable GFPGAN face restoration"),
    face_enhancer_model: str = Form("gfpgan_1.4", description="Face enhancer model name"),
    face_enhancer_blend: int = Form(80, description="Face enhancer blend percentage (0-100)"),
    occlusion_mask: bool = Form(True, description="Enable XSeg occlusion mask"),
    trim_frame_start: Optional[int] = Form(None, description="Optional start frame for video trimming"),
    trim_frame_end: Optional[int] = Form(None, description="Optional end frame for video trimming"),
    webhook_url: Optional[str] = Form(None, description="Optional webhook URL to POST job result when completed"),
):
    """
    **Scalable Asynchronous Job Submission (Images & Videos).**
    Enqueues the face swap task and immediately returns `202 Accepted` with a `job_id` so your frontend or microservice can poll `/api/v1/jobs/{job_id}` or receive a webhook.
    """
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    source_path = save_upload_file(source, f"{job_id}_src")
    target_path = save_upload_file(target, f"{job_id}_tgt")

    # 18+ / NSFW Upload Gate: block & delete immediately before queueing if either source or target is 18+
    async with INFERENCE_LOCK:
        mod_check = await asyncio.to_thread(
            content_moderation.check_source_and_target,
            [source_path],
            target_path,
            trim_frame_start,
            trim_frame_end,
            True,
        )

    if mod_check["is_18_plus"]:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "status": "rejected",
                "restricted_18_plus": True,
                "rejected_input": mod_check.get("rejected_input"),
                "rejected_file": mod_check.get("rejected_file"),
                "message": content_moderation.REJECTION_MESSAGE,
                "models": mod_check.get("models"),
            },
        )

    target_ext = Path(target_path).suffix.lower()
    is_video_target = target_ext in [".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"]
    workflow_type: Literal["image", "video"] = "video" if is_video_target else "image"

    output_filename = f"{job_id}_out{target_ext}"
    output_path = str(OUTPUTS_DIR / output_filename)

    job_params = {
        "source_path": source_path,
        "target_path": target_path,
        "output_path": output_path,
        "face_swapper_model": face_swapper_model,
        "face_swapper_pixel_boost": "128x128" if "128" in face_swapper_model else "256x256",
        "enhance_face": enhance_face,
        "face_enhancer_model": face_enhancer_model,
        "face_enhancer_blend": face_enhancer_blend,
        "occlusion_mask": occlusion_mask,
        "trim_frame_start": trim_frame_start,
        "trim_frame_end": trim_frame_end,
        "webhook_url": webhook_url,
    }

    queue_pos = JOB_QUEUE.qsize() + 1
    job_record = {
        "job_id": job_id,
        "status": "queued",
        "workflow_type": workflow_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "processing_time_seconds": None,
        "queue_position": queue_pos,
        "params": job_params,
        "error": None,
    }
    JOBS[job_id] = job_record
    await JOB_QUEUE.put(job_id)

    return serialize_job(job_record)


@app.get("/api/v1/jobs", response_model=List[JobResponse], tags=["Scalable Async Job Queue"])
async def list_jobs(limit: int = Query(20, ge=1, le=100)):
    """List recent jobs and their statuses."""
    sorted_jobs = sorted(JOBS.values(), key=lambda x: x["created_at"], reverse=True)[:limit]
    return [serialize_job(j) for j in sorted_jobs]


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, tags=["Scalable Async Job Queue"])
async def get_job_status(job_id: str):
    """Get the status, execution time, and download URLs of a specific job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return serialize_job(job)


@app.get("/api/v1/jobs/{job_id}/result", tags=["Scalable Async Job Queue"])
async def download_job_result(job_id: str):
    """Download the processed output image or video file for a completed job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is currently '{job['status']}' and not ready for download.",
        )
    output_path = job["params"]["output_path"]
    if not os.path.isfile(output_path):
        raise HTTPException(status_code=404, detail="Output file not found on disk.")

    ext = Path(output_path).suffix.lower()
    media_type = "video/mp4" if ext == ".mp4" else ("image/png" if ext == ".png" else "image/jpeg")
    return FileResponse(output_path, media_type=media_type, filename=os.path.basename(output_path))


@app.delete("/api/v1/jobs/{job_id}", tags=["Scalable Async Job Queue"], dependencies=[Depends(verify_api_key)])
async def delete_job(job_id: str):
    """Cancel/delete a job and remove its uploaded/output files from disk."""
    job = JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    job["status"] = "cancelled"
    for key in ("source_path", "target_path", "output_path"):
        path = job["params"].get(key)
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass

    return {"deleted": True, "job_id": job_id}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api_server:app", host=host, port=port, reload=False, workers=1)
