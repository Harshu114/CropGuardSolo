"""FastAPI server for CropGuard inference."""
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from model_server import (
    predict_image,
    get_mask_path,
    get_cam_path,
    get_lesion_mask_path,
    get_model,
    get_model_version,
    load_model,
    CONF_THR,
    IMG_SIZE,
    v11_gate_status,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the v5 checkpoint.
    #
    # There is deliberately NO fallback to the legacy cropguard_model_final.pt.
    # That file has 16 classes; this build is nc=11, so load_model(strict=True)
    # would reject it anyway. Falling back would only turn a clear "v5 missing"
    # error into a confusing shape-mismatch at request time.
    api_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get("MODEL_PATH",
                                os.path.join(api_dir, "cropguard_model_v5.pt"))
    if os.path.exists(model_path):
        try:
            load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
    else:
        print(f"Warning: Model file not found at {model_path}. Set MODEL_PATH env var.")
    yield
    # Shutdown: nothing needed


app = FastAPI(
    title="CropGuard API",
    description="Tomato leaf disease diagnosis API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory rate limiter (RATE_LIMIT_PER_MIN per IP, default 60).
#
# Loopback clients are exempt: the local dev site and the diagnostic scripts run
# from 127.0.0.1, and the old hardcoded 10/min cap turned every small batch test
# into a stream of 429s while protecting nothing (there is no remote client to
# protect on localhost). Raise/lower with RATE_LIMIT_PER_MIN, or set 0 to disable.
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
RATE_LIMIT_EXEMPT_IPS = {"127.0.0.1", "::1", "localhost", "test", "testclient"}
_request_counts: dict[str, list[float]] = {}


def check_rate_limit(ip: str, limit: int = RATE_LIMIT_PER_MIN,
                     window: int = 60) -> bool:
    if limit <= 0 or ip in RATE_LIMIT_EXEMPT_IPS:
        return True
    now = time.time()
    if ip not in _request_counts:
        _request_counts[ip] = []
    # Clean old requests
    _request_counts[ip] = [t for t in _request_counts[ip] if now - t < window]
    if len(_request_counts[ip]) >= limit:
        return False
    _request_counts[ip].append(now)
    return True


def get_client_ip(request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/v1/health")
async def health():
    try:
        model = get_model()
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        return {
            "status": "ok",
            "model": get_model_version(),
            "device": device
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "model": get_model_version(), "device": "unknown", "error": str(e)}
        )


@app.post("/v1/predict")
async def predict(
    image: UploadFile = File(...),
    conf: float = Form(CONF_THR),
    with_cam: bool = Form(True),
    request: Request = None,
):
    # Rate limit
    client_ip = get_client_ip(request) if request else "test"
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Limit: {RATE_LIMIT_PER_MIN} requests per minute.")

    # Validate file
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, WebP)")

    # Read file
    file_bytes = await image.read()
    if len(file_bytes) > 10 * 1024 * 1024:  # 10 MB
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")

    # Validate with PIL
    try:
        Image.open(image.file).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

    # Run inference
    try:
        result = predict_image(file_bytes, conf_thr=conf, with_cam=with_cam)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/v1/masks/{image_id}.png")
async def get_mask(image_id: str):
    path = get_mask_path(image_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Mask not found or expired")
    return FileResponse(path, media_type="image/png")


@app.get("/v1/cams/{image_id}.png")
async def get_cam(image_id: str):
    path = get_cam_path(image_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="CAM not found or expired")
    return FileResponse(path, media_type="image/png")


@app.get("/v1/lesion_masks/{image_id}.png")
async def get_lesion_mask(image_id: str):
    """Lesion overlay from the segmentation side-model (hybrid serving).
    404 when SEG_MODEL_PATH is unset, so clients treat it as optional."""
    path = get_lesion_mask_path(image_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Lesion mask not found or expired")
    return FileResponse(path, media_type="image/png")


@app.get("/v1/gate/config")
async def gate_config():
    """Abstention-gate configuration and documentation.

    Returns the v11 field gate artifact (fitted temperature + field-test operating points +
    a suggested deployment operating point) together with its known limitations. This is
    informational until the served classifier is swapped to a v11-loadable server; until
    then /v1/predict still uses the v5 classifier and severity.reliable stays False.
    """
    return v11_gate_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

