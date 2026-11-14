import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.dicom_loader import DicomSeries, WINDOW_PRESETS
from app.nifti_loader import NiftiMask
from app.api_client import NucleoApiClient
from app.job_manager import job_manager, Job
from app.metrics import check_grid_compatibility, compute_metrics

STATIC_DIR = Path(__file__).resolve().parent / "static"

# In-memory storage for default loaded case
default_series = DicomSeries()
reference_mask = NiftiMask(name="reference_lesions", color=(16, 185, 129, 140)) # Green

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load default DICOM series if present
    if settings.default_dicom_zip.exists():
        try:
            default_series.load_from_zip(settings.default_dicom_zip)
            print(f"Loaded default DICOM series: {default_series.num_slices} slices.")
        except Exception as e:
            print(f"Warning: Failed to load default DICOM series: {e}")

    # Load default reference mask if present
    if settings.default_reference_nii.exists():
        try:
            reference_mask.load(settings.default_reference_nii)
            print(f"Loaded reference lesion mask: shape {reference_mask.shape}, voxels {reference_mask.positive_voxels}.")
        except Exception as e:
            print(f"Warning: Failed to load reference mask: {e}")

    yield

app = FastAPI(title="Nucleo OmniLesion CT Viewer", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Index file not found</h1>", status_code=404)
    return FileResponse(index_file)

# --- Configuration Endpoints ---

class ConfigUpdate(BaseModel):
    base_url: Optional[str] = None
    token: Optional[str] = None

@app.get("/api/config")
async def get_config():
    return {
        "api_base_url": settings.api_base_url,
        "masked_token": settings.masked_token,
        "has_token": bool(settings.api_token)
    }

@app.post("/api/config")
async def update_config(payload: ConfigUpdate):
    settings.update(base_url=payload.base_url, token=payload.token)
    return {
        "status": "updated",
        "api_base_url": settings.api_base_url,
        "masked_token": settings.masked_token
    }

# --- Case & CT Endpoints ---

@app.get("/api/case/info")
async def get_case_info():
    if not default_series.is_loaded:
        raise HTTPException(status_code=503, detail="DICOM series is not loaded.")
    summary = default_series.get_summary()
    summary["has_reference_mask"] = reference_mask.is_loaded
    if reference_mask.is_loaded:
        summary["reference_mask"] = reference_mask.get_summary()
    return summary

@app.get("/api/case/slice/{idx}")
async def get_slice_image(
    idx: int,
    window: float = Query(400.0, description="Window Width"),
    level: float = Query(40.0, description="Window Level")
):
    if not default_series.is_loaded:
        raise HTTPException(status_code=503, detail="DICOM series not loaded.")
    try:
        png_bytes = default_series.render_slice_png(idx, window=window, level=level)
        return Response(content=png_bytes, media_type="image/png")
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Slice {idx} out of range.")

@app.get("/api/case/slice/{idx}/pixel")
async def get_pixel_hu(idx: int, x: int = Query(...), y: int = Query(...)):
    if not default_series.is_loaded:
        raise HTTPException(status_code=503, detail="DICOM series not loaded.")
    try:
        slice_hu = default_series.get_slice_hu(idx)
        # Note: DICOM slice has shape (rows, cols) where y=row, x=col
        if 0 <= y < slice_hu.shape[0] and 0 <= x < slice_hu.shape[1]:
            hu_val = int(slice_hu[y, x])
            return {"x": x, "y": y, "slice": idx, "hu": hu_val}
        return {"x": x, "y": y, "slice": idx, "hu": None}
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Slice {idx} out of range.")

# --- Reference Mask Slice Endpoint ---

@app.get("/api/reference/slice/{idx}")
async def get_reference_mask_slice(
    idx: int,
    r: int = 16,
    g: int = 185,
    b: int = 129,
    a: int = 140
):
    if not reference_mask.is_loaded:
        raise HTTPException(status_code=404, detail="Reference mask is not loaded.")
    try:
        png_bytes = reference_mask.render_slice_png(idx, color_rgba=(r, g, b, a))
        return Response(content=png_bytes, media_type="image/png")
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Slice {idx} out of range.")

# --- Model Mask Slice Endpoint ---

@app.get("/api/jobs/{job_id}/structures/{structure_name}/slice/{idx}")
async def get_job_structure_slice(
    job_id: str,
    structure_name: str,
    idx: int,
    r: Optional[int] = None,
    g: Optional[int] = None,
    b: Optional[int] = None,
    a: Optional[int] = None
):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    mask = job.masks.get(structure_name)
    if not mask or not mask.is_loaded:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_name}' not found for job '{job_id}'.")
    
    rgba = None
    if all(v is not None for v in (r, g, b, a)):
        rgba = (r, g, b, a)

    try:
        png_bytes = mask.render_slice_png(idx, color_rgba=rgba)
        return Response(content=png_bytes, media_type="image/png")
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Slice {idx} out of range.")

# --- Upload & Segmentation Jobs ---

@app.post("/api/segmentations/upload")
async def trigger_segmentation_upload(
    file: Optional[UploadFile] = File(None),
    use_default: bool = Form(False)
):
    """
    Trigger upload of a DICOM series zip to POST /v1/segmentations.
    Generates a unique job ID so concurrent requests never mix results.
    """
    if use_default or file is None:
        if not settings.default_dicom_zip.exists():
            raise HTTPException(status_code=400, detail="Default DICOM zip file not found on disk.")
        with open(settings.default_dicom_zip, "rb") as f:
            zip_bytes = f.read()
        label = "case-001 (default)"
    else:
        zip_bytes = await file.read()
        label = file.filename or "uploaded-archive.zip"

    # Create job
    job = job_manager.create_job(label=label)

    # Initialize client
    client = NucleoApiClient(
        base_url=settings.api_base_url,
        token=settings.api_token,
        timeout_seconds=90.0
    )

    # Execute in background task
    asyncio.create_task(
        job_manager.execute_upload_job(
            job_id=job.id,
            zip_bytes=zip_bytes,
            api_client=client,
            reference_mask=reference_mask
        )
    )

    return {"job_id": job.id, "status": job.status, "message": "Job initiated."}

@app.post("/api/segmentations/demo-cached")
async def create_demo_cached_job():
    """
    Instantly create a completed job using pre-cached model mask for case-001.
    Useful for instant demo or verification without waiting for external upload.
    """
    if not settings.cached_model_nii.exists():
        raise HTTPException(status_code=404, detail="Cached model mask file not found.")

    cached_mask = NiftiMask.from_file(
        settings.cached_model_nii,
        name="lesions",
        color=(245, 158, 11, 140)
    )
    job = job_manager.create_job(label="case-001 (instant demo)")
    
    grid_check = None
    metrics = None
    if reference_mask.is_loaded:
        grid_check = check_grid_compatibility(
            shape1=reference_mask.shape,
            shape2=cached_mask.shape,
            affine1=reference_mask.affine,
            affine2=cached_mask.affine
        )
        if grid_check["is_compatible"]:
            metrics = compute_metrics(
                reference_mask=reference_mask.data,
                model_mask=cached_mask.data,
                spacing_mm=reference_mask.spacing
            )

    job.set_completed(
        case_id="case-001",
        structures=[{"name": "lesions", "url": "/v1/cases/case-001/structures/lesions"}],
        masks={"lesions": cached_mask},
        metrics=metrics,
        grid_check=grid_check
    )
    return job.to_dict()

@app.get("/api/jobs")
async def list_jobs():
    return job_manager.list_jobs()

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.to_dict()

@app.get("/api/jobs/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if not job.metrics:
        raise HTTPException(status_code=400, detail="Metrics not available for this job.")
    return {
        "job_id": job.id,
        "grid_check": job.grid_check,
        "metrics": job.metrics
    }
