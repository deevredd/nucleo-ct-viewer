import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List
from app.api_client import NucleoApiClient
from app.nifti_loader import NiftiMask
from app.metrics import check_grid_compatibility, compute_metrics

class Job:
    def __init__(self, job_id: str, label: str = "upload"):
        self.id: str = job_id
        self.label: str = label
        self.created_at: float = time.time()
        self.updated_at: float = self.created_at
        self.status: str = "queued"  # queued, uploading, processing, downloading, completed, failed
        self.progress_message: str = "Job queued"
        self.retry_count: int = 0
        self.retry_wait_s: float = 0.0
        self.case_id: Optional[str] = None
        self.structures: List[Dict[str, Any]] = []
        self.masks: Dict[str, NiftiMask] = {}
        self.metrics: Optional[Dict[str, Any]] = None
        self.grid_check: Optional[Dict[str, Any]] = None
        self.error: Optional[Dict[str, Any]] = None

    def update_status(self, status: str, message: str, retry_count: int = 0, retry_wait: float = 0.0):
        self.status = status
        self.progress_message = message
        self.retry_count = retry_count
        self.retry_wait_s = retry_wait
        self.updated_at = time.time()

    def set_failed(self, error_dict: Dict[str, Any]):
        self.status = "failed"
        self.error = error_dict
        self.progress_message = error_dict.get("user_message", "Segmentation failed.")
        self.updated_at = time.time()

    def set_completed(self, case_id: str, structures: List[Dict[str, Any]], masks: Dict[str, NiftiMask], metrics: Optional[Dict[str, Any]], grid_check: Optional[Dict[str, Any]]):
        self.status = "completed"
        self.case_id = case_id
        self.structures = structures
        self.masks = masks
        self.metrics = metrics
        self.grid_check = grid_check
        self.progress_message = "Segmentation completed successfully."
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "progress_message": self.progress_message,
            "retry_count": self.retry_count,
            "retry_wait_s": round(self.retry_wait_s, 1),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "case_id": self.case_id,
            "structures": [
                {
                    "name": s.get("name"),
                    "url": s.get("url"),
                    "positive_voxels": self.masks[s.get("name")].positive_voxels if s.get("name") in self.masks else None,
                    "shape": self.masks[s.get("name")].shape if s.get("name") in self.masks else None,
                }
                for s in self.structures
            ],
            "grid_check": self.grid_check,
            "metrics": {
                "dice": self.metrics.get("dice"),
                "dice_percentage": self.metrics.get("dice_percentage"),
                "jaccard": self.metrics.get("jaccard"),
                "precision": self.metrics.get("precision"),
                "recall": self.metrics.get("recall"),
                "ref_volume_cm3": self.metrics.get("ref_volume_cm3"),
                "model_volume_cm3": self.metrics.get("model_volume_cm3"),
                "intersection_volume_cm3": self.metrics.get("intersection_volume_cm3")
            } if self.metrics else None,
            "error": self.error
        }

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    def create_job(self, label: str = "case-001") -> Job:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        job = Job(job_id=job_id, label=label)
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        # Return sorted by created_at descending
        sorted_jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in sorted_jobs]

    def clear_jobs(self):
        self.jobs.clear()

    async def execute_upload_job(
        self,
        job_id: str,
        zip_bytes: bytes,
        api_client: NucleoApiClient,
        reference_mask: Optional[NiftiMask] = None
    ):
        job = self.get_job(job_id)
        if not job:
            return

        def on_progress(state: str, attempt: int, backoff: float):
            if state == "uploading":
                job.update_status("uploading", "Uploading series to Nucleo OmniLesion API...", attempt)
            elif state == "retrying":
                job.update_status("retrying", f"Rate limit / capacity reached. Retrying in {backoff:.1f}s (Attempt {attempt})...", attempt, backoff)

        # 1. Upload DICOM series
        job.update_status("uploading", "Sending DICOM series to segmentation service...")
        result = await api_client.upload_segmentation(
            zip_bytes=zip_bytes,
            progress_callback=on_progress
        )

        if not result.get("success"):
            job.set_failed(result)
            return

        case_id = result.get("case_id", "unknown")
        structures_list = result.get("structures", [])
        job.update_status("downloading", f"Processing {len(structures_list)} structure mask(s)...")

        # 2. Download and load each returned structure mask
        loaded_masks: Dict[str, NiftiMask] = {}
        # Predefined distinct colors for structures
        palette = [
            (245, 158, 11, 150),   # Amber/Orange
            (59, 130, 246, 150),   # Blue
            (236, 72, 153, 150),   # Pink
            (168, 85, 247, 150),   # Purple
            (20, 184, 166, 150),   # Teal
        ]

        for idx, struct in enumerate(structures_list):
            s_name = struct.get("name")
            s_url = struct.get("url")
            color = palette[idx % len(palette)]
            
            try:
                mask_bytes = await api_client.download_structure_mask(s_url)
                mask = NiftiMask.from_bytes(mask_bytes, name=s_name, color=color)
                loaded_masks[s_name] = mask
            except Exception as e:
                job.set_failed({
                    "code": "mask_download_failed",
                    "reason": "download_error",
                    "user_message": f"Failed to download structure mask '{s_name}': {str(e)}",
                    "raw_message": str(e)
                })
                return

        # 3. Check grid compatibility and compute Dice with reference mask
        grid_check = None
        metrics = None

        if reference_mask and reference_mask.is_loaded:
            # We look for a lesion mask in the returned structures
            model_lesion_mask = loaded_masks.get("lesions") or (
                list(loaded_masks.values())[0] if loaded_masks else None
            )

            if model_lesion_mask:
                grid_check = check_grid_compatibility(
                    shape1=reference_mask.shape,
                    shape2=model_lesion_mask.shape,
                    affine1=reference_mask.affine,
                    affine2=model_lesion_mask.affine
                )

                if grid_check["is_compatible"]:
                    metrics = compute_metrics(
                        reference_mask=reference_mask.data,
                        model_mask=model_lesion_mask.data,
                        spacing_mm=reference_mask.spacing
                    )
                else:
                    grid_check["warning"] = "Masks are on different spatial grids; Dice coefficient could not be computed directly."

        # 4. Mark completed
        job.set_completed(
            case_id=case_id,
            structures=structures_list,
            masks=loaded_masks,
            metrics=metrics,
            grid_check=grid_check
        )

job_manager = JobManager()
