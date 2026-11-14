import asyncio
import json
import random
from typing import Dict, Any, Optional, Tuple, Callable
import httpx

ERROR_TRANSLATIONS = {
    ("bad_request", "empty_body"): "The uploaded file is empty. Please select a valid DICOM ZIP archive.",
    ("bad_request", "not_zip"): "The file provided is not a valid ZIP archive.",
    ("bad_request", "empty_archive"): "The ZIP archive does not contain any DICOM slices.",
    ("bad_request", "corrupt_zip"): "The ZIP archive is damaged or corrupted and cannot be opened.",
    ("bad_request", "not_dicom"): "The archive contains non-DICOM files or unrecognized files.",
    ("bad_request", "corrupt_dicom"): "One or more DICOM slices inside the archive are corrupt or unreadable.",
    ("bad_request", "truncated_pixel_data"): "The DICOM slice pixel data is incomplete or truncated.",
    ("bad_request", "too_many_entries"): "The ZIP archive exceeds the maximum limit of 5,000 files.",
    ("bad_request", "uncompressed_too_large"): "The uncompressed archive exceeds the maximum limit of 1 GiB.",
    ("bad_request", "member_too_large"): "A single file in the archive exceeds the limit of 64 MiB.",
    
    ("unauthorized", "missing"): "API token is missing. Please set your authorization token in settings.",
    ("unauthorized", "invalid"): "The provided API token is invalid. Please verify your token.",
    ("unauthorized", "expired"): "Your API token has expired. Please obtain a renewed token from Nucleo.",
    ("unauthorized", "revoked"): "Your API token has been revoked.",
    
    ("not_found", "unknown_series"): "The uploaded DICOM scan does not match any recognized study in the system.",
    ("not_found", None): "The requested case, route, or structure was not found on the server.",
    
    ("method_not_allowed", None): "Unsupported HTTP method on this API endpoint.",
    ("timeout", None): "Upload or processing timed out. The server did not receive the complete scan in time.",
    ("too_large", None): "The uploaded archive exceeds the maximum upload limit (150 MiB).",
    ("unsupported_media_type", None): "Invalid upload format. The server requires raw ZIP bytes (application/zip).",
    
    ("unprocessable", "missing_tags"): "Required DICOM metadata tags (e.g. ImagePosition, UIDs, spacing) are missing.",
    ("unprocessable", "multiple_series"): "The archive contains multiple DICOM series. Exactly one series must be uploaded.",
    ("unprocessable", "inconsistent_geometry"): "Inconsistent geometry detected across DICOM slices (mismatched orientation or spacing).",
    ("unprocessable", "incomplete_series"): "Incomplete series: slices are missing or there are gaps along the axial axis.",
    ("unprocessable", "inconsistent_spacing"): "Slice spacing is non-uniform across the series.",
    ("unprocessable", "duplicate_position"): "Duplicate slice positions detected in the uploaded series.",
    ("unprocessable", "duplicate_instance"): "Duplicate SOP Instance UIDs detected in the uploaded series.",
    ("unprocessable", "unsupported_transfer_syntax"): "Unsupported DICOM pixel compression or transfer syntax.",
    ("unprocessable", "bad_geometry"): "Cannot reconstruct a 3D medical volume from the provided slice geometry.",
    
    ("rate_limited", None): "API rate limit reached. The system is automatically retrying after backing off.",
    ("unavailable", None): "The segmentation model service is currently saturated. Automatically backing off and retrying.",
    ("internal", None): "Internal server error occurred on the segmentation backend. Please retry shortly."
}

def translate_error(code: str, reason: Optional[str] = None, fallback_msg: str = "") -> str:
    """
    Branch strictly on error.code and details.reason as specified in requirements.
    """
    # 1. Exact match on (code, reason)
    if (code, reason) in ERROR_TRANSLATIONS:
        return ERROR_TRANSLATIONS[(code, reason)]
    
    # 2. Match on (code, None)
    if (code, None) in ERROR_TRANSLATIONS:
        return ERROR_TRANSLATIONS[(code, None)]
    
    # 3. Code-based fallback
    code_defaults = {
        "bad_request": "The uploaded data is invalid or could not be parsed.",
        "unauthorized": "Authentication failed. Please check your API token.",
        "not_found": "Resource not found on server.",
        "timeout": "The request timed out.",
        "too_large": "Upload payload is too large.",
        "unsupported_media_type": "Unsupported media format.",
        "unprocessable": "The DICOM data is readable but rejected due to inconsistent series geometry.",
        "rate_limited": "Too many requests. Please wait before retrying.",
        "unavailable": "The service is temporarily unavailable.",
        "internal": "An unexpected server error occurred."
    }
    return code_defaults.get(code, fallback_msg or f"Error ({code}): {reason or 'unknown'}")

class NucleoApiClient:
    def __init__(self, base_url: str, token: str, timeout_seconds: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.token = token.strip()
        self.timeout = timeout_seconds

    def _get_headers(self, content_type: Optional[str] = None) -> Dict[str, str]:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    async def check_health(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/v1/health")
            resp.raise_for_status()
            return resp.json()

    async def upload_segmentation(
        self,
        zip_bytes: bytes,
        max_retries: int = 3,
        progress_callback: Optional[Callable[[str, int, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Upload DICOM series zip to POST /v1/segmentations.
        Honours Retry-After on 429 and 503 with exponential backoff and jitter.
        Returns the parsed JSON response with case_id and structures.
        """
        url = f"{self.base_url}/v1/segmentations"
        headers = self._get_headers(content_type="application/zip")

        # Use an extended timeout for the upload and simulated inference
        timeout_cfg = httpx.Timeout(connect=15.0, read=self.timeout, write=self.timeout, pool=15.0)

        attempt = 0
        while attempt <= max_retries:
            try:
                if progress_callback:
                    progress_callback("uploading", attempt, 0.0)

                async with httpx.AsyncClient(timeout=timeout_cfg) as client:
                    resp = await client.post(url, content=zip_bytes, headers=headers)

                # Success
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "success": True,
                        "case_id": data.get("case_id"),
                        "structures": data.get("structures", []),
                        "status_code": 200
                    }

                # Parse error response body
                error_body = {}
                try:
                    error_body = resp.json().get("error", {})
                except Exception:
                    pass

                code = error_body.get("code") or (
                    "rate_limited" if resp.status_code == 429 else
                    "unavailable" if resp.status_code == 503 else
                    "bad_request" if resp.status_code == 400 else
                    "unauthorized" if resp.status_code == 401 else
                    "not_found" if resp.status_code == 404 else
                    "too_large" if resp.status_code == 413 else
                    "unsupported_media_type" if resp.status_code == 415 else
                    "unprocessable" if resp.status_code == 422 else
                    "internal"
                )
                details = error_body.get("details", {})
                reason = details.get("reason") if isinstance(details, dict) else None

                # Check if retryable (429 or 503)
                if resp.status_code in (429, 503) and attempt < max_retries:
                    # Parse Retry-After header or details
                    retry_header = resp.headers.get("Retry-After")
                    retry_s = 2.0
                    if retry_header:
                        try:
                            retry_s = float(retry_header)
                        except ValueError:
                            pass
                    elif isinstance(details, dict) and "retry_after_s" in details:
                        retry_s = float(details["retry_after_s"])

                    # Exponential backoff + jitter for repeated 503s
                    jitter = random.uniform(0.1, 0.6)
                    backoff = retry_s * (1.5 ** attempt) + jitter
                    attempt += 1

                    if progress_callback:
                        progress_callback("retrying", attempt, backoff)

                    await asyncio.sleep(backoff)
                    continue

                # Non-retryable error or retries exhausted
                user_msg = translate_error(code, reason, error_body.get("message", ""))
                return {
                    "success": False,
                    "status_code": resp.status_code,
                    "code": code,
                    "reason": reason,
                    "user_message": user_msg,
                    "raw_message": error_body.get("message", resp.text),
                    "details": details
                }

            except httpx.TimeoutException:
                if attempt < max_retries:
                    attempt += 1
                    backoff = 2.0 * (1.5 ** attempt) + random.uniform(0.1, 0.5)
                    if progress_callback:
                        progress_callback("retrying", attempt, backoff)
                    await asyncio.sleep(backoff)
                    continue
                return {
                    "success": False,
                    "status_code": 408,
                    "code": "timeout",
                    "reason": "request_timeout",
                    "user_message": translate_error("timeout", None),
                    "raw_message": "Network request timed out",
                    "details": {}
                }
            except Exception as e:
                return {
                    "success": False,
                    "status_code": 500,
                    "code": "network_error",
                    "reason": "client_exception",
                    "user_message": f"Network or connection error: {str(e)}",
                    "raw_message": str(e),
                    "details": {}
                }

        return {
            "success": False,
            "status_code": 503,
            "code": "unavailable",
            "reason": "retries_exhausted",
            "user_message": "Server remained busy after multiple retries. Please try again in a few moments.",
            "raw_message": "Max retries exceeded",
            "details": {}
        }

    async def download_structure_mask(self, structure_url: str) -> bytes:
        """
        Download a .nii.gz segmentation mask from the given URL.
        Relative URLs are resolved against self.base_url.
        """
        if structure_url.startswith("http"):
            url = structure_url
        else:
            url = f"{self.base_url}{structure_url}"

        headers = self._get_headers()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.content
