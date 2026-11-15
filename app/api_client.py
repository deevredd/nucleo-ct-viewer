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
    ("internal", None): "Internal server error occurred on the segmentation backend. Please retry shortly.",
    ("network_error", None): "Unable to connect to the OmniLesion API server. Please check your network connection or verify the endpoint URL in Settings."
}

def translate_error(code: str, reason: Optional[str] = None, fallback_msg: str = "") -> str:
    if (code, reason) in ERROR_TRANSLATIONS:
        return ERROR_TRANSLATIONS[(code, reason)]
    
    if (code, None) in ERROR_TRANSLATIONS:
        return ERROR_TRANSLATIONS[(code, None)]
    
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

def normalize_structures(raw_structures: Any) -> list:
    normalized = []
    if isinstance(raw_structures, dict):
        for name, url in raw_structures.items():
            if isinstance(url, str):
                normalized.append({"name": str(name), "url": url})
            elif isinstance(url, dict) and "url" in url:
                normalized.append({"name": str(name), "url": str(url["url"])})
    elif isinstance(raw_structures, list):
        for item in raw_structures:
            if isinstance(item, dict) and "name" in item and "url" in item:
                normalized.append({"name": str(item["name"]), "url": str(item["url"])})
            elif isinstance(item, str):
                normalized.append({"name": item, "url": f"/v1/structures/{item}"})
    return normalized

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
        url = f"{self.base_url}/v1/segmentations"
        headers = self._get_headers(content_type="application/zip")

        timeout_cfg = httpx.Timeout(connect=15.0, read=self.timeout, write=self.timeout, pool=15.0)

        attempt = 0
        while attempt <= max_retries:
            try:
                if progress_callback:
                    progress_callback("uploading", attempt, 0.0)

                async with httpx.AsyncClient(timeout=timeout_cfg) as client:
                    resp = await client.post(url, content=zip_bytes, headers=headers)

                if resp.status_code == 200:
                    data = resp.json()
                    raw_structs = data.get("structures", [])
                    return {
                        "success": True,
                        "case_id": data.get("case_id"),
                        "structures": normalize_structures(raw_structs),
                        "status_code": 200
                    }

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

                if resp.status_code in (429, 503) and attempt < max_retries:
                    retry_header = resp.headers.get("Retry-After")
                    retry_s = 2.0
                    if retry_header:
                        try:
                            retry_s = float(retry_header)
                        except ValueError:
                            pass
                    elif isinstance(details, dict) and "retry_after_s" in details:
                        retry_s = float(details["retry_after_s"])

                    jitter = random.uniform(0.1, 0.6)
                    backoff = retry_s * (1.5 ** attempt) + jitter
                    attempt += 1

                    if progress_callback:
                        progress_callback("retrying", attempt, backoff)

                    await asyncio.sleep(backoff)
                    continue

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
                return {
                    "success": False,
                    "status_code": 408,
                    "code": "timeout",
                    "reason": "upload_timeout",
                    "user_message": "Upload request timed out while waiting for server response. The server may still be processing your scan. Please check jobs or retry manually.",
                    "raw_message": "POST upload request timed out",
                    "details": {}
                }
            except (httpx.ConnectError, httpx.NetworkError):
                return {
                    "success": False,
                    "status_code": 503,
                    "code": "network_error",
                    "reason": "connection_failed",
                    "user_message": translate_error("network_error", None),
                    "raw_message": "Failed to connect to the OmniLesion API server (Network / DNS unreachable)",
                    "details": {}
                }
            except Exception as e:
                err_text = str(e).strip()
                return {
                    "success": False,
                    "status_code": 500,
                    "code": "network_error",
                    "reason": "client_exception",
                    "user_message": translate_error("network_error", None),
                    "raw_message": err_text or type(e).__name__,
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
        if structure_url.startswith("http"):
            url = structure_url
        else:
            url = f"{self.base_url}{structure_url}"

        headers = self._get_headers()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.content
