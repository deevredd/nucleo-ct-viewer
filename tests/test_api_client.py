import pytest
import httpx
from unittest.mock import patch, MagicMock
from app.api_client import NucleoApiClient, translate_error

@pytest.fixture
def client():
    return NucleoApiClient(
        base_url="https://test.nucleo.local",
        token="test_token_123"
    )

def test_translate_error_known_reasons():
    assert "damaged or corrupted" in translate_error("bad_request", "corrupt_zip")
    assert "non-DICOM" in translate_error("bad_request", "not_dicom")
    assert "does not match any recognized study" in translate_error("not_found", "unknown_series")
    assert "invalid" in translate_error("unauthorized", "invalid").lower()
    assert "multiple DICOM series" in translate_error("unprocessable", "multiple_series")
    assert "maximum upload limit" in translate_error("too_large", None)

def test_translate_error_fallback():
    msg = translate_error("unknown_code", "some_reason", fallback_msg="Custom fallback")
    assert "Custom fallback" in msg

@pytest.mark.asyncio
async def test_upload_success(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "case_id": "case-001",
        "structures": [{"name": "lesions", "url": "/v1/cases/case-001/structures/lesions"}]
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        res = await client.upload_segmentation(b"dummy_zip_bytes")
        assert res["success"] is True
        assert res["case_id"] == "case-001"
        assert len(res["structures"]) == 1

@pytest.mark.asyncio
async def test_upload_failed_bad_request(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {
        "error": {
            "code": "bad_request",
            "message": "Archive contains no DICOM files",
            "details": {"reason": "empty_archive"}
        }
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        res = await client.upload_segmentation(b"bad_zip_bytes", max_retries=0)
        assert res["success"] is False
        assert res["code"] == "bad_request"
        assert res["reason"] == "empty_archive"
        assert "does not contain any DICOM slices" in res["user_message"]

@pytest.mark.asyncio
async def test_upload_honours_retry_after(client):
    # First response: 503 with Retry-After: 0.1
    # Second response: 200 success
    resp_503 = MagicMock()
    resp_503.status_code = 503
    resp_503.headers = {"Retry-After": "0.05"}
    resp_503.json.return_value = {
        "error": {
            "code": "unavailable",
            "message": "Server saturated",
            "details": {"reason": "uploads_in_flight", "retry_after_s": 0.05}
        }
    }

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "case_id": "case-001",
        "structures": [{"name": "lesions", "url": "/v1/cases/case-001/structures/lesions"}]
    }

    with patch("httpx.AsyncClient.post", side_effect=[resp_503, resp_200]):
        callback_log = []
        def on_progress(state, attempt, backoff):
            callback_log.append((state, attempt))

        res = await client.upload_segmentation(
            b"zip_bytes",
            max_retries=2,
            progress_callback=on_progress
        )
        assert res["success"] is True
        assert res["case_id"] == "case-001"
        assert any(item[0] == "retrying" for item in callback_log)

@pytest.mark.asyncio
async def test_upload_multiple_series_error(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.json.return_value = {
        "error": {
            "code": "unprocessable",
            "message": "Multiple series in zip",
            "details": {"reason": "multiple_series"}
        }
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        res = await client.upload_segmentation(b"multi_series_bytes", max_retries=0)
        assert res["success"] is False
        assert res["code"] == "unprocessable"
        assert res["reason"] == "multiple_series"
        assert "multiple DICOM series" in res["user_message"]

@pytest.mark.asyncio
async def test_upload_unauthorized_expired_token(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {
        "error": {
            "code": "unauthorized",
            "message": "Token has expired",
            "details": {"reason": "expired"}
        }
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        res = await client.upload_segmentation(b"zip_bytes", max_retries=0)
        assert res["success"] is False
        assert res["code"] == "unauthorized"
        assert res["reason"] == "expired"
        assert "token has expired" in res["user_message"].lower()
