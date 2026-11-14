import pytest
from fastapi.testclient import TestClient
from app.main import app

def test_api_routes_end_to_end():
    with TestClient(app) as client:
        # Case info
        res = client.get("/api/case/info")
        assert res.status_code == 200
        data = res.json()
        assert data["num_slices"] == 180
        assert data["rows"] == 350
        assert data["cols"] == 416
        assert data["has_reference_mask"] is True

        # Slice rendering
        res_slice = client.get("/api/case/slice/90?window=400&level=40")
        assert res_slice.status_code == 200
        assert res_slice.content[:8] == b'\x89PNG\r\n\x1a\n'

        # Reference mask rendering
        res_ref = client.get("/api/reference/slice/90")
        assert res_ref.status_code == 200
        assert res_ref.content[:8] == b'\x89PNG\r\n\x1a\n'

        # Pixel HU query
        res_px = client.get("/api/case/slice/90/pixel?x=200&y=175")
        assert res_px.status_code == 200
        assert "hu" in res_px.json()

        # Jobs listing after demo load
        client.post("/api/segmentations/demo-cached")
        res_jobs = client.get("/api/jobs")
        assert res_jobs.status_code == 200
        jobs = res_jobs.json()
        assert len(jobs) >= 1
        job_id = jobs[0]["id"]
        assert jobs[0]["metrics"]["dice"] == pytest.approx(0.9284, abs=1e-3)

        # Diff / Error map slice rendering
        res_diff = client.get(f"/api/jobs/{job_id}/diff_slice/90")
        assert res_diff.status_code == 200
        assert res_diff.content[:8] == b'\x89PNG\r\n\x1a\n'
