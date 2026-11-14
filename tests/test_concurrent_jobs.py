import asyncio
import pytest
from unittest.mock import patch, MagicMock
from app.job_manager import JobManager
from app.api_client import NucleoApiClient
from app.nifti_loader import NiftiMask

@pytest.mark.asyncio
async def test_concurrent_jobs_do_not_mix_results():
    manager = JobManager()
    
    # Create 3 concurrent jobs with distinct simulated outcomes
    job1 = manager.create_job(label="Job-Case-001")
    job2 = manager.create_job(label="Job-Case-002")
    job3 = manager.create_job(label="Job-Failing")

    # Mock client 1: returns case-001 with lesions
    client1 = NucleoApiClient("https://api.local", "token1")
    async def mock_upload1(zip_bytes, **kwargs):
        await asyncio.sleep(0.05)
        return {
            "success": True,
            "case_id": "case-001",
            "structures": [{"name": "lesions", "url": "/v1/cases/case-001/structures/lesions"}]
        }
    async def mock_download_mask1(url):
        return b"" # dummy mask

    # Mock client 2: returns case-002 with liver and tumors
    client2 = NucleoApiClient("https://api.local", "token2")
    async def mock_upload2(zip_bytes, **kwargs):
        await asyncio.sleep(0.02)
        return {
            "success": True,
            "case_id": "case-002",
            "structures": [
                {"name": "liver", "url": "/v1/cases/case-002/structures/liver"},
                {"name": "tumors", "url": "/v1/cases/case-002/structures/tumors"}
            ]
        }

    # Mock client 3: returns 400 corrupt_zip error
    client3 = NucleoApiClient("https://api.local", "token3")
    async def mock_upload3(zip_bytes, **kwargs):
        await asyncio.sleep(0.01)
        return {
            "success": False,
            "status_code": 400,
            "code": "bad_request",
            "reason": "corrupt_zip",
            "user_message": "The ZIP archive is damaged or corrupted and cannot be opened."
        }

    with patch.object(client1, "upload_segmentation", side_effect=mock_upload1), \
         patch.object(client1, "download_structure_mask", side_effect=mock_download_mask1), \
         patch.object(client2, "upload_segmentation", side_effect=mock_upload2), \
         patch.object(client2, "download_structure_mask", side_effect=mock_download_mask1), \
         patch.object(client3, "upload_segmentation", side_effect=mock_upload3), \
         patch.object(NiftiMask, "from_bytes") as mock_nifti:

        dummy_mask = MagicMock()
        dummy_mask.positive_voxels = 100
        dummy_mask.shape = (416, 350, 180)
        dummy_mask.is_loaded = True
        mock_nifti.return_value = dummy_mask

        # Run all three jobs concurrently
        await asyncio.gather(
            manager.execute_upload_job(job1.id, b"zip1", client1),
            manager.execute_upload_job(job2.id, b"zip2", client2),
            manager.execute_upload_job(job3.id, b"zip3", client3)
        )

    # Verification: Check that results were completely isolated!
    res1 = manager.get_job(job1.id).to_dict()
    res2 = manager.get_job(job2.id).to_dict()
    res3 = manager.get_job(job3.id).to_dict()

    # Job 1 verification
    assert res1["status"] == "completed"
    assert res1["case_id"] == "case-001"
    assert len(res1["structures"]) == 1
    assert res1["structures"][0]["name"] == "lesions"
    assert res1["error"] is None

    # Job 2 verification
    assert res2["status"] == "completed"
    assert res2["case_id"] == "case-002"
    assert len(res2["structures"]) == 2
    structure_names_job2 = [s["name"] for s in res2["structures"]]
    assert "liver" in structure_names_job2
    assert "tumors" in structure_names_job2
    assert "lesions" not in structure_names_job2
    assert res2["error"] is None

    # Job 3 verification (failed)
    assert res3["status"] == "failed"
    assert res3["case_id"] is None
    assert len(res3["structures"]) == 0
    assert res3["error"]["code"] == "bad_request"
    assert res3["error"]["reason"] == "corrupt_zip"
    assert "damaged or corrupted" in res3["error"]["user_message"]
