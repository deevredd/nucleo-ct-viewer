import numpy as np
import pytest
from app.metrics import check_grid_compatibility, check_ct_to_mask_compatibility, compute_metrics

def test_check_grid_compatibility_exact_match():
    shape = (416, 350, 180)
    affine = np.diag([-0.7, -0.7, 1.0, 1.0])
    
    result = check_grid_compatibility(
        shape1=shape,
        shape2=shape,
        affine1=affine,
        affine2=affine.copy()
    )
    assert result["is_compatible"] is True
    assert result["shape_match"] is True
    assert result["affine_match"] is True
    assert result["affine_diff_max"] == 0.0

def test_check_grid_compatibility_mismatched_shape():
    shape1 = (416, 350, 180)
    shape2 = (512, 512, 100)
    
    result = check_grid_compatibility(shape1=shape1, shape2=shape2)
    assert result["is_compatible"] is False
    assert result["shape_match"] is False

def test_check_grid_compatibility_mismatched_affine():
    shape = (416, 350, 180)
    affine1 = np.diag([1.0, 1.0, 1.0, 1.0])
    affine2 = np.diag([2.0, 2.0, 2.0, 1.0])
    
    result = check_grid_compatibility(shape1=shape, shape2=shape, affine1=affine1, affine2=affine2)
    assert result["is_compatible"] is False
    assert result["shape_match"] is True
    assert result["affine_match"] is False

def test_compute_dice_perfect_overlap():
    ref = np.zeros((10, 10, 10), dtype=np.uint8)
    model = np.zeros((10, 10, 10), dtype=np.uint8)
    ref[2:6, 2:6, 2:6] = 1
    model[2:6, 2:6, 2:6] = 1

    res = compute_metrics(ref, model)
    assert res["dice"] == 1.0
    assert res["dice_percentage"] == 100.0
    assert res["jaccard"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0

def test_compute_dice_zero_overlap():
    ref = np.zeros((10, 10, 10), dtype=np.uint8)
    model = np.zeros((10, 10, 10), dtype=np.uint8)
    ref[0:2, 0:2, 0:2] = 1
    model[5:8, 5:8, 5:8] = 1

    res = compute_metrics(ref, model)
    assert res["dice"] == 0.0
    assert res["intersection_voxels"] == 0
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0

def test_compute_dice_partial_overlap():
    ref = np.zeros((20, 20, 10), dtype=np.uint8)
    model = np.zeros((20, 20, 10), dtype=np.uint8)
    ref[0:10, 0:10, 0:1] = 1
    model[0:10, 5:15, 0:1] = 1

    res = compute_metrics(ref, model)
    assert res["dice"] == 0.5
    assert res["intersection_voxels"] == 50
    assert res["ref_voxels"] == 100
    assert res["model_voxels"] == 100
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5

def test_compute_dice_both_empty():
    ref = np.zeros((5, 5, 5), dtype=np.uint8)
    model = np.zeros((5, 5, 5), dtype=np.uint8)

    res = compute_metrics(ref, model)
    assert res["dice"] == 1.0
    assert res["ref_voxels"] == 0
    assert res["model_voxels"] == 0

def test_check_ct_to_mask_compatibility():
    ct_affine = np.diag([-0.70898438, -0.70898438, 1.0, 1.0])
    mask_shape = (416, 350, 180)
    mask_affine = ct_affine.copy()

    res = check_ct_to_mask_compatibility(
        ct_num_slices=180,
        ct_rows=350,
        ct_cols=416,
        ct_affine_ras=ct_affine,
        mask_shape=mask_shape,
        mask_affine=mask_affine
    )
    assert res["is_compatible"] is True
    assert res["shape_match"] is True
    assert res["affine_match"] is True

def test_check_ct_to_mask_compatibility_rejects_gantry_tilt():
    ct_affine = np.diag([-0.70898438, -0.70898438, 1.0, 1.0])
    mask_shape = (416, 350, 180)
    res = check_ct_to_mask_compatibility(
        ct_num_slices=180,
        ct_rows=350,
        ct_cols=416,
        ct_affine_ras=ct_affine,
        mask_shape=mask_shape,
        mask_affine=ct_affine,
        has_gantry_tilt=True,
        gantry_tilt_angle_deg=12.5
    )
    assert res["is_compatible"] is False
    assert "Gantry tilt" in res["message"]

def test_check_ct_to_mask_compatibility_rejects_nonuniform_spacing():
    ct_affine = np.diag([-0.70898438, -0.70898438, 1.0, 1.0])
    mask_shape = (416, 350, 180)
    res = check_ct_to_mask_compatibility(
        ct_num_slices=180,
        ct_rows=350,
        ct_cols=416,
        ct_affine_ras=ct_affine,
        mask_shape=mask_shape,
        mask_affine=ct_affine,
        has_uniform_spacing=False
    )
    assert res["is_compatible"] is False
    assert "Non-uniform slice spacing" in res["message"]

def test_compute_metrics_affine_determinant_volume():
    ref = np.ones((10, 10, 10), dtype=np.uint8)
    model = np.ones((10, 10, 10), dtype=np.uint8)
    
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    res = compute_metrics(ref, model, affine=affine)
    assert res["ref_volume_cm3"] == 8.0
    assert res["model_volume_cm3"] == 8.0
    assert res["intersection_volume_cm3"] == 8.0

def test_check_ct_to_mask_compatibility_rejects_corner_displacement():
    ct_affine = np.diag([-0.70898438, -0.70898438, 1.0, 1.0])
    mask_shape = (416, 350, 180)
    
    shifted_affine = ct_affine.copy()
    shifted_affine[0, 3] += 1.0

    res = check_ct_to_mask_compatibility(
        ct_num_slices=180,
        ct_rows=350,
        ct_cols=416,
        ct_affine_ras=ct_affine,
        mask_shape=mask_shape,
        mask_affine=shifted_affine,
        tolerance=2.0,
        max_displacement_mm=0.5
    )
    assert res["is_compatible"] is False
    assert res["affine_match"] is False
    assert "physical displacement" in res["message"].lower()
    assert res["corner_disp_max_mm"] >= 1.0
