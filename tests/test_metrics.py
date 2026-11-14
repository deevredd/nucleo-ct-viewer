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
    # Ref has 100 voxels (0..9, 0..9)
    ref[0:10, 0:10, 0:1] = 1
    # Model has 100 voxels (0..9, 5..14)
    model[0:10, 5:15, 0:1] = 1

    res = compute_metrics(ref, model)
    # intersection = 50, total = 200, Dice = 2 * 50 / 200 = 0.5
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
    # CT: 180 slices, 350 rows, 416 cols -> NIfTI expected shape: (416, 350, 180)
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

def test_compute_metrics_affine_determinant_volume():
    ref = np.ones((10, 10, 10), dtype=np.uint8)  # 1000 voxels
    model = np.ones((10, 10, 10), dtype=np.uint8)
    
    # Affine with diagonal [2.0, 2.0, 2.0, 1.0] -> |det(A)| = 8.0 mm3/voxel = 0.008 cm3/voxel
    # 1000 voxels * 0.008 cm3 = 8.0 cm3
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    res = compute_metrics(ref, model, affine=affine)
    assert res["ref_volume_cm3"] == 8.0
    assert res["model_volume_cm3"] == 8.0
    assert res["intersection_volume_cm3"] == 8.0
