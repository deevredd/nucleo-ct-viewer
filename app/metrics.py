from typing import Dict, Any, Optional, Tuple, List
import numpy as np

def check_grid_compatibility(
    shape1: Tuple[int, ...],
    shape2: Tuple[int, ...],
    affine1: Optional[np.ndarray] = None,
    affine2: Optional[np.ndarray] = None,
    tolerance: float = 1e-3
) -> Dict[str, Any]:
    """
    Check if two 3D masks are defined on the same spatial grid.
    Compares array dimensions and affine transformation matrices.
    """
    shape_match = tuple(shape1) == tuple(shape2)
    
    affine_match = True
    affine_diff_max = 0.0
    if affine1 is not None and affine2 is not None:
        a1 = np.asarray(affine1, dtype=float)
        a2 = np.asarray(affine2, dtype=float)
        if a1.shape == (4, 4) and a2.shape == (4, 4):
            affine_diff_max = float(np.max(np.abs(a1 - a2)))
            affine_match = np.allclose(a1, a2, atol=tolerance)
        else:
            affine_match = False
            
    is_compatible = shape_match and affine_match
    
    details = []
    if not shape_match:
        details.append(f"Shape mismatch: {shape1} vs {shape2}")
    if not affine_match:
        details.append(f"Affine mismatch (max diff: {affine_diff_max:.5f})")
    if is_compatible:
        details.append(f"Perfect grid alignment: shape {shape1}")

    return {
        "is_compatible": is_compatible,
        "shape_match": shape_match,
        "affine_match": affine_match,
        "shape1": list(shape1),
        "shape2": list(shape2),
        "affine_diff_max": affine_diff_max,
        "message": "; ".join(details)
    }

def check_ct_to_mask_compatibility(
    ct_num_slices: int,
    ct_rows: int,
    ct_cols: int,
    ct_affine_ras: Optional[np.ndarray],
    mask_shape: Tuple[int, ...],
    mask_affine: Optional[np.ndarray],
    tolerance: float = 1e-3
) -> Dict[str, Any]:
    """
    Validates physical alignment between reconstructed 3D DICOM CT volume
    and 3D NIfTI segmentation mask.
    In NIfTI, shape is (cols, rows, slices).
    """
    expected_nifti_shape = (ct_cols, ct_rows, ct_num_slices)
    shape_match = (tuple(mask_shape) == expected_nifti_shape)
    
    affine_match = True
    affine_diff_max = 0.0
    if ct_affine_ras is not None and mask_affine is not None:
        c_aff = np.asarray(ct_affine_ras, dtype=float)
        m_aff = np.asarray(mask_affine, dtype=float)
        if c_aff.shape == (4, 4) and m_aff.shape == (4, 4):
            affine_diff_max = float(np.max(np.abs(c_aff - m_aff)))
            affine_match = np.allclose(c_aff, m_aff, atol=tolerance)
        else:
            affine_match = False
            
    is_compatible = shape_match and affine_match
    return {
        "is_compatible": is_compatible,
        "shape_match": shape_match,
        "affine_match": affine_match,
        "ct_expected_shape": list(expected_nifti_shape),
        "mask_shape": list(mask_shape),
        "affine_diff_max": affine_diff_max,
        "message": "CT and mask physically aligned" if is_compatible else f"CT-to-mask mismatch (shape_match={shape_match}, affine_diff={affine_diff_max:.5f})"
    }

def compute_metrics(
    reference_mask: np.ndarray,
    model_mask: np.ndarray,
    affine: Optional[np.ndarray] = None,
    spacing_mm: Optional[Tuple[float, float, float]] = (0.708984375, 0.708984375, 1.0)
) -> Dict[str, Any]:
    """
    Compute Dice Similarity Coefficient (DSC) and volumetric metrics
    between reference (ground-truth) mask and model prediction mask.
    Voxel volume is computed from |det(A_3x3)| to support general coordinate frames.
    """
    ref_bool = (reference_mask > 0)
    model_bool = (model_mask > 0)

    intersection = int(np.sum(ref_bool & model_bool))
    ref_voxels = int(np.sum(ref_bool))
    model_voxels = int(np.sum(model_bool))
    union = int(np.sum(ref_bool | model_bool))

    fp = int(np.sum((~ref_bool) & model_bool))
    fn = int(np.sum(ref_bool & (~model_bool)))

    total = ref_voxels + model_voxels
    if total == 0:
        dice = 1.0
        jaccard = 1.0
        precision = 1.0
        recall = 1.0
    else:
        dice = (2.0 * intersection) / float(total)
        jaccard = float(intersection) / float(union) if union > 0 else 1.0
        precision = float(intersection) / float(model_voxels) if model_voxels > 0 else 0.0
        recall = float(intersection) / float(ref_voxels) if ref_voxels > 0 else 0.0

    # Voxel volume computation using the affine determinant |det(A_3x3)|
    # This general formula correctly computes physical voxel volume for arbitrary orientations/shears
    if affine is not None and np.asarray(affine).shape >= (3, 3):
        a = np.asarray(affine, dtype=float)
        voxel_vol_cm3 = abs(float(np.linalg.det(a[:3, :3]))) / 1000.0
    elif spacing_mm and len(spacing_mm) == 3:
        voxel_vol_cm3 = float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0
    else:
        voxel_vol_cm3 = 1.0 / 1000.0

    ref_vol_cm3 = round(ref_voxels * voxel_vol_cm3, 2)
    model_vol_cm3 = round(model_voxels * voxel_vol_cm3, 2)
    inter_vol_cm3 = round(intersection * voxel_vol_cm3, 2)

    # Per-slice metrics along the slice axis (axis 2 for (416, 350, 180) or last axis)
    num_slices = reference_mask.shape[-1]
    slice_metrics: List[Dict[str, Any]] = []
    
    for s in range(num_slices):
        r_s = ref_bool[:, :, s] if ref_bool.ndim == 3 else ref_bool[s]
        m_s = model_bool[:, :, s] if model_bool.ndim == 3 else model_bool[s]

        r_sum = int(np.sum(r_s))
        m_sum = int(np.sum(m_s))
        inter_s = int(np.sum(r_s & m_s))
        
        tot_s = r_sum + m_sum
        if tot_s == 0:
            s_dice = 1.0
        else:
            s_dice = (2.0 * inter_s) / float(tot_s)
            
        slice_metrics.append({
            "slice": s,
            "has_lesion": (r_sum > 0) or (m_sum > 0),
            "ref_voxels": r_sum,
            "model_voxels": m_sum,
            "intersection": inter_s,
            "dice": round(s_dice, 4)
        })

    return {
        "dice": round(float(dice), 4),
        "dice_percentage": round(float(dice) * 100.0, 2),
        "jaccard": round(float(jaccard), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "intersection_voxels": intersection,
        "ref_voxels": ref_voxels,
        "model_voxels": model_voxels,
        "false_positive_voxels": fp,
        "false_negative_voxels": fn,
        "ref_volume_cm3": ref_vol_cm3,
        "model_volume_cm3": model_vol_cm3,
        "intersection_volume_cm3": inter_vol_cm3,
        "slice_metrics": slice_metrics
    }
