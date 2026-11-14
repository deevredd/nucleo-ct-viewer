import gzip
import pytest
from app.nifti_loader import NiftiMask

def test_load_bytes_gzipped(tmp_path):
    with open("data/lesions.nii.gz", "rb") as f:
        gz_bytes = f.read()

    assert gz_bytes[:2] == b'\x1f\x8b'
    mask = NiftiMask.from_bytes(gz_bytes, name="lesions")
    assert mask.is_loaded is True
    assert mask.shape == (416, 350, 180)
    assert mask.positive_voxels > 0

def test_load_bytes_uncompressed():
    with open("data/lesions.nii.gz", "rb") as f:
        gz_bytes = f.read()
    raw_bytes = gzip.decompress(gz_bytes)
    
    mask = NiftiMask.from_bytes(raw_bytes, name="lesions")
    assert mask.is_loaded is True
    assert mask.shape == (416, 350, 180)
