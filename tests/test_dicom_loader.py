import io
import zipfile
import pytest
import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage
from app.dicom_loader import DicomSeries

def create_synthetic_dicom_bytes(instance_number: int, z_pos: float, pixel_val: int = 40) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = f"1.2.3.4.{instance_number}"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = f"1.2.3.4.{instance_number}"
    ds.SeriesInstanceUID = "1.2.3.4"
    ds.StudyInstanceUID = "1.2.3"
    ds.InstanceNumber = instance_number
    ds.ImagePositionPatient = [0.0, 0.0, z_pos]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.Rows = 16
    ds.Columns = 16
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed int16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024  # Standard CT offset

    pixels = np.full((16, 16), pixel_val + 1024, dtype=np.int16)
    ds.PixelData = pixels.tobytes()

    bio = io.BytesIO()
    ds.save_as(bio, write_like_original=False)
    return bio.getvalue()

def test_dicom_series_loading_and_sorting():
    # Create zip with slices out of order
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, 'w') as zf:
        zf.writestr("slice2.dcm", create_synthetic_dicom_bytes(2, 20.0, pixel_val=50))
        zf.writestr("slice1.dcm", create_synthetic_dicom_bytes(1, 10.0, pixel_val=30))
        zf.writestr("slice3.dcm", create_synthetic_dicom_bytes(3, 30.0, pixel_val=70))

    series = DicomSeries.from_zip(zbuf.getvalue())
    assert series.num_slices == 3
    assert series.rows == 16
    assert series.cols == 16
    # Verify sorting by Z position
    assert series.z_positions == [10.0, 20.0, 30.0]
    # Verify HU conversion (pixel + slope + intercept = (val+1024)*1 - 1024 = val)
    assert series.get_slice_hu(0)[0, 0] == 30
    assert series.get_slice_hu(1)[0, 0] == 50
    assert series.get_slice_hu(2)[0, 0] == 70

def test_dicom_series_windowing_png():
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, 'w') as zf:
        zf.writestr("slice1.dcm", create_synthetic_dicom_bytes(1, 0.0, pixel_val=40))

    series = DicomSeries.from_zip(zbuf.getvalue())
    png_bytes = series.render_slice_png(0, window=400, level=40)
    assert len(png_bytes) > 0
    assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n'
