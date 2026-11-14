import io
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pydicom
from PIL import Image

WINDOW_PRESETS = {
    "soft_tissue": {"name": "Soft Tissue (Default)", "window": 400, "level": 40},
    "liver": {"name": "Liver Lesion", "window": 150, "level": 30},
    "abdomen": {"name": "Abdomen", "window": 350, "level": 50},
    "lung": {"name": "Lung", "window": 1500, "level": -600},
    "bone": {"name": "Bone", "window": 1800, "level": 400},
    "brain": {"name": "Brain", "window": 80, "level": 40},
}

class DicomSeries:
    def __init__(self):
        self.slices: List[Dict[str, Any]] = []
        self.volume_hu: Optional[np.ndarray] = None  # shape: (num_slices, rows, cols)
        self.rows: int = 0
        self.cols: int = 0
        self.num_slices: int = 0
        self.pixel_spacing: Tuple[float, float] = (1.0, 1.0)
        self.slice_thickness: float = 1.0
        self.series_uid: str = ""
        self.study_uid: str = ""
        self.z_positions: List[float] = []
        self.is_loaded: bool = False

    @classmethod
    def from_zip(cls, zip_input: Any) -> "DicomSeries":
        series = cls()
        series.load_from_zip(zip_input)
        return series

    def load_from_zip(self, zip_input: Any):
        """
        Load DICOM series from a zip file path, bytes, or file-like object.
        """
        if isinstance(zip_input, (str, Path)):
            zf = zipfile.ZipFile(zip_input, 'r')
        elif isinstance(zip_input, bytes):
            zf = zipfile.ZipFile(io.BytesIO(zip_input), 'r')
        else:
            zf = zipfile.ZipFile(zip_input, 'r')

        raw_slices = []
        for name in zf.namelist():
            # Skip metadata and non-dicom paths
            if name.startswith('__MACOSX') or name.endswith('/'):
                continue
            if not (name.lower().endswith('.dcm') or '.' not in Path(name).name):
                # We can check if it starts with DICM or try to read
                pass
            
            try:
                data = zf.read(name)
                # Parse DICOM
                dcm = pydicom.dcmread(io.BytesIO(data), force=True)
                if not hasattr(dcm, 'PixelData'):
                    continue
                raw_slices.append(dcm)
            except Exception:
                continue

        if not raw_slices:
            raise ValueError("No valid DICOM files found in zip archive.")

        # Validate consistent SeriesInstanceUID
        first = raw_slices[0]
        self.series_uid = getattr(first, 'SeriesInstanceUID', 'unknown')
        self.study_uid = getattr(first, 'StudyInstanceUID', 'unknown')
        self.rows = int(first.Rows)
        self.cols = int(first.Columns)
        self.pixel_spacing = (float(first.PixelSpacing[0]), float(first.PixelSpacing[1]))
        self.slice_thickness = float(getattr(first, 'SliceThickness', 1.0))

        # Sort slices by ImagePositionPatient[2] (axial Z)
        def get_z(d):
            if hasattr(d, 'ImagePositionPatient') and len(d.ImagePositionPatient) >= 3:
                return float(d.ImagePositionPatient[2])
            elif hasattr(d, 'SliceLocation'):
                return float(d.SliceLocation)
            elif hasattr(d, 'InstanceNumber'):
                return float(d.InstanceNumber)
            return 0.0

        raw_slices.sort(key=get_z)
        self.num_slices = len(raw_slices)
        self.z_positions = [get_z(d) for d in raw_slices]

        # Extract HU pixel volume (num_slices, rows, cols)
        volume = np.zeros((self.num_slices, self.rows, self.cols), dtype=np.int16)
        for i, d in enumerate(raw_slices):
            arr = d.pixel_array.astype(np.float32)
            slope = float(getattr(d, 'RescaleSlope', 1.0))
            intercept = float(getattr(d, 'RescaleIntercept', 0.0))
            hu = arr * slope + intercept
            volume[i] = hu.astype(np.int16)

        self.volume_hu = volume
        self.is_loaded = True

    def get_slice_hu(self, slice_idx: int) -> np.ndarray:
        if not self.is_loaded or self.volume_hu is None:
            raise RuntimeError("DICOM series is not loaded.")
        if slice_idx < 0 or slice_idx >= self.num_slices:
            raise IndexError(f"Slice index {slice_idx} out of range (0..{self.num_slices-1})")
        return self.volume_hu[slice_idx]

    def render_slice_png(
        self,
        slice_idx: int,
        window: float = 400.0,
        level: float = 40.0
    ) -> bytes:
        """
        Apply windowing (window width, window level) and return PNG bytes.
        """
        hu = self.get_slice_hu(slice_idx).astype(np.float32)
        
        # Window / Level formula:
        # lower = level - window / 2
        # upper = level + window / 2
        lower = level - (window / 2.0)
        upper = level + (window / 2.0)
        
        clipped = np.clip(hu, lower, upper)
        norm = ((clipped - lower) / (upper - lower) * 255.0).astype(np.uint8)
        
        img = Image.fromarray(norm)
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=False)
        return buf.getvalue()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "is_loaded": self.is_loaded,
            "num_slices": self.num_slices,
            "rows": self.rows,
            "cols": self.cols,
            "pixel_spacing": list(self.pixel_spacing),
            "slice_thickness": self.slice_thickness,
            "series_uid": self.series_uid,
            "study_uid": self.study_uid,
            "z_range": [min(self.z_positions), max(self.z_positions)] if self.z_positions else [],
            "presets": WINDOW_PRESETS
        }
