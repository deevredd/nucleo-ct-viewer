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
        self.affine_ras: Optional[np.ndarray] = None
        self.has_gantry_tilt: bool = False
        self.has_uniform_spacing: bool = True
        self.gantry_tilt_angle_deg: float = 0.0
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

        # Validate consistent SeriesInstanceUID and dimensions
        first = raw_slices[0]
        self.series_uid = getattr(first, 'SeriesInstanceUID', 'unknown')
        self.study_uid = getattr(first, 'StudyInstanceUID', 'unknown')
        self.rows = int(first.Rows)
        self.cols = int(first.Columns)
        self.pixel_spacing = (float(first.PixelSpacing[0]), float(first.PixelSpacing[1]))
        self.slice_thickness = float(getattr(first, 'SliceThickness', 1.0))

        iop_first = getattr(first, 'ImageOrientationPatient', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        X = np.array(iop_first[:3], dtype=float)
        Y = np.array(iop_first[3:], dtype=float)
        N = np.cross(X, Y)
        norm = np.linalg.norm(N)
        if norm > 1e-6:
            N = N / norm
        else:
            N = np.array([0.0, 0.0, 1.0])

        # Validate consistency across all slices
        for idx, d in enumerate(raw_slices):
            uid = getattr(d, 'SeriesInstanceUID', '')
            if uid and self.series_uid and uid != self.series_uid:
                raise ValueError(f"Multiple series detected: slice has SeriesInstanceUID '{uid}', expected '{self.series_uid}'.")
            if int(d.Rows) != self.rows or int(d.Columns) != self.cols:
                raise ValueError(f"Inconsistent slice dimensions: {d.Rows}x{d.Columns} vs {self.rows}x{self.cols}.")

        # Sort slices by physical position along the slice normal N
        def get_dist_along_normal(d):
            if hasattr(d, 'ImagePositionPatient') and len(d.ImagePositionPatient) >= 3:
                ipp = np.array([float(x) for x in d.ImagePositionPatient[:3]])
                return float(np.dot(ipp, N))
            elif hasattr(d, 'SliceLocation'):
                return float(d.SliceLocation)
            elif hasattr(d, 'InstanceNumber'):
                return float(d.InstanceNumber)
            return 0.0

        raw_slices.sort(key=get_dist_along_normal)
        self.num_slices = len(raw_slices)
        self.z_positions = [get_dist_along_normal(d) for d in raw_slices]

        # Detect duplicate slice positions or gaps
        if len(self.z_positions) > 1:
            diffs = np.diff(self.z_positions)
            if np.any(np.isclose(diffs, 0.0, atol=1e-3)):
                print("Warning: Duplicate slice positions detected in DICOM series.")
            median_spacing = float(np.median(diffs))
            if median_spacing > 0 and np.any(diffs > median_spacing * 1.5):
                print(f"Warning: Potential missing slices detected (expected ~{median_spacing:.2f}mm spacing).")

        # Extract HU pixel volume (num_slices, rows, cols)
        volume = np.zeros((self.num_slices, self.rows, self.cols), dtype=np.int16)
        for i, d in enumerate(raw_slices):
            arr = d.pixel_array.astype(np.float32)
            slope = float(getattr(d, 'RescaleSlope', 1.0))
            intercept = float(getattr(d, 'RescaleIntercept', 0.0))
            hu = arr * slope + intercept
            volume[i] = hu.astype(np.int16)

        self.volume_hu = volume

        # Construct 4x4 physical affine matrix in scanner RAS+ convention
        #
        # Coordinate Space & Axis Order Mapping:
        # - In DICOM: ImageOrientationPatient defines:
        #     X (iop[:3]): unit vector along columns (increasing column index c across a row)
        #     Y (iop[3:]): unit vector along rows (increasing row index r down a column)
        #     N = cross(X, Y): unit vector orthogonal to the axial slice plane
        # - In NIfTI: voxel array is ordered (i, j, k) = (columns, rows, slices) = (416, 350, 180).
        #     Voxel index i corresponds to DICOM column c (spacing = PixelSpacing[1])
        #     Voxel index j corresponds to DICOM row r (spacing = PixelSpacing[0])
        #     Voxel index k corresponds to DICOM slice k (spacing = step between consecutive slices)
        #
        # - Column 0 of Affine: maps index i (DICOM columns) to scanner RAS+ coordinates
        # - Column 1 of Affine: maps index j (DICOM rows) to scanner RAS+ coordinates
        # - Column 2 of Affine: maps index k (DICOM slices) to scanner RAS+ coordinates
        # - Column 3 of Affine: physical coordinate of voxel (0, 0, 0) in scanner RAS+
        col_spacing = float(self.pixel_spacing[1])
        row_spacing = float(self.pixel_spacing[0])
        first_ipp = np.array([float(x) for x in getattr(raw_slices[0], 'ImagePositionPatient', [0.0, 0.0, 0.0])[:3]])

        # Determine physical slice step vector in LPS
        if len(raw_slices) > 1:
            second_ipp = np.array([float(x) for x in getattr(raw_slices[1], 'ImagePositionPatient', [0.0, 0.0, 0.0])[:3]])
            step_lps = second_ipp - first_ipp
            step_len = float(np.linalg.norm(step_lps))
            step_dir = step_lps / step_len if step_len > 1e-6 else N
            
            # Gantry Tilt Detection: Check if slice step vector is collinear with slice normal N
            alignment = float(np.dot(step_dir, N))
            self.gantry_tilt_angle_deg = float(np.degrees(np.arccos(np.clip(abs(alignment), 0.0, 1.0))))
            self.has_gantry_tilt = (self.gantry_tilt_angle_deg > 0.5)
            
            # Non-uniform Slice Spacing Check
            diffs_array = np.diff(self.z_positions)
            median_step = float(np.median(diffs_array))
            spacing_variance = float(np.max(np.abs(diffs_array - median_step))) if len(diffs_array) > 0 else 0.0
            self.has_uniform_spacing = (spacing_variance < 0.05 * median_step) if median_step > 0 else True
            
            u_slice_lps = step_lps
        else:
            u_slice_lps = N * self.slice_thickness
            self.has_gantry_tilt = False
            self.has_uniform_spacing = True

        # Vectors in DICOM LPS+:
        u_col_lps = X * col_spacing
        u_row_lps = Y * row_spacing

        # Convert DICOM LPS+ to NIfTI RAS+ by flipping X and Y axes:
        # RAS_x = -LPS_x, RAS_y = -LPS_y, RAS_z = LPS_z
        u_col_ras = np.array([-u_col_lps[0], -u_col_lps[1], u_col_lps[2]])
        u_row_ras = np.array([-u_row_lps[0], -u_row_lps[1], u_row_lps[2]])
        u_slice_ras = np.array([-u_slice_lps[0], -u_slice_lps[1], u_slice_lps[2]])
        origin_ras = np.array([-first_ipp[0], -first_ipp[1], first_ipp[2]])

        aff_ras = np.eye(4, dtype=float)
        aff_ras[:3, 0] = u_col_ras
        aff_ras[:3, 1] = u_row_ras
        aff_ras[:3, 2] = u_slice_ras
        aff_ras[:3, 3] = origin_ras
        self.affine_ras = aff_ras

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
            "affine_ras": self.affine_ras.tolist() if self.affine_ras is not None else None,
            "has_gantry_tilt": self.has_gantry_tilt,
            "gantry_tilt_angle_deg": round(self.gantry_tilt_angle_deg, 2),
            "has_uniform_spacing": self.has_uniform_spacing,
            "presets": WINDOW_PRESETS
        }
