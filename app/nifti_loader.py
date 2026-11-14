import io
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import nibabel as nib
from PIL import Image

class NiftiMask:
    def __init__(self, name: str = "mask", color: Tuple[int, int, int, int] = (16, 185, 129, 140)):
        self.name: str = name
        self.color: Tuple[int, int, int, int] = color  # RGBA
        self.data: Optional[np.ndarray] = None  # shape: (cols, rows, slices) = (416, 350, 180)
        self.affine: Optional[np.ndarray] = None
        self.shape: Tuple[int, ...] = ()
        self.spacing: Tuple[float, ...] = ()
        self.positive_voxels: int = 0
        self.is_loaded: bool = False

    @classmethod
    def from_file(cls, path: Any, name: str = "mask", color: Tuple[int, int, int, int] = (16, 185, 129, 140)) -> "NiftiMask":
        mask = cls(name=name, color=color)
        mask.load(path)
        return mask

    @classmethod
    def from_bytes(cls, raw_bytes: bytes, name: str = "mask", color: Tuple[int, int, int, int] = (245, 158, 11, 140)) -> "NiftiMask":
        mask = cls(name=name, color=color)
        mask.load_bytes(raw_bytes)
        return mask

    def load(self, path: Any):
        img = nib.load(str(path))
        self._init_from_nib(img)

    def load_bytes(self, raw_bytes: bytes):
        # Load from in-memory gzip bytes
        fh = nib.FileHolder(fileobj=io.BytesIO(raw_bytes))
        img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        self._init_from_nib(img)

    def _init_from_nib(self, img: nib.Nifti1Image):
        self.data = np.asanyarray(img.dataobj, dtype=np.uint8)
        self.affine = np.array(img.affine, dtype=float)
        self.shape = tuple(self.data.shape)
        header = img.header
        zooms = header.get_zooms()
        self.spacing = tuple(float(z) for z in zooms[:3])
        self.positive_voxels = int(np.sum(self.data > 0))
        self.is_loaded = True

    def get_slice_2d(self, slice_idx: int) -> np.ndarray:
        """
        Extract 2D mask for axial slice `slice_idx`.
        Transposes from NIfTI (cols, rows) to DICOM (rows, cols).
        """
        if not self.is_loaded or self.data is None:
            raise RuntimeError(f"Mask {self.name} is not loaded.")
        if slice_idx < 0 or slice_idx >= self.shape[2]:
            raise IndexError(f"Slice index {slice_idx} out of range (0..{self.shape[2]-1})")
        
        # In NIfTI, shape is (cols, rows, slices)
        # Slicing along axis 2 gives (cols, rows)
        # Transpose (.T) produces (rows, cols) = (350, 416)
        slice_cols_rows = self.data[:, :, slice_idx]
        return slice_cols_rows.T

    def render_slice_png(
        self,
        slice_idx: int,
        color_rgba: Optional[Tuple[int, int, int, int]] = None
    ) -> bytes:
        """
        Render RGBA PNG of the mask for the specified slice.
        Pixels with value > 0 are colored; background is fully transparent.
        """
        mask_2d = self.get_slice_2d(slice_idx)
        rows, cols = mask_2d.shape
        
        rgba = color_rgba or self.color
        r, g, b, a = rgba
        
        out = np.zeros((rows, cols, 4), dtype=np.uint8)
        pos = (mask_2d > 0)
        out[pos, 0] = r
        out[pos, 1] = g
        out[pos, 2] = b
        out[pos, 3] = a
        
        img = Image.fromarray(out)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "is_loaded": self.is_loaded,
            "shape": list(self.shape),
            "spacing": list(self.spacing),
            "positive_voxels": self.positive_voxels,
            "color": list(self.color)
        }
