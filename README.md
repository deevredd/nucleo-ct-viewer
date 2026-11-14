# Nucleo OmniLesion CT Viewer

An interactive clinical CT axial slice viewer and automated pan-cancer lesion segmentation inspection tool built for the **Nucleo Research AI Engineer Take-Home**.

The application connects to the Nucleo inference API (which mimics the award-winning **OmniLesion** model from FLARE 2026 / MICCAI 2026), loads and overlays dynamic model segmentation masks onto an axial CT series, compares them against a manual ground-truth reference segmentation, validates spatial grid alignment, computes the Dice Similarity Coefficient (DSC) and volumetric statistics, and robustly handles network retries, concurrent uploads, and error states.

---

## Quickstart (One Command)

### Prerequisites
- Python 3.9+ installed
- Bash shell (macOS or Linux)

### Run
To launch the viewer with a single command:

```bash
./run.sh
```

Or directly with Python:

```bash
python run.py
```

The script sets up a virtual environment in `./venv` (if not already created), installs dependencies from `requirements.txt`, initializes the viewer, loads `case-001`, and starts the local server at **[http://localhost:8000](http://localhost:8000)** (auto-opening your default browser).

---

## Configuration

The API base URL and personal authorization token are configured via environment variables or a `.env` file (copied from `.env.example`). They can also be modified live directly inside the UI via the **Settings (gear icon)** modal.

### Setting Environment Variables

```bash
export NUCLEO_API_BASE_URL="https://bnj2vb4zpt.us-east-1.awsapprunner.com"
export NUCLEO_API_TOKEN="tid_..."
python run.py
```

### `.env` File
Create a `.env` file in the project root:

```ini
NUCLEO_API_BASE_URL=https://bnj2vb4zpt.us-east-1.awsapprunner.com
NUCLEO_API_TOKEN=tid_...
PORT=8000
HOST=127.0.0.1
```

> **Security Note:** `.env` and personal token credentials are added to `.gitignore` and are never committed to version control.

---

## How Dice Similarity Coefficient (DSC) Was Computed

### 1. Spatial Grid Compatibility Verification
Before computing DSC, the app strictly verifies whether the returned model mask and the ground-truth reference mask reside on the same spatial grid:
- **Shape Check:** Ensures identical array dimensions (`(416, 350, 180)`).
- **Affine & Spacing Check:** Validates that the 4×4 affine transformation matrix (mapping voxel `(i, j, k, 1)` to scanner coordinates in LPS/RAS+ mm) matches within tolerance (`atol=1e-3`).

### 2. DSC Formula
For binary masks $X$ (Model prediction) and $Y$ (Reference ground-truth):

$$\text{DSC} = \frac{2 \cdot |X \cap Y|}{|X| + |Y|}$$

- **Intersection ($|X \cap Y|$):** Number of positive voxels where both $X = 1$ and $Y = 1$.
- **Edge cases:** If $|X| + |Y| = 0$ (both masks empty), $\text{DSC} = 1.0$. If one is empty and the other non-empty, $\text{DSC} = 0.0$.
- **Measurements in cm³:** Voxel volume is computed directly from the physical affine matrix determinant $|\det(\mathbf{A}_{3 \times 3})|$ to ensure exact volume measurements even under arbitrary rotations, shears, or anisotropic spacing:
  $$\text{Volume (cm}^3\text{)} = \frac{\text{Voxels} \times |\det(\mathbf{A}_{3 \times 3})|}{1000}$$
  For case-001, $|\det(\mathbf{A}_{3 \times 3})| = |-0.708984 \times -0.708984 \times 1.000| \approx 0.50266\text{ mm}^3/\text{voxel}$.
- **Additional Clinical Metrics:**
  - Jaccard Index (IoU): $\frac{|X \cap Y|}{|X \cup Y|}$
  - Precision (PPV): $\frac{|X \cap Y|}{|X|}$
  - Recall (Sensitivity): $\frac{|X \cap Y|}{|Y|}$
  - Per-slice lesion distribution profile along the axial $z$-axis.

For **case-001**, the OmniLesion model achieves **$\text{DSC} = 0.9284$ (92.84%)**, with $364.7\text{ cm}^3$ intersection volume out of $398.7\text{ cm}^3$ reference and $386.9\text{ cm}^3$ predicted volume ($0.70898 \times 0.70898 \times 1.000\text{ mm} \approx 0.5027\text{ mm}^3/\text{voxel}$).

---

## Failure Cases Handled

The application is engineered to never crash or freeze when encountering invalid input, server hiccups, or edge cases. Errors are mapped into clear, plain language tailored for clinical or non-technical users by branching strictly on `error.code` and `details.reason` (rather than raw message strings):

1. **HTTP 400 Bad Request:**
   - `empty_body` / `empty_archive`: Plain-language alert when an archive has no usable DICOM files.
   - `not_zip` / `corrupt_zip`: Alerts user to damaged or invalid ZIP files.
   - `not_dicom` / `corrupt_dicom` / `truncated_pixel_data`: Identifies invalid or corrupted slice data.
   - `too_many_entries` / `uncompressed_too_large` / `member_too_large`: Flags archives exceeding safe limits.
2. **HTTP 401 Unauthorized:**
   - Differentiates between `missing`, `invalid`, `expired`, and `revoked` tokens.
3. **HTTP 404 Not Found:**
   - `unknown_series`: Informs the user that the uploaded DICOM series does not match any recognized study.
4. **HTTP 413 Too Large & 415 Unsupported Media Type:**
   - Explains size limits (150 MiB) or warns if a non-ZIP content type is submitted.
5. **HTTP 422 Unprocessable Entity:**
   - Handles `missing_tags`, `multiple_series`, `inconsistent_geometry`, `incomplete_series`, `inconsistent_spacing`, `duplicate_position`, and `unsupported_transfer_syntax`.
6. **HTTP 429 Rate Limited & 503 Unavailable:**
   - Parses the `Retry-After` response header (or `details.retry_after_s`).
   - Applies exponential backoff with random jitter (`retry_after * 1.5^attempt + jitter`).
   - Displays an interactive retry countdown timer on the screen ("Retrying in X seconds (Attempt N)...").
7. **HTTP 408 / Slow Uplink Timeout:**
   - If an uplink is constrained and exceeds the server's 60-second receive timeout, the app captures the timeout gracefully without crashing and offers an instant "Quick Demo" fallback to inspect pre-cached model masks.
8. **Concurrent Requests Isolation:**
   - Every upload is assigned an immutable, isolated `job_id` (UUID).
   - In-flight uploads maintain independent state machines and mask buffers. Concurrent uploads never overwrite or cross-contaminate one another. The UI includes a Job Queue switcher to view results from any concurrent run.

---

## Features

- **Axial Slice CT Viewer:** 60fps canvas renderer with mouse wheel scroll, keyboard arrow keys (`↑`/`↓`), and interactive scrubber.
- **Radiology Windowing:** One-click presets for Soft Tissue (400/40), Liver (150/30), Abdomen (350/50), Lung (1500/-600), Bone (1800/400), and Brain (80/40), plus continuous Window Width / Window Level sliders.
- **Live HU Cursor Readout:** Probes Hounsfield Units directly under the mouse pointer with anatomic tissue classification (Air, Fat, Water, Soft Tissue, Bone).
- **Dynamic Mask Overlays:** Overlays whatever structures the API lists (`lesions`, organs, etc.) with independent toggles, color badges, and opacity sliders.
- **Reference Comparison:** Side-by-side comparison with the ground-truth mask, showing true positive overlap vs model discrepancies.
- **Lesion Distribution Mini-Chart:** Visualizes lesion voxel density across slices $z=34$ to $z=164$.
- **Pan & Zoom:** Interactive drag-to-pan, Ctrl+wheel / pinch to zoom, and one-click "Reset View".

---

## Automated Test Suite

Run the focused test suite using pytest:

```bash
./venv/bin/pytest -v
```

The 30 tests cover:
- API response normalization supporting both dictionary (`{"lesions": url}`) and list (`[{"name": "lesions", "url": ...}]`) structure formats.
- Disabling automatic retries for ambiguous POST timeouts to prevent duplicate backend jobs.
- Target-specific lesion identification for Dice metrics (preventing erroneous organ mask fallbacks).
- Reconstructed DICOM-to-NIfTI physical affine compatibility (`check_ct_to_mask_compatibility`) confirming maximum coordinate difference $\Delta < 1.25 \times 10^{-8}\text{ mm}$ within floating-point tolerance.
- Gantry tilt angle calculation and detection for non-orthogonal slice stepping.
- Non-uniform slice spacing variance detection across the axial axis.
- Exact physical volume computation using the affine determinant $|\det(\mathbf{A}_{3 \times 3})|$.
- Failed uploads (`bad_request: empty_archive`, `corrupt_zip`, `not_dicom`, etc.) and plain-language translations.
- Multiple DICOM series detection (`unprocessable: multiple_series`).
- Inconsistent slice dimensions and non-uniform geometry.
- Authorization handling and expired tokens (`unauthorized: expired`).
- Honouring `Retry-After` on 429 and 503 with backoff and retry.
- Concurrent requests isolation verifying results are never mixed.
- DICOM series loading, slice ordering along the axial $z$-axis, and HU conversion.
- In-memory Gzip decompression and header decoding for `.nii.gz` masks.
- Grid compatibility validation (exact match vs mismatched dimensions or affine).
- Dice computation across edge cases (perfect overlap, partial overlap, zero overlap, empty masks).
- End-to-end FastAPI endpoint integration, diff map rendering, and job history clearing.

---

## What I Would Do Next (With More Time)

If allotted additional time, the following clinical and technical capabilities would be prioritized, directly aligning with Nucleo's longitudinal oncology platform:

1. **Automated RECIST 1.1 Tumor Tracking:** Perform 3D connected-component analysis on predicted lesion masks to segment individual lesions, determine longest axial diameters in the transaxial plane, and classify lesions into target vs. non-target lesions for longitudinal tumor burden response tracking.
2. **Body Composition & Sarcopenia Profiling:** Segment skeletal muscle and adipose tissue (subcutaneous and visceral fat) at the third lumbar vertebra (L3) cross-section to compute Skeletal Muscle Index (SMI) and sarcopenia staging.
3. **Multi-Planar Reconstruction (MPR):** Implement real-time Sagittal and Coronal orthogonal re-slicing alongside the Axial plane to allow oncologists and radiologists to inspect tumor margins across all three anatomical planes.
4. **Interactive Radiologist Annotation & Refinement:** Allow clinicians to touch up or correct predicted boundaries using an interactive brush and export the adjusted contours as standardized DICOM RT-STRUCT or NIfTI for downstream radiation therapy or surgical resection planning.
5. **Chunked Streaming Uploader:** Implement chunked file transfer with resume capability for constrained hospital network uplinks.

---

## Note on AI Coding Tools

This solution was developed using Antigravity AI coding tools for rapid scaffolding, exploration, and test generation. All mathematical formulations (DICOM LPS+ to NIfTI RAS+ affine transformation, vector projection $\mathbf{IPP} \cdot \mathbf{N}$, voxel volume calculations), architecture, error-branching logic, and test suites have been designed, reviewed, and verified from first principles.

---

## Attribution

The CT scan and reference lesion mask are derived from:
- Wasserthal, J. (2026). *Liver lesions CT training dataset*. Zenodo. [DOI: 10.5281/zenodo.20272572](https://doi.org/10.5281/zenodo.20272572). Licensed under CC BY 4.0.
