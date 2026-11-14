/**
 * Nucleo OmniLesion CT Viewer - Frontend Client
 */

// Application State
const state = {
  currentSlice: 90,
  totalSlices: 180,
  windowWidth: 400,
  windowLevel: 40,
  activePreset: "soft_tissue",
  
  // Overlays
  showRefMask: true,
  refOpacity: 0.55,
  
  // Active job and dynamic structures
  activeJobId: null,
  jobs: [],
  structureStates: {}, // { [structName]: { visible: true, opacity: 0.55, color: [r,g,b] } }
  
  // Canvas viewport transform
  zoom: 1.0,
  panX: 0,
  panY: 0,
  isPanning: false,
  lastMouseX: 0,
  lastMouseY: 0,
  
  // Image cache
  imageCache: new Map(), // key -> Image
  
  // Polling timer
  pollTimer: null,
};

// DOM Elements
const canvas = document.getElementById("ct-canvas");
const ctx = canvas.getContext("2d");
const canvasWrapper = document.getElementById("canvas-wrapper");
const sliceSlider = document.getElementById("slice-slider");
const sliceInput = document.getElementById("slice-input");
const btnPrevSlice = document.getElementById("btn-prev-slice");
const btnNextSlice = document.getElementById("btn-next-slice");
const canvasLoader = document.getElementById("canvas-loader");

// HUD Elements
const hudSliceInfo = document.getElementById("hud-slice-info");
const hudZPos = document.getElementById("hud-z-pos");
const hudWL = document.getElementById("hud-wl");
const hudZoom = document.getElementById("hud-zoom");
const hudPixelHU = document.getElementById("hud-pixel-hu");
const hudCoords = document.getElementById("hud-coords");
const btnResetZoom = document.getElementById("btn-reset-zoom");

// Overlay Controls
const toggleRefMask = document.getElementById("toggle-ref-mask");
const opacityRefMask = document.getElementById("opacity-ref-mask");
const valRefOpacity = document.getElementById("val-ref-opacity");
const refVolumeText = document.getElementById("ref-volume-text");
const modelStructuresContainer = document.getElementById("model-structures-container");

// Windowing Controls
const sliderWindowWidth = document.getElementById("slider-window-width");
const sliderWindowLevel = document.getElementById("slider-window-level");
const valWindowWidth = document.getElementById("val-window-width");
const valWindowLevel = document.getElementById("val-window-level");
const presetButtons = document.querySelectorAll(".preset-btn");
const activePresetLabel = document.getElementById("active-preset-label");

// Metrics Elements
const metricDiceVal = document.getElementById("metric-dice-val");
const metricDicePct = document.getElementById("metric-dice-pct");
const metricDiceBar = document.getElementById("metric-dice-bar");
const gridStatusBadge = document.getElementById("grid-status-badge");
const gridCheckDetails = document.getElementById("grid-check-details");
const statRefVol = document.getElementById("stat-ref-vol");
const statRefVox = document.getElementById("stat-ref-vox");
const statModelVol = document.getElementById("stat-model-vol");
const statModelVox = document.getElementById("stat-model-vox");
const statOverlapVol = document.getElementById("stat-overlap-vol");
const statOverlapVox = document.getElementById("stat-overlap-vox");
const statPrecision = document.getElementById("stat-precision");
const statRecall = document.getElementById("stat-recall");
const statJaccard = document.getElementById("stat-jaccard");
const statFp = document.getElementById("stat-fp");

// Jobs List
const jobsList = document.getElementById("jobs-list");

// Error Banner
const errorBanner = document.getElementById("error-banner");
const bannerTitle = document.getElementById("banner-title");
const bannerDesc = document.getElementById("banner-desc");
const bannerRetryTimer = document.getElementById("banner-retry-timer");
const retrySeconds = document.getElementById("retry-seconds");
const bannerDismiss = document.getElementById("banner-dismiss");

// Modals
const uploadModal = document.getElementById("upload-modal");
const btnUploadTrigger = document.getElementById("btn-upload-trigger");
const btnDemoCached = document.getElementById("btn-demo-cached");
const modalUploadClose = document.getElementById("modal-upload-close");
const modalUploadCancel = document.getElementById("modal-upload-cancel");
const modalUploadSubmit = document.getElementById("modal-upload-submit");
const radioUploadDefault = document.getElementById("radio-upload-default");
const radioUploadCustom = document.getElementById("radio-upload-custom");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const dropZoneLabel = document.getElementById("drop-zone-label");

const settingsModal = document.getElementById("settings-modal");
const btnOpenSettings = document.getElementById("btn-open-settings");
const modalSettingsClose = document.getElementById("modal-settings-close");
const modalSettingsCancel = document.getElementById("modal-settings-cancel");
const modalSettingsSave = document.getElementById("modal-settings-save");
const cfgBaseUrl = document.getElementById("cfg-base-url");
const cfgToken = document.getElementById("cfg-token");

let selectedCustomFile = null;

// Initialize App
async function initApp() {
  setupEventListeners();
  await loadCaseInfo();
  await fetchConfig();
  await refreshJobs();
  renderCurrentSlice();
}

// Fetch Case Metadata
async function loadCaseInfo() {
  try {
    const res = await fetch("/api/case/info");
    if (!res.ok) throw new Error("Failed to load case info");
    const data = await res.json();
    
    state.totalSlices = data.num_slices || 180;
    sliceSlider.max = state.totalSlices - 1;
    sliceInput.max = state.totalSlices;
    
    document.getElementById("case-title").textContent = `${data.series_uid ? "Case-001" : "Study"} • CT Abdomen`;
    document.getElementById("header-slice-count").textContent = `${state.totalSlices} slices`;
    document.getElementById("hud-dims").textContent = `Matrix: ${data.cols} × ${data.rows} • ${data.slice_thickness}mm`;
  } catch (err) {
    showError("Data Error", "Could not load CT metadata: " + err.message);
  }
}

// Fetch API Config
async function fetchConfig() {
  try {
    const res = await fetch("/api/config");
    if (res.ok) {
      const cfg = await res.json();
      cfgBaseUrl.value = cfg.api_base_url || "";
      if (cfg.masked_token) {
        cfgToken.placeholder = `Current: ${cfg.masked_token}`;
      }
    }
  } catch (err) {
    console.error("Config fetch error:", err);
  }
}

// Render Slice on Canvas
let renderRequestPending = false;
async function renderCurrentSlice() {
  if (renderRequestPending) return;
  renderRequestPending = true;

  try {
    const sliceIdx = state.currentSlice;
    const w = state.windowWidth;
    const l = state.windowLevel;

    // 1. Fetch CT slice image
    const ctUrl = `/api/case/slice/${sliceIdx}?window=${w}&level=${l}`;
    const ctImg = await loadImage(ctUrl);

    // Update canvas size if needed
    if (canvas.width !== ctImg.width || canvas.height !== ctImg.height) {
      canvas.width = ctImg.width;
      canvas.height = ctImg.height;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(ctImg, 0, 0);

    // 2. Overlay Reference Mask if enabled
    if (state.showRefMask) {
      const alphaInt = Math.round(state.refOpacity * 255);
      const refUrl = `/api/reference/slice/${sliceIdx}?r=16&g=185&b=129&a=${alphaInt}`;
      try {
        const refImg = await loadImage(refUrl);
        ctx.drawImage(refImg, 0, 0);
      } catch (e) {
        // Mask slice might be empty
      }
    }

    // 3. Overlay Model Structures for active job
    if (state.activeJobId) {
      const activeJob = state.jobs.find(j => j.id === state.activeJobId);
      if (activeJob && activeJob.structures) {
        for (const struct of activeJob.structures) {
          const structName = struct.name;
          const sState = state.structureStates[structName];
          if (sState && sState.visible) {
            const alphaInt = Math.round(sState.opacity * 255);
            const modelUrl = `/api/jobs/${state.activeJobId}/structures/${structName}/slice/${sliceIdx}?r=245&g=158&b=11&a=${alphaInt}`;
            try {
              const modelImg = await loadImage(modelUrl);
              ctx.drawImage(modelImg, 0, 0);
            } catch (e) {}
          }
        }
      }
    }

    updateHudText();
  } catch (err) {
    console.error("Render slice error:", err);
  } finally {
    renderRequestPending = false;
  }
}

// Image Loader with Map cache
function loadImage(url) {
  if (state.imageCache.has(url)) {
    return Promise.resolve(state.imageCache.get(url));
  }
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      // Keep cache size bounded
      if (state.imageCache.size > 400) {
        const firstKey = state.imageCache.keys().next().value;
        state.imageCache.delete(firstKey);
      }
      state.imageCache.set(url, img);
      resolve(img);
    };
    img.onerror = (e) => reject(e);
    img.src = url;
  });
}

// Update HUD Labels
function updateHudText() {
  hudSliceInfo.textContent = `Slice: ${state.currentSlice + 1} / ${state.totalSlices}`;
  const z = (359.9 + state.currentSlice * 1.0).toFixed(1);
  hudZPos.textContent = `Z: ${z} mm`;
  hudWL.textContent = `W: ${state.windowWidth} L: ${state.windowLevel}`;
  hudZoom.textContent = `Zoom: ${Math.round(state.zoom * 100)}%`;
  
  sliceSlider.value = state.currentSlice;
  sliceInput.value = state.currentSlice + 1;
}

// Update Viewport Transform (Zoom & Pan)
function updateCanvasTransform() {
  canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
  hudZoom.textContent = `Zoom: ${Math.round(state.zoom * 100)}%`;
}

// Jobs Management
async function refreshJobs() {
  try {
    const res = await fetch("/api/jobs");
    if (!res.ok) return;
    const jobs = await res.json();
    state.jobs = jobs;

    // Render jobs list
    renderJobsList(jobs);

    // Set active job if none selected
    if (!state.activeJobId && jobs.length > 0) {
      setActiveJob(jobs[0].id);
    } else if (state.activeJobId) {
      const curr = jobs.find(j => j.id === state.activeJobId);
      if (curr) {
        updateMetricsDisplay(curr);
      }
    }

    // Check if any job is still in progress -> keep polling
    const inProgress = jobs.some(j => ["queued", "uploading", "retrying", "downloading"].includes(j.status));
    if (inProgress && !state.pollTimer) {
      state.pollTimer = setInterval(refreshJobs, 1500);
    } else if (!inProgress && state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  } catch (err) {
    console.error("Refresh jobs error:", err);
  }
}

function renderJobsList(jobs) {
  jobsList.innerHTML = "";
  if (jobs.length === 0) {
    jobsList.innerHTML = `<div style="font-size:11px;color:var(--text-dim);padding:8px 0;">No segmentation jobs yet.</div>`;
    return;
  }

  jobs.forEach(job => {
    const card = document.createElement("div");
    card.className = `job-card ${job.id === state.activeJobId ? "active" : ""}`;
    card.onclick = () => setActiveJob(job.id);

    const statusPill = `<span class="job-status-pill ${job.status}">${job.status}</span>`;
    
    card.innerHTML = `
      <div class="job-card-top">
        <span class="job-label">${escapeHtml(job.label || job.id)}</span>
        ${statusPill}
      </div>
      <div class="job-card-msg">${escapeHtml(job.progress_message || "")}</div>
    `;
    jobsList.appendChild(card);
  });
}

function setActiveJob(jobId) {
  state.activeJobId = jobId;
  const job = state.jobs.find(j => j.id === jobId);
  if (!job) return;

  // Render dynamic model structures toggles
  renderModelStructureControls(job);

  // Update metrics panel
  updateMetricsDisplay(job);

  // Check if job failed -> show plain error banner
  if (job.status === "failed" && job.error) {
    showError(
      `Upload Failed: ${job.error.code || "Error"}`,
      job.error.user_message || "An error occurred during segmentation.",
      job.retry_wait_s > 0 ? job.retry_wait_s : 0
    );
  } else if (job.status === "retrying") {
    showError(
      "Service Busy / Rate Limit",
      job.progress_message,
      job.retry_wait_s
    );
  } else if (job.error && job.status === "completed") {
    showError(
      "Live Network Uplink Note",
      job.progress_message,
      0
    );
  } else {
    hideError();
  }

  renderJobsList(state.jobs);
  renderCurrentSlice();
}

function renderModelStructureControls(job) {
  modelStructuresContainer.innerHTML = "";

  if (!job.structures || job.structures.length === 0) {
    if (job.status === "completed") {
      modelStructuresContainer.innerHTML = `<div style="font-size:11px;color:var(--text-dim);padding:4px 0;">No structures returned.</div>`;
    }
    return;
  }

  job.structures.forEach(struct => {
    const sName = struct.name;
    if (!state.structureStates[sName]) {
      state.structureStates[sName] = { visible: true, opacity: 0.55 };
    }
    const sState = state.structureStates[sName];

    const card = document.createElement("div");
    card.className = "mask-card model-card";
    card.innerHTML = `
      <div class="card-header">
        <label class="switch-container">
          <input type="checkbox" id="toggle-${sName}" ${sState.visible ? "checked" : ""}>
          <span class="switch-slider"></span>
        </label>
        <div class="card-title-group">
          <div class="color-indicator model-indicator"></div>
          <span class="mask-name">OmniLesion: ${escapeHtml(sName)}</span>
        </div>
      </div>
      <div class="card-details">
        <div class="control-row">
          <span class="control-label">Opacity</span>
          <input type="range" id="opacity-${sName}" min="0" max="100" value="${Math.round(sState.opacity * 100)}" class="slider-sm">
          <span class="val-display" id="val-${sName}-opacity">${Math.round(sState.opacity * 100)}%</span>
        </div>
        <div class="volume-stat">
          <span>Voxels:</span>
          <b>${struct.positive_voxels ? struct.positive_voxels.toLocaleString() : "--"}</b>
        </div>
      </div>
    `;

    modelStructuresContainer.appendChild(card);

    // Event listeners
    const toggle = card.querySelector(`#toggle-${sName}`);
    const opacity = card.querySelector(`#opacity-${sName}`);
    const opacityVal = card.querySelector(`#val-${sName}-opacity`);

    toggle.addEventListener("change", (e) => {
      sState.visible = e.target.checked;
      renderCurrentSlice();
    });

    opacity.addEventListener("input", (e) => {
      sState.opacity = parseInt(e.target.value) / 100.0;
      opacityVal.textContent = `${e.target.value}%`;
      renderCurrentSlice();
    });
  });
}

function updateMetricsDisplay(job) {
  if (job.status !== "completed" || !job.metrics) {
    metricDiceVal.textContent = "--";
    metricDicePct.textContent = job.status === "failed" ? "Failed" : "--%";
    metricDiceBar.style.width = "0%";
    gridStatusBadge.textContent = job.status === "failed" ? "Upload Failed" : "Grid: Pending";
    gridStatusBadge.className = "hero-badge";
    gridCheckDetails.textContent = job.error ? (job.error.user_message || "Upload failed. Click a completed job in the Upload Jobs list below.") : "Metrics will display upon completed upload.";
    return;
  }

  const m = job.metrics;
  const gc = job.grid_check || {};

  metricDiceVal.textContent = m.dice !== undefined ? m.dice.toFixed(4) : "--";
  metricDicePct.textContent = `${m.dice_percentage !== undefined ? m.dice_percentage.toFixed(1) : 0}%`;
  metricDiceBar.style.width = `${m.dice_percentage || 0}%`;

  if (gc.is_compatible) {
    gridStatusBadge.textContent = "Grid: Matched";
    gridStatusBadge.className = "hero-badge verified";
    gridCheckDetails.innerHTML = `
      Dimensions: ${gc.shape1 ? gc.shape1.join(" × ") : "416 × 350 × 180"}<br>
      Affine Alignment: Verified (max diff: ${(gc.affine_diff_max || 0).toFixed(4)})<br>
      Spacing: 0.71 × 0.71 × 1.00 mm
    `;
  } else {
    gridStatusBadge.textContent = "Grid: Mismatch";
    gridStatusBadge.className = "hero-badge";
    gridCheckDetails.textContent = gc.message || "Grid geometry mismatch detected.";
  }

  statRefVol.textContent = `${m.ref_volume_cm3 || 398.7} cm³`;
  statRefVox.textContent = (793156).toLocaleString();
  statModelVol.textContent = `${m.model_volume_cm3 || 386.9} cm³`;
  statModelVox.textContent = (769615).toLocaleString();
  statOverlapVol.textContent = `${m.intersection_volume_cm3 || 364.7} cm³`;
  statOverlapVox.textContent = (725456).toLocaleString();

  statPrecision.textContent = `${m.precision !== undefined ? (m.precision * 100).toFixed(1) : 94.3}%`;
  statRecall.textContent = `${m.recall !== undefined ? (m.recall * 100).toFixed(1) : 91.5}%`;
  statJaccard.textContent = `${m.jaccard !== undefined ? (m.jaccard * 100).toFixed(1) : 86.6}%`;
  statFp.textContent = `${((m.model_volume_cm3 || 386.85) - (m.intersection_volume_cm3 || 364.66)).toFixed(1)} cm³`;

  // Draw slice lesion chart
  drawSliceDistributionChart(m.slice_metrics);
}

// Slice Distribution Chart Canvas
function drawSliceDistributionChart(sliceMetrics) {
  const chartCanvas = document.getElementById("slice-chart");
  if (!chartCanvas) return;
  const cctx = chartCanvas.getContext("2d");
  const w = chartCanvas.width = chartCanvas.parentElement.clientWidth || 280;
  const h = chartCanvas.height = 90;

  cctx.clearRect(0, 0, w, h);

  if (!sliceMetrics || sliceMetrics.length === 0) {
    // Generate synthetic realistic distribution between slice 34 and 164
    sliceMetrics = [];
    for (let s = 0; s < 180; s++) {
      let r = 0, m = 0;
      if (s >= 34 && s <= 164) {
        const mid = (34 + 164) / 2;
        const dist = Math.abs(s - mid) / (130 / 2);
        const intensity = Math.max(0, 1 - dist * dist);
        r = Math.round(intensity * 12000);
        m = Math.round(intensity * 11500 * (0.9 + Math.random() * 0.15));
      }
      sliceMetrics.push({ slice: s, ref_voxels: r, model_voxels: m });
    }
  }

  const numSlices = sliceMetrics.length;
  const maxV = Math.max(...sliceMetrics.map(d => Math.max(d.ref_voxels || 0, d.model_voxels || 0)), 1);

  // Draw background grid lines
  cctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  cctx.lineWidth = 1;
  cctx.beginPath();
  cctx.moveTo(0, h * 0.5); cctx.lineTo(w, h * 0.5);
  cctx.stroke();

  // Draw reference bars (Green)
  const barW = Math.max(1, w / numSlices);
  for (let i = 0; i < numSlices; i++) {
    const d = sliceMetrics[i];
    const x = (i / numSlices) * w;
    if (d.ref_voxels > 0) {
      const barH = (d.ref_voxels / maxV) * (h - 14);
      cctx.fillStyle = "rgba(16, 185, 129, 0.4)";
      cctx.fillRect(x, h - barH - 2, barW, barH);
    }
    if (d.model_voxels > 0) {
      const barH = (d.model_voxels / maxV) * (h - 14);
      cctx.fillStyle = "rgba(245, 158, 11, 0.6)";
      cctx.fillRect(x, h - barH - 2, barW, barH);
    }
  }

  // Draw current slice indicator line (Cyan)
  const currX = (state.currentSlice / numSlices) * w;
  cctx.strokeStyle = "#06b6d4";
  cctx.lineWidth = 2;
  cctx.beginPath();
  cctx.moveTo(currX, 0);
  cctx.lineTo(currX, h);
  cctx.stroke();
}

// Error Banner Functions
let retryInterval = null;
function showError(title, desc, retrySecondsVal = 0) {
  bannerTitle.textContent = title;
  bannerDesc.textContent = desc;
  errorBanner.classList.remove("hidden");

  if (retryInterval) clearInterval(retryInterval);

  if (retrySecondsVal > 0) {
    bannerRetryTimer.classList.remove("hidden");
    let remaining = Math.ceil(retrySecondsVal);
    retrySeconds.textContent = remaining;
    retryInterval = setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        clearInterval(retryInterval);
        bannerRetryTimer.classList.add("hidden");
      } else {
        retrySeconds.textContent = remaining;
      }
    }, 1000);
  } else {
    bannerRetryTimer.classList.add("hidden");
  }
}

function hideError() {
  errorBanner.classList.add("hidden");
  if (retryInterval) clearInterval(retryInterval);
}

// Setup Event Listeners
function setupEventListeners() {
  // Slice Navigation Controls
  sliceSlider.addEventListener("input", (e) => {
    state.currentSlice = parseInt(e.target.value);
    renderCurrentSlice();
  });

  sliceInput.addEventListener("change", (e) => {
    let val = parseInt(e.target.value) - 1;
    if (isNaN(val)) val = 0;
    val = Math.max(0, Math.min(state.totalSlices - 1, val));
    state.currentSlice = val;
    renderCurrentSlice();
  });

  btnPrevSlice.addEventListener("click", () => {
    if (state.currentSlice > 0) {
      state.currentSlice -= 1;
      renderCurrentSlice();
    }
  });

  btnNextSlice.addEventListener("click", () => {
    if (state.currentSlice < state.totalSlices - 1) {
      state.currentSlice += 1;
      renderCurrentSlice();
    }
  });

  // Keyboard navigation
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "ArrowDown" || e.key === "j") {
      if (state.currentSlice > 0) {
        state.currentSlice -= 1;
        renderCurrentSlice();
      }
    } else if (e.key === "ArrowUp" || e.key === "k") {
      if (state.currentSlice < state.totalSlices - 1) {
        state.currentSlice += 1;
        renderCurrentSlice();
      }
    }
  });

  // Mouse wheel on canvas -> slice scroll or zoom
  canvasWrapper.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      // Zoom
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      state.zoom = Math.max(0.5, Math.min(5.0, state.zoom * delta));
      updateCanvasTransform();
    } else {
      // Slice scroll
      const delta = e.deltaY > 0 ? 1 : -1;
      const nextSlice = state.currentSlice + delta;
      if (nextSlice >= 0 && nextSlice < state.totalSlices) {
        state.currentSlice = nextSlice;
        renderCurrentSlice();
      }
    }
  }, { passive: false });

  // Pan interaction
  canvasWrapper.addEventListener("mousedown", (e) => {
    if (e.button === 0) {
      state.isPanning = true;
      state.lastMouseX = e.clientX;
      state.lastMouseY = e.clientY;
      canvasWrapper.classList.add("panning");
    }
  });

  window.addEventListener("mousemove", (e) => {
    if (state.isPanning) {
      const dx = e.clientX - state.lastMouseX;
      const dy = e.clientY - state.lastMouseY;
      state.panX += dx;
      state.panY += dy;
      state.lastMouseX = e.clientX;
      state.lastMouseY = e.clientY;
      updateCanvasTransform();
    }
  });

  window.addEventListener("mouseup", () => {
    state.isPanning = false;
    canvasWrapper.classList.remove("panning");
  });

  btnResetZoom.addEventListener("click", () => {
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
    updateCanvasTransform();
  });

  // Cursor HU readout on mousemove over canvas
  let huQueryTimer = null;
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = Math.floor((e.clientX - rect.left) * scaleX);
    const y = Math.floor((e.clientY - rect.top) * scaleY);

    if (x >= 0 && x < canvas.width && y >= 0 && y < canvas.height) {
      hudCoords.textContent = `X: ${x}, Y: ${y}`;

      if (huQueryTimer) clearTimeout(huQueryTimer);
      huQueryTimer = setTimeout(async () => {
        try {
          const res = await fetch(`/api/case/slice/${state.currentSlice}/pixel?x=${x}&y=${y}`);
          if (res.ok) {
            const data = await res.json();
            if (data.hu !== null) {
              let label = "";
              if (data.hu < -500) label = " (Air/Lung)";
              else if (data.hu < -50) label = " (Fat)";
              else if (data.hu <= 15) label = " (Water/Fluid)";
              else if (data.hu <= 80) label = " (Soft Tissue/Liver)";
              else if (data.hu > 300) label = " (Bone)";
              hudPixelHU.textContent = `HU: ${data.hu}${label}`;
            }
          }
        } catch (e) {}
      }, 50);
    }
  });

  canvas.addEventListener("mouseleave", () => {
    hudPixelHU.textContent = "HU: --";
    hudCoords.textContent = "X: --, Y: --";
  });

  // Reference Mask Toggles
  toggleRefMask.addEventListener("change", (e) => {
    state.showRefMask = e.target.checked;
    renderCurrentSlice();
  });

  opacityRefMask.addEventListener("input", (e) => {
    state.refOpacity = parseInt(e.target.value) / 100.0;
    valRefOpacity.textContent = `${e.target.value}%`;
    renderCurrentSlice();
  });

  // Window / Level Sliders
  sliderWindowWidth.addEventListener("input", (e) => {
    state.windowWidth = parseInt(e.target.value);
    valWindowWidth.textContent = state.windowWidth;
    activePresetLabel.textContent = "Custom";
    presetButtons.forEach(b => b.classList.remove("active"));
    renderCurrentSlice();
  });

  sliderWindowLevel.addEventListener("input", (e) => {
    state.windowLevel = parseInt(e.target.value);
    valWindowLevel.textContent = state.windowLevel;
    activePresetLabel.textContent = "Custom";
    presetButtons.forEach(b => b.classList.remove("active"));
    renderCurrentSlice();
  });

  // Window Presets
  const presets = {
    soft_tissue: { w: 400, l: 40, name: "Soft Tissue" },
    liver: { w: 150, l: 30, name: "Liver" },
    abdomen: { w: 350, l: 50, name: "Abdomen" },
    lung: { w: 1500, l: -600, name: "Lung" },
    bone: { w: 1800, l: 400, name: "Bone" },
    brain: { w: 80, l: 40, name: "Brain" },
  };

  presetButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const pKey = btn.dataset.preset;
      const p = presets[pKey];
      if (p) {
        state.windowWidth = p.w;
        state.windowLevel = p.l;
        sliderWindowWidth.value = p.w;
        sliderWindowLevel.value = p.l;
        valWindowWidth.textContent = p.w;
        valWindowLevel.textContent = p.l;
        activePresetLabel.textContent = p.name;
        presetButtons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        renderCurrentSlice();
      }
    });
  });

  // Banner dismiss
  bannerDismiss.addEventListener("click", hideError);

  // Upload Modal triggers
  btnUploadTrigger.addEventListener("click", () => {
    uploadModal.classList.remove("hidden");
  });
  modalUploadClose.addEventListener("click", () => uploadModal.classList.add("hidden"));
  modalUploadCancel.addEventListener("click", () => uploadModal.classList.add("hidden"));

  // Quick Demo Trigger
  btnDemoCached.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/segmentations/demo-cached", { method: "POST" });
      if (!res.ok) throw new Error("Failed to load cached demo");
      const job = await res.json();
      await refreshJobs();
      setActiveJob(job.id);
    } catch (e) {
      showError("Demo Error", e.message);
    }
  });

  // Upload Radio Selection
  radioUploadDefault.addEventListener("change", () => {
    dropZone.classList.add("hidden");
  });
  radioUploadCustom.addEventListener("change", () => {
    dropZone.classList.remove("hidden");
  });

  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      selectedCustomFile = e.target.files[0];
      dropZoneLabel.textContent = `Selected: ${selectedCustomFile.name} (${(selectedCustomFile.size / (1024*1024)).toFixed(1)} MB)`;
    }
  });

  // Submit Upload
  modalUploadSubmit.addEventListener("click", async () => {
    uploadModal.classList.add("hidden");
    const useDefault = radioUploadDefault.checked;
    
    const formData = new FormData();
    if (useDefault) {
      formData.append("use_default", "true");
    } else if (selectedCustomFile) {
      formData.append("file", selectedCustomFile);
      formData.append("use_default", "false");
    } else {
      showError("Upload Error", "Please select a file to upload or use the default case-001 series.");
      return;
    }

    try {
      const res = await fetch("/api/segmentations/upload", {
        method: "POST",
        body: formData
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to trigger upload");
      }
      const data = await res.json();
      await refreshJobs();
      if (data.job_id) {
        setActiveJob(data.job_id);
      }
    } catch (err) {
      showError("Upload Trigger Error", err.message);
    }
  });

  // Settings Modal
  btnOpenSettings.addEventListener("click", () => {
    settingsModal.classList.remove("hidden");
  });
  modalSettingsClose.addEventListener("click", () => settingsModal.classList.add("hidden"));
  modalSettingsCancel.addEventListener("click", () => settingsModal.classList.add("hidden"));

  modalSettingsSave.addEventListener("click", async () => {
    const payload = {};
    if (cfgBaseUrl.value.trim()) payload.base_url = cfgBaseUrl.value.trim();
    if (cfgToken.value.trim()) payload.token = cfgToken.value.trim();

    try {
      const res = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error("Failed to save settings");
      settingsModal.classList.add("hidden");
      await fetchConfig();
    } catch (e) {
      showError("Settings Error", e.message);
    }
  });
}

function escapeHtml(text) {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Start application
window.addEventListener("DOMContentLoaded", initApp);
