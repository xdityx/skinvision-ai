/**
 * app.js — SkinVision AI frontend logic
 *
 * State machine:
 *   idle       → image selected → ready
 *   ready      → Analyze clicked → loading
 *   loading    → API success → result
 *   loading    → API error   → error (back to ready)
 *   result     → new image   → ready
 */

'use strict';

const API_BASE = 'http://localhost:8000';
const PREDICT_URL = `${API_BASE}/api/v1/predict`;
const READY_URL   = `${API_BASE}/ready`;

// ── DOM refs ──────────────────────────────────────────────────────
const fileInput       = document.getElementById('file-input');
const dropZone        = document.getElementById('drop-zone');
const dropZoneContent = document.getElementById('drop-zone-content');
const previewWrapper  = document.getElementById('preview-wrapper');
const previewImg      = document.getElementById('preview-img');
const previewClear    = document.getElementById('preview-clear');
const ttaCheckbox     = document.getElementById('tta-checkbox');
const btnAnalyze      = document.getElementById('btn-analyze');

const errorBanner     = document.getElementById('error-banner');
const errorCode       = document.getElementById('error-code');
const errorMessage    = document.getElementById('error-message');
const faceWarningBanner  = document.getElementById('face-warning-banner');
const faceWarningMessage = document.getElementById('face-warning-message');

const emptyState      = document.getElementById('empty-state');
const skeletonState   = document.getElementById('skeleton-state');
const resultContent   = document.getElementById('result-content');

const severityBadge   = document.getElementById('severity-badge');
const severityDot     = document.getElementById('severity-dot');   // inside badge via class
const severityLabel   = document.getElementById('severity-label');
const confidenceValue = document.getElementById('confidence-value');

const fillMild        = document.getElementById('fill-mild');
const fillModerate    = document.getElementById('fill-moderate');
const fillSevere      = document.getElementById('fill-severe');
const pctMild         = document.getElementById('pct-mild');
const pctModerate     = document.getElementById('pct-moderate');
const pctSevere       = document.getElementById('pct-severe');
const probRowMild     = document.getElementById('prob-mild');
const probRowModerate = document.getElementById('prob-moderate');
const probRowSevere   = document.getElementById('prob-severe');

const metaTime        = document.getElementById('meta-time');
const metaTta         = document.getElementById('meta-tta');
const metaModel       = document.getElementById('meta-model');
const modelVersionBadge = document.getElementById('model-version-badge');

// ── State ─────────────────────────────────────────────────────────
let selectedFile = null;

// ── Init ──────────────────────────────────────────────────────────
(async function init() {
  try {
    const res = await fetch(READY_URL, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const data = await res.json();
      if (data.model_version) {
        modelVersionBadge.textContent = data.model_version;
        modelVersionBadge.title = `Checkpoint epoch ${data.checkpoint_epoch}, val F1 ${data.checkpoint_val_f1?.toFixed(4)}`;
      }
    }
  } catch {
    // API not running — silent fail, user will see error when they submit
  }
})();

// ── File input ────────────────────────────────────────────────────
fileInput.addEventListener('change', () => {
  if (fileInput.files && fileInput.files[0]) {
    setFile(fileInput.files[0]);
  }
});

// ── Drag & drop ───────────────────────────────────────────────────
dropZone.addEventListener('dragenter', (e) => { e.preventDefault(); dropZone.classList.add('dragging'); });
dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('dragging'); });
dropZone.addEventListener('dragleave', (e) => {
  if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('dragging');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragging');
  const file = e.dataTransfer?.files?.[0];
  if (file) setFile(file);
});

// Keyboard accessibility for drop zone
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

// ── Clear preview ─────────────────────────────────────────────────
previewClear.addEventListener('click', (e) => {
  e.stopPropagation();
  resetFile();
});

// ── Analyze ───────────────────────────────────────────────────────
btnAnalyze.addEventListener('click', runPrediction);

// ── File helpers ──────────────────────────────────────────────────

function setFile(file) {
  // Validate type
  const ext = file.name.split('.').pop().toLowerCase();
  const allowed = ['jpg', 'jpeg', 'png', 'webp'];
  if (!allowed.includes(ext)) {
    resetFile();
    showError('INVALID_FILE_TYPE', `Unsupported format ".${ext}". Please upload a JPG, PNG, or WebP image.`);
    return;
  }
  // Validate size (10 MB client-side guard)
  if (file.size > 10 * 1024 * 1024) {
    resetFile();
    showError('FILE_TOO_LARGE', 'File exceeds 10 MB. Please compress or resize your image.');
    return;
  }

  selectedFile = file;
  hideError();

  // Show preview
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.onload = () => URL.revokeObjectURL(url);

  dropZoneContent.hidden = true;
  previewWrapper.hidden  = false;

  btnAnalyze.disabled = false;

  // Reset results to empty state
  showEmptyState();
}

function resetFile() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src  = '';
  previewWrapper.hidden  = true;
  dropZoneContent.hidden = false;
  btnAnalyze.disabled    = true;
  hideError();
  showEmptyState();
}

// ── Results state machine ─────────────────────────────────────────

function showEmptyState() {
  emptyState.hidden    = false;
  skeletonState.hidden = true;
  resultContent.hidden = true;
  hideFaceWarning();
}

function showSkeleton() {
  emptyState.hidden    = true;
  skeletonState.hidden = false;
  resultContent.hidden = true;
  hideFaceWarning();
}

function showResult(data) {
  emptyState.hidden    = true;
  skeletonState.hidden = true;
  resultContent.hidden = false;

  if (data.face_warning) {
    showFaceWarning(data.face_warning);
  } else {
    hideFaceWarning();
  }

  const sev = data.predicted_severity;   // "mild" | "moderate" | "severe"
  const conf = (data.confidence * 100).toFixed(1);

  // Badge
  severityBadge.className = `severity-badge ${sev}`;
  severityLabel.textContent = sev;

  // Confidence
  confidenceValue.textContent = `${conf}%`;

  // Colour for confidence based on severity
  const colourMap = { mild: '#10b981', moderate: '#f59e0b', severe: '#ef4444' };
  confidenceValue.style.color = colourMap[sev] || 'var(--text)';

  // Probability bars — animate after a tick so CSS transition fires
  const probs = data.class_probabilities;
  setTimeout(() => {
    setBar('mild',     probs.mild);
    setBar('moderate', probs.moderate);
    setBar('severe',   probs.severe);
  }, 60);

  // Active row highlight
  [probRowMild, probRowModerate, probRowSevere].forEach(r => r.classList.remove('active'));
  const activeRow = { mild: probRowMild, moderate: probRowModerate, severe: probRowSevere }[sev];
  if (activeRow) activeRow.classList.add('active');

  // Meta
  metaTime.textContent  = `${data.inference_time_ms.toFixed(0)} ms`;
  metaTta.textContent   = data.tta_enabled ? `Yes (${data.tta_views} views)` : 'No';
  metaModel.textContent = data.model_version;
}

function setBar(cls, prob) {
  const pct = (prob * 100).toFixed(1);
  document.getElementById(`fill-${cls}`).style.width = `${prob * 100}%`;
  document.getElementById(`pct-${cls}`).textContent   = `${pct}%`;
}

// ── Error helpers ─────────────────────────────────────────────────

function showError(code, message) {
  errorCode.textContent    = code.replace(/_/g, ' ');
  errorMessage.textContent = message;
  errorBanner.hidden       = false;
}

function hideError() {
  errorBanner.hidden = true;
}

function showFaceWarning(message) {
  faceWarningMessage.textContent = message;
  faceWarningBanner.hidden = false;
}

function hideFaceWarning() {
  faceWarningBanner.hidden = true;
  faceWarningMessage.textContent = '';
}

// ── Loading state ─────────────────────────────────────────────────

function setLoading(on) {
  btnAnalyze.disabled        = on;
  btnAnalyze.setAttribute('aria-busy', on ? 'true' : 'false');
  if (on) {
    showSkeleton();
    hideError();
  }
}

// ── API call ──────────────────────────────────────────────────────

async function runPrediction() {
  if (!selectedFile) return;

  setLoading(true);

  const formData = new FormData();
  formData.append('file', selectedFile, selectedFile.name);

  const tta = ttaCheckbox.checked;
  const url = `${PREDICT_URL}?tta=${tta}&strict_face=false`;

  console.log(`Sending request to URL: ${url}`);

  try {
    const res = await fetch(url, {
      method: 'POST',
      body:   formData,
    });

    console.log('Response status:', res.status);

    let data = null;
    const contentType = res.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      try {
        data = await res.json();
        console.log('Response JSON:', data);
      } catch (jsonErr) {
        console.error('Failed to parse JSON response:', jsonErr);
      }
    } else {
      try {
        const text = await res.text();
        console.log('Response Text:', text);
      } catch (textErr) {
        console.error('Failed to read text response:', textErr);
      }
    }

    if (!res.ok) {
      // API returned a structured error
      let code = `HTTP_${res.status}`;
      let msg = `Server returned status code ${res.status}.`;

      if (data && data.detail) {
        if (typeof data.detail === 'object' && !Array.isArray(data.detail)) {
          code = data.detail.code || code;
          msg = data.detail.message || msg;
        } else if (Array.isArray(data.detail)) {
          code = 'VALIDATION_ERROR';
          msg = data.detail.map(err => {
            const locStr = err.loc ? err.loc.join('.') : '';
            return locStr ? `${locStr}: ${err.msg}` : err.msg;
          }).join('; ');
        } else if (typeof data.detail === 'string') {
          msg = data.detail;
        }
      }

      showError(code, msg);
      showEmptyState();
    } else {
      showResult(data);
    }

  } catch (err) {
    console.error('Fetch error:', err);
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      showError(
        'API_UNREACHABLE',
        `Could not connect to the API at ${API_BASE}. ` +
        'Make sure the server is running: uvicorn api.main:app --reload --port 8000'
      );
    } else {
      showError('REQUEST_FAILED', err.message || 'An unexpected error occurred.');
    }
    showEmptyState();
  } finally {
    setLoading(false);
    btnAnalyze.disabled = !selectedFile;
  }
}
