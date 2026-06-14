# SkinVision AI — Acne Severity Classifier

An end-to-end deep learning pipeline for ordinal acne severity classification:
**Mild → Moderate → Severe** using EfficientNet-B2 + CORN ordinal loss.

> **Medical Disclaimer:** This tool is for research and educational purposes only.
> It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## Quick Start

### 1. Start the API server

```bash
# From the project root
uvicorn api.main:app --reload --port 8000
```

Expected output:
```
INFO  | Loading AcnePredictor from phase1/checkpoints/best_model.pt ...
INFO  | GPU: NVIDIA GeForce RTX 3050 Laptop GPU | VRAM: 4.3 GB
INFO  | GPU warm-up complete.
INFO  | FaceValidator initialised
INFO  | API ready.
INFO  | Application startup complete.
```

API docs available at: **http://localhost:8000/docs**

---

### 2. Open the frontend

**Option A — Direct file (simplest):**
```
Open frontend/index.html in your browser
```
> Some browsers block `file://` → `http://` requests. If predictions fail,
> use Option B below.

**Option B — Local HTTP server (recommended):**
```bash
# Python built-in server
python -m http.server 3000 --directory frontend

# Then open: http://localhost:3000
```

**Option C — VS Code:**
Install the **Live Server** extension, right-click `frontend/index.html` → *Open with Live Server*.

---

### 3. Upload and analyze an image

1. Drag and drop or click the **upload zone** to select a face photo.
2. Optionally enable **TTA** (5-view test-time augmentation) for more robust results.
3. Click **Analyze Image**.
4. View the predicted severity, confidence, and probability breakdown.

**Supported formats:** JPG, JPEG, PNG, WebP — max 10 MB  
**Requirements:** Clear, front-facing photo with exactly one visible face.

---

## Project Structure

```
FaceRecogintion/
├── api/                        # Phase 2C — FastAPI wrapper
│   ├── main.py                 #   App factory + lifespan (model load + GPU warmup)
│   ├── config.py               #   Settings (env-var overrides with ACNE_ prefix)
│   ├── models.py               #   Pydantic v2 response schemas
│   ├── face_validator.py       #   MediaPipe face detection guard
│   ├── dependencies.py         #   FastAPI dependency providers
│   ├── routers/
│   │   ├── health.py           #   GET /health, GET /ready
│   │   └── predict.py          #   POST /api/v1/predict
│   └── tests/                  #   31 API tests
│
├── frontend/                   # Phase 2D — Demo frontend
│   ├── index.html              #   Single-page app
│   ├── style.css               #   Dark glassmorphism design
│   └── app.js                  #   Fetch API + state machine
│
├── phase0/                     # Phase 0 — Data pipeline
│   ├── config/phase0.yaml      #   Dataset config
│   └── src/                    #   Ingestion, EDA, quality audit, splits
│
├── phase1/                     # Phase 1 — Model training
│   ├── config/phase1.yaml      #   Training config (tuned for RTX 3050 4GB)
│   ├── checkpoints/
│   │   └── best_model.pt       #   Trained checkpoint (epoch 16, val F1=0.753)
│   ├── logs/                   #   Training curves, confusion matrix, metrics
│   ├── src/
│   │   ├── model.py            #   EfficientNet-B2 + CORN ordinal head
│   │   ├── dataset.py          #   AcneDataset (3-class: mild/moderate/severe)
│   │   ├── transforms.py       #   Albumentations pipelines
│   │   ├── trainer.py          #   Training loop (AMP, cosine LR, early stop)
│   │   ├── metrics.py          #   Macro-F1, QWK, confusion matrix
│   │   └── inference.py        #   AcnePredictor + 5-view TTA
│   └── scripts/
│       ├── train.py            #   Training entry point
│       └── predict.py          #   CLI prediction tool
│
├── data/
│   └── raw/ACNE04/             #   Dataset (not committed)
└── pyproject.toml
```

---

## API Reference

### `GET /health`
Liveness probe — always returns 200 if the server is running.

```json
{ "status": "ok" }
```

### `GET /ready`
Readiness probe — returns model metadata when loaded.

```json
{
  "ready": true,
  "model_loaded": true,
  "checkpoint_epoch": 16,
  "checkpoint_val_f1": 0.7529,
  "model_version": "acne-classifier-v1.0"
}
```

### `POST /api/v1/predict`
**Request:** `multipart/form-data`
- `file` — image file (JPG/PNG/WebP, ≤ 10 MB)
- `tta` — query param `?tta=true` to enable 5-view TTA

**Response:**
```json
{
  "predicted_class": 0,
  "predicted_severity": "mild",
  "confidence": 0.685547,
  "class_probabilities": {
    "mild": 0.685547,
    "moderate": 0.314209,
    "severe": 0.000105
  },
  "tta_enabled": false,
  "tta_views": null,
  "inference_time_ms": 42.3,
  "model_version": "acne-classifier-v1.0"
}
```

**Error codes:**
| Code | Status | Cause |
|---|---|---|
| `NO_FACE_DETECTED` | 400 | No face found in the image |
| `MULTIPLE_FACES` | 400 | More than one face found |
| `INVALID_FILE_TYPE` | 400 | Unsupported extension / content-type |
| `FILE_TOO_LARGE` | 413 | Image exceeds 10 MB |
| `INFERENCE_ERROR` | 500 | Unexpected model failure |

---

## Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | 75.4% |
| Macro F1 | 75.9% |
| **QWK** | **0.779** (substantial agreement — clinical grade) |
| Mild accuracy | 85.1% |
| Moderate accuracy | 67.9% |
| Severe accuracy | 74.5% |

Trained on ACNE04 dataset (1,406 images, 3 classes).  
Backbone: EfficientNet-B2 (7.7M params, ImageNet pretrained).  
Training: ~14 minutes on NVIDIA RTX 3050 4GB (early stop at epoch 26, best epoch 16).

---

## Environment Setup

```bash
# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install timm albumentations fastapi uvicorn python-multipart \
            pydantic-settings mediapipe pillow opencv-python tqdm
```

---

## Run Tests

```bash
# Phase 1 model tests (28 tests)
python -m pytest phase1/tests/ -v

# API tests (31 tests)
python -m pytest api/tests/ -v

# All tests
python -m pytest -v
```

---

## Configuration

All API settings can be overridden with environment variables (prefix: `ACNE_`):

| Variable | Default | Description |
|---|---|---|
| `ACNE_CHECKPOINT_PATH` | `phase1/checkpoints/best_model.pt` | Path to model checkpoint |
| `ACNE_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `ACNE_MAX_FILE_SIZE_MB` | `10` | Upload size limit |
| `ACNE_FACE_DETECTION_CONFIDENCE` | `0.5` | MediaPipe min confidence |
| `ACNE_MODEL_VERSION` | `acne-classifier-v1.0` | Version string in API response |
