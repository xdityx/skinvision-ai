# SkinVision AI - Acne Severity Analysis + Ingredient Guidance

SkinVision AI is an end-to-end computer vision MVP that analyzes a face or acne-region image, predicts acne severity as **mild**, **moderate**, or **severe**, and returns educational ingredient guidance based on that severity.

The project covers dataset auditing, leakage-aware splitting, ordinal model training, CLI inference, a FastAPI backend, a plain HTML/CSS/JS frontend demo, and a rule-based acne ingredient recommendation engine. It is designed as a research and portfolio project, not as a clinical product.

> **Medical disclaimer:** This is not a medical diagnosis, prescription, or substitute for professional medical advice. Ingredient guidance is educational only. Severe, painful, worsening, or scarring acne should be evaluated by a qualified dermatologist.

---

## Current Status

MVP acne severity detection and Phase 3 ingredient guidance are complete and working.

| Phase | Status | Output |
|---|---:|---|
| Phase 0 | Complete | ACNE04 dataset audit, quality checks, and leakage-aware splits |
| Phase 1 | Complete | EfficientNet-B2 model trained with CORN ordinal loss |
| Phase 2B | Complete | CLI inference workflow |
| Phase 2C | Complete | FastAPI prediction API |
| Phase 2D | Complete | Browser frontend demo |
| Phase 3 | Complete | Rule-based acne ingredient guidance engine |
| Cleanup | Complete | Dataset and model artifacts excluded from Git history/tracking |

API test status: **46 passed**.

---

## Problem Statement

Acne severity is naturally ordinal: severe acne is not just a different category from mild acne, it is a higher level of severity. A standard multi-class classifier can ignore this ordering. This project uses an ordinal learning approach so the model can learn the progression from mild to moderate to severe.

The dataset also contains many partial-face crops, such as cheek, forehead, side-face, and lower-face images. The API therefore supports relaxed face validation by default so valid acne-region crops are not rejected just because a full face is not detected.

After severity prediction, Phase 3 adds a rule-based guidance layer that maps the predicted severity to acne-related ingredients, routine guidance, cautions, and dermatologist escalation messaging. It does not recommend product names and does not scrape ecommerce sites.

---

## Key Features

- Acne severity classification: `mild`, `moderate`, `severe`
- Confidence score and class probabilities
- EfficientNet-B2 backbone with an ordinal CORN head
- Optional 5-view test-time augmentation (TTA)
- Configurable face validation:
  - `strict_face=false` by default
  - partial-face ACNE04 crops are allowed with a warning
  - `strict_face=true` rejects no-face and multiple-face inputs
- FastAPI backend with readiness and prediction endpoints
- Plain HTML/CSS/JS frontend demo
- Rule-based ingredient guidance from a versioned YAML knowledge base
- Routine guidance by severity
- Cautions for irritation, sun sensitivity, pregnancy risk, and over-layering actives
- Dermatologist escalation messaging, especially for severe acne
- Dataset and model artifact safety:
  - raw dataset files are not committed
  - model weights are not committed
  - large artifacts are ignored

---

## Prediction Flow

```text
Image upload
  -> upload validation
  -> configurable face validation
  -> acne severity prediction: mild / moderate / severe
  -> confidence + class probabilities
  -> face validation warning if needed
  -> ingredient guidance based on severity
  -> routine guidance, cautions, and dermatologist escalation
  -> JSON response
  -> frontend result view
```

---

## Architecture

```text
ACNE04 dataset (local only)
        |
        v
Phase 0: audit, EDA, quality checks, cluster-aware split generation
        |
        v
Phase 1: EfficientNet-B2 + CORN ordinal training
        |
        v
Phase 2B: CLI inference
        |
        v
Phase 2C: FastAPI API
        |
        v
Phase 3: rule-based recommendation engine + YAML knowledge base
        |
        v
Phase 2D/3 UI: HTML/CSS/JS frontend demo with ingredient guidance
```

Runtime components:

```text
FastAPI /api/v1/predict
  -> AcnePredictor inference
  -> prediction response model
  -> recommendation engine
  -> api/recommendations/kb/acne_ingredients.v1.yaml
  -> ingredient_guidance response block
```

---

## Project Structure

```text
FaceRecogintion/
  api/                                  FastAPI backend
    main.py                             App factory and lifespan
    config.py                           API settings
    face_validator.py                   MediaPipe face detection wrapper
    models.py                           Pydantic response schemas
    routers/
      health.py                         GET /health, GET /ready
      predict.py                        POST /api/v1/predict
    recommendations/
      engine.py                         Rule-based guidance builder
      loader.py                         YAML loading and validation
      schemas.py                        Pydantic guidance schemas
      kb/
        acne_ingredients.v1.yaml        Versioned ingredient knowledge base
    tests/                              API and recommendation tests

  frontend/                             Plain HTML/CSS/JS demo
    index.html
    style.css
    app.js

  phase0/                               Dataset audit and split pipeline
    config/phase0.yaml
    src/
    tests/

  phase1/                               Model training and inference code
    config/phase1.yaml
    scripts/
    src/
      model.py                          EfficientNet-B2 + CORN model
      dataset.py
      transforms.py
      trainer.py
      metrics.py
      inference.py                      AcnePredictor and TTA inference
    tests/

  reports/                              Final reports and lightweight docs
  pyproject.toml
  README.md
```

Local-only paths such as `data/raw/`, `phase1/checkpoints/`, and `phase1/logs/` are intentionally ignored.

---

## Dataset Handling

Dataset: **ACNE04 from Kaggle**

The dataset is not committed to this repository. To run training or reproduce local inference, place the dataset locally under:

```text
data/raw/ACNE04/
```

The repository is configured to ignore raw data and generated artifacts:

```text
acne_1024/
sim_acne.csv
data/raw/
data/phase0_outputs/
phase1/checkpoints/
phase1/logs/
*.pt
*.pth
*.ckpt
*.onnx
*.npy
```

This keeps the public repository lightweight and avoids redistributing dataset files or model weights.

---

## Leakage-Aware Splitting

Medical and face-image datasets can contain near-duplicates, repeated subjects, or visually similar crops. If similar images land in both train and test sets, performance can look better than it really is.

Phase 0 includes clustering and split-generation utilities to reduce leakage risk. The split strategy groups visually related images and keeps those clusters from being spread across train, validation, and test splits where possible. This makes evaluation more conservative than a naive random split.

This does not prove the model generalizes clinically; it simply reduces one common source of evaluation leakage.

---

## Model Details

| Item | Value |
|---|---|
| Backbone | EfficientNet-B2 |
| Pretraining | ImageNet |
| Objective | CORN ordinal loss |
| Classes | `mild`, `moderate`, `severe` |
| Inference engine | `phase1/src/inference.py` |
| Optional inference mode | 5-view TTA |
| Backend serving | FastAPI |
| Frontend | Plain HTML/CSS/JS |
| Recommendation engine | Rule-based YAML knowledge base |
| Knowledge base | `api/recommendations/kb/acne_ingredients.v1.yaml` |

CORN is used because acne severity has an ordered label structure. The model predicts severity while preserving the mild-to-moderate-to-severe relationship more directly than a flat multi-class objective.

---

## Test Metrics

| Metric | Score |
|---|---:|
| Accuracy | 75.4% |
| Macro F1 | 75.9% |
| Quadratic Weighted Kappa (QWK) | 0.779 |
| Mild accuracy | 85.1% |
| Moderate accuracy | 67.9% |
| Severe accuracy | 74.5% |

The moderate class remains the hardest class, which is expected in an ordinal severity task where boundary cases can be visually ambiguous.

---

## Phase 3: Ingredient Guidance

Phase 3 adds a deterministic, rule-based guidance layer after model inference. The acne model still only predicts severity; the recommendation engine maps that severity to educational ingredient guidance.

The engine returns:

- recommended ingredients
- why each ingredient may be useful
- ingredient-specific cautions
- routine guidance
- general cautions
- dermatologist escalation messaging
- low-confidence warning when model confidence is below `0.60`

Design constraints:

- No product recommendations
- No ecommerce scraping
- No user accounts
- No model retraining
- No changes to CORN/model inference logic
- Educational guidance only

Severe acne guidance intentionally emphasizes dermatologist evaluation because severe acne can be painful, may scar, and may require prescription treatment.

---

## API Endpoints

### `GET /health`

Liveness check.

Example response:

```json
{
  "status": "ok"
}
```

### `GET /ready`

Readiness check with model metadata.

Example response:

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

Predict acne severity for one uploaded image and return severity-based ingredient guidance.

Query parameters:

| Parameter | Default | Description |
|---|---:|---|
| `tta` | `false` | Enables 5-view test-time augmentation |
| `strict_face` | `false` | Requires exactly one full face when `true` |

Form data:

| Field | Description |
|---|---|
| `file` | JPG, JPEG, PNG, or WebP image, up to 10 MB |

Face validation behavior:

| Mode | No full face detected | Multiple faces detected |
|---|---|---|
| `strict_face=false` | Prediction allowed with warning | Prediction allowed with warning |
| `strict_face=true` | Rejected with `NO_FACE_DETECTED` | Rejected with `MULTIPLE_FACES_DETECTED` |

Example response:

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
  "face_detected": true,
  "face_warning": null,
  "inference_time_ms": 316.12,
  "model_version": "acne-classifier-v1.0",
  "ingredient_guidance": {
    "schema_version": "acne_ingredients.v1",
    "severity": "mild",
    "disclaimer": "Educational ingredient guidance only. This is not a diagnosis, prescription, or substitute for care from a qualified dermatologist.",
    "recommended_ingredients": [
      {
        "id": "gentle_cleanser",
        "name": "Gentle cleanser",
        "category": "routine_foundation",
        "priority": "supportive",
        "why": "Cleanses without scrubbing or stripping the skin barrier, which can help reduce irritation.",
        "cautions": [
          "Avoid abrasive scrubs and harsh cleansing tools that can worsen irritation."
        ]
      },
      {
        "id": "salicylic_acid",
        "name": "Salicylic acid",
        "category": "exfoliating_acid",
        "priority": "core",
        "why": "Helps loosen dead skin cells and oil inside pores, which may support blackhead, whitehead, and clogged-pore care.",
        "cautions": [
          "Can cause dryness, stinging, peeling, or irritation, especially when combined with other exfoliating ingredients.",
          "Introduce gradually and pause if significant irritation occurs."
        ]
      }
    ],
    "routine_guidance": {
      "morning": [
        "Gentle cleanser",
        "Lightweight non-comedogenic moisturizer",
        "Broad-spectrum sunscreen"
      ],
      "evening": [
        "Gentle cleanser",
        "One acne-focused active introduced gradually",
        "Moisturizer if skin feels dry or tight"
      ],
      "general": [
        "Start with one active ingredient at a time.",
        "Avoid picking, popping, scrubbing, or frequent routine changes.",
        "Give a consistent routine several weeks before judging results."
      ]
    },
    "cautions": [
      "Do not combine multiple exfoliating or drying actives at once.",
      "Stop or reduce frequency if irritation becomes significant."
    ],
    "dermatologist_escalation": {
      "level": "watchful",
      "message": "Mild acne may be managed with gentle over-the-counter ingredient guidance, but see a dermatologist if it persists, worsens, becomes painful, or leaves marks."
    },
    "confidence_warning": null
  }
}
```

Common error codes:

| Code | Status | Cause |
|---|---:|---|
| `NO_FACE_DETECTED` | 400 | No face found when `strict_face=true` |
| `MULTIPLE_FACES_DETECTED` | 400 | Multiple faces found when `strict_face=true` |
| `INVALID_FILE_TYPE` | 400 | Unsupported extension or content type |
| `FILE_TOO_LARGE` | 413 | Upload exceeds 10 MB |
| `IMAGE_DECODE_ERROR` | 400 | Uploaded file could not be decoded as an image |
| `IMAGE_INVALID` | 400 | Image could not be processed by the inference engine |
| `INFERENCE_ERROR` | 500 | Unexpected model failure |

---

## Frontend Demo

The frontend is a plain HTML/CSS/JS demo that uploads an image to the FastAPI backend and renders the complete prediction response.

Current UI support:

- image upload with preview
- optional TTA toggle
- severity badge
- confidence score
- class probability bars
- face validation warning banner
- ingredient cards
- caution banner
- dermatologist escalation banner
- routine guidance section
- model metadata

The frontend sends `strict_face=false` by default, so partial ACNE04-style acne crops can still be analyzed. If the API reports a face validation concern, the frontend displays it as a warning banner instead of blocking the result.

---

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations fastapi uvicorn python-multipart pydantic-settings mediapipe pillow opencv-python tqdm pytest pyyaml
```

For CPU-only environments, install the appropriate PyTorch build from the official PyTorch installation instructions, then install the remaining dependencies.

Add local assets that are intentionally not tracked:

```text
data/raw/ACNE04/                       # needed for dataset work
phase1/checkpoints/best_model.pt       # needed for API/CLI inference
```

---

## Run the API

From the project root:

```bash
uvicorn api.main:app --reload --port 8000
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

Health checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

---

## Run the Frontend Demo

Option 1: open the file directly:

```text
frontend/index.html
```

Some browsers restrict `file://` pages from calling `http://localhost:8000`. If that happens, serve the frontend locally:

```bash
python -m http.server 3000 --directory frontend
```

Then open:

```text
http://localhost:3000
```

---

## Sample Curl Request

Relaxed validation, matching the frontend default:

```bash
curl -X POST "http://localhost:8000/api/v1/predict?strict_face=false" ^
  -F "file=@data/raw/ACNE04/acne0_1024/levle0_1.jpg"
```

With TTA enabled:

```bash
curl -X POST "http://localhost:8000/api/v1/predict?strict_face=false&tta=true" ^
  -F "file=@data/raw/ACNE04/acne0_1024/levle0_1.jpg"
```

Strict full-face validation:

```bash
curl -X POST "http://localhost:8000/api/v1/predict?strict_face=true" ^
  -F "file=@data/raw/ACNE04/acne0_1024/levle0_1.jpg"
```

On macOS/Linux shells, replace `^` line continuations with `\`.

---

## Run Tests

API tests:

```bash
python -m pytest api/tests/ -v
```

Current API test result:

```text
46 passed
```

Other test suites:

```bash
python -m pytest phase0/tests/ -v
python -m pytest phase1/tests/ -v
python -m pytest -v
```

---

## Configuration

API settings can be overridden with environment variables using the `ACNE_` prefix.

| Variable | Default | Description |
|---|---|---|
| `ACNE_CHECKPOINT_PATH` | `phase1/checkpoints/best_model.pt` | Local checkpoint path |
| `ACNE_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `ACNE_MAX_FILE_SIZE_MB` | `10` | Upload size limit |
| `ACNE_FACE_DETECTION_CONFIDENCE` | `0.5` | MediaPipe minimum face detection confidence |
| `ACNE_FACE_MODEL_SELECTION` | `1` | MediaPipe face detector model selection |
| `ACNE_MODEL_VERSION` | `acne-classifier-v1.0` | Version string returned by the API |

---

## Limitations

- This is not a medical diagnosis.
- This is not a prescription or treatment plan.
- Ingredient guidance is educational and rule-based, not personalized medical advice.
- The system does not collect medical history, medication use, allergies, pregnancy status, skin sensitivity, age, or prior treatment response.
- The system does not detect allergy risk, pregnancy compatibility, or individual sensitivity from an image.
- Severe, painful, worsening, or scarring acne should be evaluated by a dermatologist.
- The model is not medically approved and is not ready for clinical deployment.
- Performance is measured on the project test split, not on prospective clinical data.
- ACNE04 images can differ from real-world smartphone, lighting, demographic, and skin-tone distributions.
- The moderate class is less reliable than mild and severe based on current test metrics.
- MediaPipe face detection is used only as an input-quality signal; relaxed mode allows partial-face acne crops by design.
- The public repository does not include the dataset or trained checkpoint, so local inference requires adding those files manually.

---

## Roadmap

Completed:

- MVP acne severity detection
- CLI inference
- FastAPI backend
- Plain HTML/CSS/JS frontend demo
- Rule-based ingredient guidance

Next:

- Screenshots and documentation polish
- Hyperpigmentation and dark spot detection
- Multi-concern analysis
- Model card with dataset, evaluation, and ethical considerations
- CI checks for API tests and artifact safety

Later:

- Product recommendation based on ingredients
- Progress tracking
- Calibration and uncertainty improvements
- Batch inference for local research workflows
- Lightweight Docker setup for reproducible API runs

---

## Repository Safety

The GitHub repository has been cleaned after an accidental dataset commit. Current safeguards:

- `acne_1024/` is ignored
- `sim_acne.csv` is ignored
- `data/raw/` is ignored
- `data/phase0_outputs/` is ignored
- `phase1/checkpoints/` and `phase1/logs/` are ignored
- common model artifact extensions are ignored: `.pt`, `.pth`, `.ckpt`, `.onnx`, `.npy`

Raw dataset files and trained model weights should remain local-only unless a separate release process is created with the right permissions and documentation.
