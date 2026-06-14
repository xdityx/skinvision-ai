# Phase 0 Feasibility Report — ACNE04 Dataset

**Generated:** 2026-06-08 18:49 UTC   **Seed:** 42


## 1. Executive Summary

**Overall verdict:** 🟡 **MARGINAL**


**Decision Gates:**


| Gate | Status |
|------|--------|
| Sufficient images in all training classes | ✅ |
| Acceptable class imbalance ratio | ✅ |
| Low corrupted image percentage | ✅ |
| Low face identity leakage risk | ❌ |
| Split verification passed | ✅ |

**Total images:** 1,406 — **Quality pass:** 1,279 (91.0%)

**Top recommendations:**

1. CRITICAL — Leakage Risk Index is 72.5% (> 15%). A large fraction of images may belong to the same individual. Consider curating a dataset with verified unique subjects before proceeding to Phase 1.
2. Test split is frozen — do NOT use for hyperparameter tuning, architecture search, or threshold calibration. Reserve exclusively for final model evaluation.
3. For RTX 3050 4 GB VRAM: Use EfficientNet-B2 (imagenet pretrained) with mixed precision (torch.cuda.amp). Batch size 32, gradient accumulation 2. Expected VRAM peak ≈ 3.6 GB — leaves headroom for augmentation on-GPU. If OOM: switch to EfficientNet-B1 or reduce crop to 224×224.


## 2. Dataset Inventory

| Metric | Value |
|--------|-------|
| format_detected | metadata_jsonl_acne04 |
| total_records | 1406 |
| duplicate_count | 36 |
| images_not_found | 0 |
| timestamp | 2026-06-08T18:47:57.557331+00:00 |


## 3. Note on `sim_acne.csv`

`sim_acne.csv` has been **excluded** from all training artefacts. Investigation confirmed it is a single-subject (N-of-1) longitudinal trial artefact containing repeated photos of one individual across multiple severity grades. Including it would introduce severe within-subject leakage and artificially inflate cross-class identity overlap.


## 4. Class Distribution (Quality-Pass Images)

| Class | Severity | Count | % of QP Total |
|-------|----------|-------|---------------|
| 0 | Mild | 469 | 36.7% |
| 1 | Moderate | 590 | 46.1% |
| 2 | Severe | 220 | 17.2% |

**Imbalance Ratio (max/min):** 2.68

**Suggested class weights:** `{'Mild': 1.258, 'Moderate': 1.0, 'Severe': 2.682}`


## 5. Image Quality Analysis

| Face Flag | Count |
|-----------|-------|
| `ok` | 1,187 |
| `low_confidence` | 92 |
| `no_face` | 84 |
| `multi_face` | 43 |

**Blur score (Laplacian variance):** mean=253.58, std=194.72, min=1.52, max=1387.92

**Mean pixel intensity:** mean=98.17, std=16.61

**Corrupted / filtered images:** 127 (9.03% of total) ✅


## 6. Face Clustering & Leakage Analysis

| Metric | Value |
|--------|-------|
| InsightFace model | `buffalo_sc` |
| DBSCAN best ε | 0.3 |
| Best silhouette score | 0.5431 |
| Non-noise clusters | 231 |
| Images in multi-image clusters | 861 |
| Leakage Risk Index (LRI) | 72.54% |
| Leakage verdict | ❌ HIGH_RISK |


### DBSCAN Hyperparameter Sweep

| ε | Clusters | Noise | Silhouette |
|---|----------|-------|------------|
| 0.3 ★ | 231 | 326 | 0.5431 |
| 0.35 | 269 | 197 | 0.5266 |
| 0.4 | 283 | 143 | 0.5131 |
| 0.45 | 286 | 112 | 0.4704 |
| 0.5 | 244 | 105 | 0.3688 |


## 7. Train / Val / Test Split Summary

| Split | Total | Clusters | Mild | Moderate | Severe |
| --- | --- | --- | --- | --- | --- |
| Train | 1026 | 162 | 364 | 462 | 200 |
| Val | 181 | 35 | 60 | 80 | 41 |
| Test | 199 | 34 | 67 | 81 | 51 |

**Split verification:** ✅

  - No Cluster In Multiple Splits: ✅

  - All Images Assigned: ✅

  - No Duplicate Across Splits: ✅

  - Class Distribution Within Tolerance: ✅


## 8. Phase 1 Recommendations

1. CRITICAL — Leakage Risk Index is 72.5% (> 15%). A large fraction of images may belong to the same individual. Consider curating a dataset with verified unique subjects before proceeding to Phase 1.

2. Test split is frozen — do NOT use for hyperparameter tuning, architecture search, or threshold calibration. Reserve exclusively for final model evaluation.

3. For RTX 3050 4 GB VRAM: Use EfficientNet-B2 (imagenet pretrained) with mixed precision (torch.cuda.amp). Batch size 32, gradient accumulation 2. Expected VRAM peak ≈ 3.6 GB — leaves headroom for augmentation on-GPU. If OOM: switch to EfficientNet-B1 or reduce crop to 224×224.


---
*End of Phase 0 Feasibility Report*
