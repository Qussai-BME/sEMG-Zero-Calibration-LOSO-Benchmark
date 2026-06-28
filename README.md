# sEMG Zero-Calibration LOSO Benchmark

**Hand-Crafted Features Outperform CNN-1D Under Zero-Calibration Cross-Subject sEMG: A Three-Database LOSO Benchmark Across Intact-Limb and Transradial Amputee Populations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)](#)

**Maintainer:** Qussai Adlbi
**Affiliations:** Al-Andalus University for Medical Sciences (Syria) · Pázmány Péter Catholic University (Budapest, Hungary)
**Contact:** qussai.adlbi@au.edu.sy

---

## Overview

Most sEMG gesture-recognition studies report **within-subject** accuracy — a number that measures memorisation, not generalisation to a new, uncalibrated patient. This repository contains the complete pipeline, ablations, and pre-computed results for the first **simultaneous Leave-One-Subject-Out (LOSO)** benchmark of a 420-dimensional hand-crafted feature set across three NinaPro databases of escalating clinical difficulty:

| Database | Population | Subjects | Classes |
|---|---|---|---|
| **DB2** | Intact-limb | 40 | 41 (40 gestures + Rest) |
| **DB3** | Transradial amputees | 11 | 41 harmonised (17 active classes present) |
| **DB7** | Mixed intact + amputee | 22 | 41 (40 gestures + Rest) |

Four classical classifiers (XGBoost, LDA, LinearSVC, Random Forest) and a naive end-to-end **CNN-1D** baseline are compared, CPU-only, with **zero subject-specific calibration**.

**Why this matters:** the gap between "accuracy" and "macro-F1" reported here exposes a reporting blind spot common across the field — overall accuracy can be driven almost entirely by the Rest class, hiding near-total failure on active gestures. This repository documents that effect explicitly, with active-only confusion matrices, for the first time on a transradial amputee cohort.

---

## Key Results

**Table — LOSO accuracy ± SD (macro-F1), all classifiers, all databases**

| Classifier | DB7 (n=22) | DB3 (n=11) | DB2 (n=40) |
|---|---|---|---|
| **XGBoost** ★ | **65.96 ± 6.01%** (F1=27.14%) | **43.46 ± 10.73%** (F1=4.01%) | **54.64 ± 7.84%** (F1=21.89%) |
| LDA | 65.55 ± 5.78% (F1=28.92%) | 32.69 ± 17.52% (F1=4.81%) | 53.00 ± 7.26% (F1=21.77%) |
| LinearSVC | 65.50 ± 5.86% (F1=25.57%) | 40.34 ± 13.97% (F1=4.56%) | 53.97 ± 7.94% (F1=19.48%) |
| RandomForest | 65.26 ± 5.62% (F1=24.66%) | 43.64 ± 10.54% (F1=3.72%) | 53.64 ± 8.08% (F1=19.39%) |
| CNN-1D (naive, zero-calibration) | 21.60 ± 6.83% | 4.69 ± 1.64% | 15.58 ± 4.24% |
| Chance level | 2.44% | 2.44% | 2.44% |

**Headline findings:**
- On **DB7**, all four classical classifiers are statistically indistinguishable (Friedman χ²=4.91, p=0.179; max spread = 0.70 pp) — the **420D feature representation, not the classifier**, is the binding constraint. This pattern does **not** generalise to DB2/DB3, where classifier choice matters significantly (Friedman p<0.001 for both).
- Hand-crafted features beat the naive end-to-end **CNN-1D** baseline by **38–44 percentage points** under identical zero-calibration LOSO conditions. This is a **regime-specific** finding (naive, from-scratch, no transfer learning) and is **not** a general claim against deep learning — see Discussion §5.2 of the paper.
- **Rest-class dominance:** on DB3, accuracy (43.46%) and macro-F1 (4.01%) diverge by ~39.86 pp — Rest recall is 97.5% while active-gesture accuracy is only 3.6%. This is, to our knowledge, the first explicit quantification of this effect on transradial amputees with active-only confusion matrices.
- SHAP analysis identifies amplitude-distribution (Histogram) features as the dominant cross-subject discriminator in **all three** databases (23.6–26.3%); AR coefficients contribute ~0% everywhere (hypothesis for future ablation — not yet experimentally validated; see §5.5/§5.7 of the paper).

Full statistical tests (Friedman, Nemenyi, Wilcoxon + Holm–Šidák, Cohen's d), the feature-group and window-size ablations, per-subject variability analysis, and CPU timing benchmarks are reported in the paper (Tables 3–14) and reproduced in [`results/`](#repository-structure) below.

> **Scope note:** the CNN-1D comparison is specific to the naive, zero-calibration, from-scratch regime studied here. It does not constitute a general claim against deep learning for sEMG — see Discussion §5.2 of the paper for the full argument.

---

## Repository Structure

```
sEMG-Zero-Calibration-LOSO-Benchmark/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── config/                          # Global paths, dataset configs, hyperparameters
├── data_loaders/                    # NinaPro DB2 / DB3 / DB7 loading + 41-class harmonised label mapping
├── preprocessing_and_features/      # Bandpass (20–450Hz) + 50Hz notch + 400ms windowing
│                                     # + 678D raw feature extraction → SelectKBest (k=420)
├── classifiers/                     # XGBoost / LDA / LinearSVC / RandomForest LOSO runner
├── cnn_baseline/                    # Multi-scale CNN-1D (v8): 3 conv branches, 48,472 params
├── validate_engine/                 # Core LOSO fold management — per-fold leakage-free fitting
├── metrics/                         # Accuracy, macro-F1, confusion matrix computation
├── statistics/                      # Friedman / Iman-Davenport, Nemenyi, Wilcoxon+Holm-Šidák, Cohen's d
├── shap_analysis/                   # SHAP TreeExplainer — group + individual feature importance
├── ablation_studies/                # Feature-group ablation (Table 11) + window-size ablation (Table 12)
├── confusion_matrices/              # Aggregate (Figure 3) and active-only (Figure 4) CM generation
├── per_class_and_timing/            # Per-class F1/Precision/Recall + CPU timing benchmarks (Table 14)
├── final_analysis/                  # Final aggregated tables and cross-database summaries
├── report_generator/                # Automated Markdown / HTML / JSON report generation
│
└── results/
    ├── loso_checkpoints/            # ninapro_db2_checkpoint.json, ninapro_db3_checkpoint.json,
    │                                 # ninapro_db7_checkpoint.json — per-fold raw LOSO outputs
    └── final_outputs/                # Pre-computed final results used directly in the paper:
        ├── confusion_matrix_db{2,3,7}.xlsx / .npy        # Aggregate confusion matrices (Fig. 3)
        ├── confusion_matrix_raw_db{3,7}.npy              # Raw (non-normalised) confusion matrices
        ├── confusion_xgb_db{2,3,7}.pdf                   # Confusion matrix figures
        ├── shap_group_importance_db{2,3,7}.xlsx          # SHAP feature-group importance (Table 9)
        ├── shap_top20_db{2,3,7}.xlsx                      # Top-20 individual SHAP features (Table 10)
        ├── shap_all_features_db{2,3,7}.xlsx              # Full per-feature SHAP values
        ├── shap_group_pie_db{2,3,7}.pdf                   # SHAP group importance pie charts (Fig. 6)
        ├── shap_top20_bar_db{2,3,7}.pdf                   # SHAP top-20 bar charts
        ├── Table1_dataset_characteristics.xlsx           # Table 1
        ├── Table4_literature_comparison.xlsx             # Table 8
        ├── TableS_feature_ablation_db7.xlsx              # Table 11
        ├── TableS_window_ablation_db7.xlsx                # Table 12
        ├── TableS6_fold_details_db{2,3,7}.xlsx           # Per-fold LOSO results
        ├── TableS6_perclass_f1_db{2,3,7}.xlsx            # Per-class F1 (Supplementary S2d–f)
        └── TableS7_real_timing_db{2,3,7}.xlsx            # CPU timing benchmarks (Table 14, Supp. S3)
```

> Folder names above reflect a cleaned, paper-aligned reorganisation of the original development codebase. Every file under `results/final_outputs/` corresponds directly to a numbered Table or Figure in the manuscript — intermediate development/debugging artifacts were intentionally excluded to keep this repository a faithful, citable record of what is reported in the paper.

---

## Pipeline (5 Phases)

```
Phase 1 — Data Acquisition
    NinaPro DB2 / DB3 / DB7 (12-channel Delsys Trigno, 2000 Hz)
            │
Phase 2 — Preprocessing
    Bandpass (20–450 Hz, 4th-order zero-phase Butterworth) → 50 Hz notch
    → Euclidean alignment (train-fold covariance only)
    → 400 ms windows, 50% overlap → fold-level Z-score normalisation
            │
Phase 3 — Feature Extraction (678D raw → 420D)
    Time-domain (372D) + Histogram (120D) + Hjorth (36D)
    + Frequency-domain (84D) + Inter-channel correlation (66D)
    → SelectKBest (ANOVA F-test, k=420, fitted on training folds only)
            │
Phase 4 — Classification
    XGBoost (300 rounds, depth=6) · LDA (SVD) · LinearSVC (C=1.0)
    · RandomForest (100 trees) · CNN-1D baseline (48,472 params, raw waveform input)
            │
Phase 5 — Evaluation
    Accuracy + macro-F1 → Friedman/Iman-Davenport → Nemenyi post-hoc
    → Wilcoxon + Holm–Šidák → Cohen's d → SHAP (TreeExplainer)
```

**Data leakage prevention:** all preprocessing statistics, feature-selection scores, and classifier hyperparameters are fitted **exclusively on training-fold subjects** within each LOSO fold. The held-out test subject is never seen during normalisation, feature selection, or training.

---

## Installation & Quick Start

```bash
git clone https://github.com/USERNAME/sEMG-Zero-Calibration-LOSO-Benchmark.git
cd sEMG-Zero-Calibration-LOSO-Benchmark

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Run the LOSO pipeline
```bash
python -m classifiers.main --db db2
python -m classifiers.main --db db3
python -m classifiers.main --db db7
```

### Use pre-computed results directly (no re-run needed)
Everything reported in the paper is already available under [`results/final_outputs/`](#repository-structure) — confusion matrices, SHAP values, ablation tables, and per-fold timing benchmarks.

---

## Data Availability

The **NinaPro** databases are publicly available but **not redistributed** in this repository due to licensing:

- NinaPro DB2 / DB3 / DB7: http://ninaweb.hevs.ch

Download the raw `.mat` files and place them under `data/raw/`; `data_loaders/` handles parsing and the 41-class harmonisation.

```bibtex
@article{atzori2014electromyography,
  title={Electromyography data for non-invasive naturally-controlled robotic hand prostheses},
  author={Atzori, Manfredo and Gijsberts, Arjan and Castellini, Claudio and others},
  journal={Scientific Data}, volume={1}, pages={140053}, year={2014},
  doi={10.1038/sdata.2014.53}
}
```

**Code and pre-computed results:** archived in full in this repository. A versioned, DOI-citable release will be published via Zenodo following the manuscript's first public GitHub release; this section will be updated with that DOI once available.

---

## Environment

| Component | Version |
|---|---|
| Python | 3.10 |
| scikit-learn | 1.3 |
| XGBoost | 1.7 |
| PyTorch | 2.0 |
| SHAP | 0.42 |
| SciPy | 1.10 |
| scikit-posthocs | 0.7 |

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
xgboost>=1.7.0
shap>=0.42.0
scikit-posthocs>=0.7.0
torch>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
h5py>=3.9.0
```

---

## Limitations (condensed — see paper §5.6 for the complete list)

- DB3 amputee cohort is small (n=11); population-level claims require larger registries.
- Evaluation is offline; real-time electrode shift, fatigue, and donning variability are untested.
- CNN-1D received no transfer learning or domain adaptation — this is a zero-calibration comparison, not a ceiling for deep learning on sEMG.
- 95% CIs use the t-distribution despite confirmed non-normality (Shapiro–Wilk p<0.05) — treat as approximate.
- AR-coefficient removal is a SHAP-based hypothesis, not yet validated by a dedicated ablation.

---

## Citation

If you use this code or these results, please cite the paper:

```bibtex
@article{adlbi2026handcrafted,
  title   = {Hand-Crafted Features Outperform CNN-1D Under Zero-Calibration Cross-Subject sEMG:
             A Three-Database LOSO Benchmark Across Intact-Limb and Transradial Amputee Populations},
  author  = {Adlbi, Qussai and Darwich, Mohamad Ayham},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Under review}
}
```

> Note: the citation above lists both authors of the manuscript as published — that is a factual record of authorship and is kept complete regardless of repository maintainer credit below.

**Repository maintainer & code author:** Qussai Adlbi — qussai.adlbi@au.edu.sy

---

## License

MIT — see [LICENSE](LICENSE). The NinaPro databases are subject to their own terms of use; please cite Atzori et al. (2014) when using the data.

---

## What this is not

This is a research benchmark, not a clinical or regulatory product. It is not FDA-approved or CE-marked, not clinically validated on patient populations, and not suitable for diagnostic or treatment decisions. Any clinical application would require IRB-approved trials and regulatory review.
