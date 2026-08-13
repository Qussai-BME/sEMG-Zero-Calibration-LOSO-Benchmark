# sEMG Zero-Calibration LOSO Benchmark

**Rest-Class Metric Inflation in Zero-Calibration Cross-Subject sEMG: A Three-Database LOSO Benchmark Across Intact-Limb and Transradial Amputee Populations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)](#)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20982280.svg)](https://doi.org/10.5281/zenodo.20982280)

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

Full statistical tests (Friedman, Nemenyi, Wilcoxon + Holm–Šidák, Cohen's d), the feature-group and window-size ablations, per-subject variability analysis, and CPU timing benchmarks are reported in the paper (Tables 3–14) and reproduced in [`validation/results/`](#repository-structure) below.

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
├── .gitattributes
│
├── src/
│   ├── __init__.py
│   └── core_engine.py                   # EMGConfig / EMGFeatureExtractor — signal-processing config
│                                         #   object and filter design (bandpass + notch) used by
│                                         #   process_engine.py. Imported as `from src.core_engine
│                                         #   import EMGConfig, EMGFeatureExtractor`; run scripts from
│                                         #   inside validation/ as shown below — the auto-detection
│                                         #   logic in each script resolves this import automatically.
│
└── validation/                          # Single importable package — the complete pipeline
    │
    ├── config.yaml                      # All paths, dataset configs, hyperparameters
    ├── data_loaders.py                  # NinaPro DB2 / DB3 / DB7 loading + 41-class harmonised label mapping
    ├── process_engine.py                # Bandpass (20–450Hz) + 50Hz notch + 400ms windowing
    │                                     #   + 678D raw feature extraction → SelectKBest (k=420)
    ├── validate_engine.py                # Core LOSO fold engine — per-fold, leakage-free fit/transform/eval
    ├── metrics.py                        # Accuracy, macro-F1, confusion matrix computation
    ├── checkpoint.py                     # Resume-safe checkpointing for long LOSO runs
    ├── statistical_reporter.py           # Statistical-test helpers (Wilcoxon, bootstrap, ...)
    ├── report_generator.py               # Automated Markdown / HTML / JSON report generation
    ├── shap_analysis.py                  # SHAP TreeExplainer — group + individual feature importance
    ├── cnn_baseline.py                   # Early CNN-1D model definition (the final v8 model used for
    │                                     #   the paper is self-contained in 03_run_cnn1d_baseline.py)
    │
    ├── 01a_run_loso_benchmark.py                    # Main LOSO benchmark: XGBoost/LDA/LinearSVC/RandomForest
    │                                                 #   × DB2/DB3/DB7 → Table 2, Table S1
    ├── 01b_recompute_loso_statistics.py             # Statistical tests from saved results (no re-run needed)
    │                                                 #   → Tables S2–S4
    ├── 01c_correct_friedman_test.py                 # Corrected Friedman p-values → Table S2 (corrected)
    ├── 02a_run_window_size_ablation.py              # Window-size ablation: 7 sizes × 4 classifiers, DB7
    ├── 02b_run_feature_group_ablation.py            # Feature-group ablation: 5 configs × 2 classifiers, DB7
    ├── 02c_analyze_ablation_statistics.py           # Friedman / Nemenyi / Wilcoxon for both ablations above
    ├── 02d_run_full_ablation_suite.py               # Single-command runner for 02a + 02b + 02c
    ├── 02e_generate_posthoc_figures_and_tables.py   # Main-results figure, confusion-matrix figures,
    │                                                 #   per-class F1, Nemenyi post-hoc tables
    ├── 03_run_cnn1d_baseline.py                     # Multi-scale CNN-1D (v8) baseline, 48,472 params
    ├── 04_compute_perclass_f1_and_timing.py         # Per-class F1 + CPU timing benchmarks → Tables S6–S7
    ├── 05_run_final_analyses.py                     # SHAP, confusion matrices, Table 1, Table 4,
    │                                                 #   ablation summary tables
    ├── 06a_extract_confusion_matrices.py            # Per-database confusion-matrix figures from saved JSON
    ├── 06b_generate_figure4_confusion_matrices.py   # Combined 3-panel Figure 4
    │
    ├── tools/
    │   └── inspect_ninapro_mat_files.py  # Sanity-check a raw NinaPro .mat download
    │
    ├── dev_history/                      # Archived / exploratory — NOT used for any reported result
    │   ├── README.md                     #   (see this file for what's here and why)
    │   ├── diagnose_minirocket_exploration.py
    │   └── prototype_multi_classifier_runner.py
    │
    └── results/
        ├── Table2_main_results.csv, TableS1–S4_*.csv, Figure4_confusion_matrices.pdf/png, ...
        ├── loso_checkpoints/             # ninapro_db{2,3,7}_checkpoint.json — per-fold raw LOSO outputs
        └── final_outputs/                # Pre-computed final results used directly in the paper:
            ├── confusion_matrix_db{3,7}.npy, confusion_xgb_db{2,3,7}.png/pdf   # Fig. 3 confusion matrices
            ├── shap_group_importance_db{2,3,7}.csv     # SHAP feature-group importance (Table 9)
            ├── shap_top20_db{2,3,7}.csv                # Top-20 individual SHAP features (Table 10)
            ├── shap_all_features_db{2,3,7}.csv         # Full per-feature SHAP values
            ├── shap_group_pie_db{2,3,7}.png/pdf        # SHAP group importance pie charts (Fig. 6)
            ├── shap_top20_bar_db{2,3,7}.png/pdf        # SHAP top-20 bar charts
            ├── Table1_dataset_characteristics.csv      # Table 1
            ├── Table4_literature_comparison.csv        # Table 8
            ├── TableS_feature_ablation_db7.csv         # Table 11
            ├── TableS_window_ablation_db7.csv          # Table 12
            ├── TableS6_fold_details_db{2,3,7}.csv      # Per-fold LOSO results
            ├── TableS6_perclass_f1_db{2,3,7}.csv       # Per-class F1 (Supplementary S2d–f)
            └── TableS7_real_timing_db{2,3,7}.csv       # CPU timing benchmarks (Table 14, Supp. S3)
```

> The pipeline scripts are numbered by phase (`01` = main LOSO benchmark, `02` = ablations, `03` = CNN-1D baseline, `04` = per-class F1 & timing, `05` = final analyses/SHAP, `06` = figure generation) rather than by development date, so the order in a file listing matches the order described in the paper. Two early, superseded scripts are kept for transparency under `dev_history/` but are not part of the reported pipeline. Every file under `results/final_outputs/` corresponds directly to a numbered Table or Figure in the manuscript.

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
git clone https://github.com/Qussai-BME/sEMG-Zero-Calibration-LOSO-Benchmark.git
cd sEMG-Zero-Calibration-LOSO-Benchmark

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Run the LOSO pipeline
```bash
cd validation

# Phase 1 — main LOSO benchmark (all 4 classifiers × DB2/DB3/DB7) + statistics
python 01a_run_loso_benchmark.py
python 01b_recompute_loso_statistics.py
python 01c_correct_friedman_test.py

# Phase 2 — ablations (run individually, or all at once with 02d)
python 02a_run_window_size_ablation.py
python 02b_run_feature_group_ablation.py
python 02c_analyze_ablation_statistics.py
python 02e_generate_posthoc_figures_and_tables.py

# Phase 3 — CNN-1D baseline (repeat --db for each database)
python 03_run_cnn1d_baseline.py --db ninapro_db7

# Phase 4 — per-class F1 + CPU timing
python 04_compute_perclass_f1_and_timing.py --db db7 --task all

# Phase 5 — SHAP, confusion matrices, Table 1 / Table 4, ablation summaries
python 05_run_final_analyses.py --db db7 --task all

# Phase 6 — figures
python 06a_extract_confusion_matrices.py
python 06b_generate_figure4_confusion_matrices.py
```
Every script accepts `--help` for its full set of flags (subject subsets, fast/debug modes, `--skip-existing`, etc.) — see each script's module docstring for details.

### Use pre-computed results directly (no re-run needed)
Everything reported in the paper is already available under [`validation/results/`](#repository-structure) — confusion matrices, SHAP values, ablation tables, and per-fold timing benchmarks (the SHAP/timing/Table 1/Table 4 outputs specifically live under `validation/results/final_outputs/`).

---

## Data Availability

The **NinaPro** databases are publicly available but **not redistributed** in this repository due to licensing:

- NinaPro DB2 / DB3 / DB7: http://ninaweb.hevs.ch

Download the raw `.mat` files and place them under `data/raw/`; `validation/data_loaders.py` handles parsing and the 41-class harmonisation.

```bibtex
@article{atzori2014electromyography,
  title={Electromyography data for non-invasive naturally-controlled robotic hand prostheses},
  author={Atzori, Manfredo and Gijsberts, Arjan and Castellini, Claudio and others},
  journal={Scientific Data}, volume={1}, pages={140053}, year={2014},
  doi={10.1038/sdata.2014.53}
}
```

**Code and pre-computed results:** archived in full at Zenodo — [https://doi.org/10.5281/zenodo.20982280](https://doi.org/10.5281/zenodo.20982280) (Release v1.0, June 2026), and on GitHub at the link above.

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
| statsmodels | 0.14 |

See [`requirements.txt`](requirements.txt) for the full, versioned dependency list (built directly from the imports in `validation/`).

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
@article{adlbi2026restclass,
  title   = {Rest-Class Metric Inflation in Zero-Calibration Cross-Subject sEMG:
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