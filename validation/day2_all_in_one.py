#!/usr/bin/env python3
"""
day2_all_in_one.py — THE COMPLETE DAY 2 SOLUTION (ONE COMMAND)
================================================================
Runs ALL Day 2 experiments in a single command:

  Phase 1: Window Size Ablation (7 sizes x 4 classifiers on DB7 = 28 experiments)
  Phase 2: Feature Group Ablation (5 configs x 2 classifiers on DB7 = 10 experiments)
  Phase 3: Statistical Analysis (Friedman, Nemenyi, Wilcoxon, Cohen's d)

Total experiments: 38
Estimated runtime: ~10-14 hours (resume-safe: re-run picks up where it left off)

USAGE (works from anywhere):
  python day2_all_in_one.py                        # Run everything
  python day2_all_in_one.py --skip-existing        # Skip completed (default: YES)
  python day2_all_in_one.py --phase window          # Only window ablation
  python day2_all_in_one.py --phase feature         # Only feature ablation
  python day2_all_in_one.py --phase stats           # Only statistical analysis
  python day2_all_in_one.py --phase all             # Run all (default)
  python day2_all_in_one.py --summary-only          # Generate summaries only

IMPORTANT CORRECTIONS from Day 1 review:
  - Feature dimensions: 438D pre-FS → 420D after FS (k=420)
    NOT "426D" (that number never existed)
  - Pre-FS breakdown: 312 (TD: 26x12) + 66 (ICC) + 12 (Corr) + 48 (TKEO) = 438D
  - DB3 = 29 movements (movement_map removes E3), DB7 = 41, DB2 = 41

OUTPUT (in validation/paper1_results/):
  Window Ablation:
    DB7_window_{ms}_{clf}_results.json     (28 files)
    Table_window_ablation.csv
    Table_window_ablation.tex
    figure_window_ablation.png
  Feature Ablation:
    DB7_feat_{config}_{clf}_results.json   (10 files)
    Table_feature_ablation.csv
    Table_feature_ablation.tex
    figure_feature_ablation.png
  Statistical Analysis:
    Table_window_stats.csv / .tex
    Table_feature_stats.csv / .tex

Author: Paper 1 — Day 2 Complete Automation
"""

import os
import re
import sys
import json
import time
import copy
import argparse
import traceback
import numpy as np
from datetime import datetime

# ============================================================================
# AUTO-DETECT PATHS (works from both project root and validation/ folder)
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(SCRIPT_DIR, "config.yaml")):
    VALIDATION_DIR = SCRIPT_DIR
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
elif os.path.exists(os.path.join(SCRIPT_DIR, "validation", "config.yaml")):
    PROJECT_ROOT = SCRIPT_DIR
    VALIDATION_DIR = os.path.join(PROJECT_ROOT, "validation")
elif os.path.exists(os.path.join(os.path.dirname(SCRIPT_DIR), "validation", "config.yaml")):
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    VALIDATION_DIR = os.path.join(PROJECT_ROOT, "validation")
else:
    # Fallback: assume script is in validation/
    VALIDATION_DIR = SCRIPT_DIR
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CONFIG_PATH = os.path.join(VALIDATION_DIR, "config.yaml")
RESULTS_DIR = os.path.join(VALIDATION_DIR, "paper1_results")

print(f"[PATHS] Project root:    {PROJECT_ROOT}")
print(f"[PATHS] Validation dir:  {VALIDATION_DIR}")
print(f"[PATHS] Config:          {CONFIG_PATH}")
print(f"[PATHS] Results dir:     {RESULTS_DIR}")


# ============================================================================
# CONFIGURATION
# ============================================================================

# ── Window Ablation ──
WINDOW_SIZES_MS = [100, 150, 200, 250, 300, 400, 500]
WINDOW_CLASSIFIERS = ['XGBoost', 'LDA', 'LinearSVC', 'RandomForest']

# ── Feature Ablation ──
# CORRECTED: 438D pre-FS, 420D after FS (k=420 via hybrid MI+f_classif)
# Pre-FS: 312(TD:26x12) + 66(ICC) + 12(Corr) + 48(TKEO:4x12) = 438D
FEATURE_CONFIGS = [
    {
        'name': 'Full',
        'description': 'All features ON (438D -> 420D after FS)',
        'label': 'Full (420D)',
        'overrides': {},
    },
    {
        'name': 'noICC',
        'description': 'Remove Inter-Channel Correlation (-66D)',
        'label': '{-ICC}',
        'overrides': {'compute_inter_channel_corr': False},
    },
    {
        'name': 'noTKEO',
        'description': 'Remove TKEO Band Energy features (-48D)',
        'label': '{-TKEO}',
        'overrides': {'compute_tkeo_bands': False},
    },
    {
        'name': 'noHjorth',
        'description': 'Remove Hjorth parameters (-36D)',
        'label': '{-Hjorth}',
        'overrides': {'compute_hjorth': False},
    },
    {
        'name': 'noFreq',
        'description': 'Remove Frequency features (-84D)',
        'label': '{-Freq}',
        'overrides': {'compute_freq_features': False},
    },
]
FEATURE_CLASSIFIERS = ['XGBoost', 'LDA']

# ── Plot styling ──
PLOT_COLORS = {
    'XGBoost': '#1f77b4',
    'LDA': '#ff7f0e',
    'LinearSVC': '#2ca02c',
    'RandomForest': '#d62728',
}
PLOT_MARKERS = {
    'XGBoost': 'o',
    'LDA': 's',
    'LinearSVC': '^',
    'RandomForest': 'D',
}
BAR_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

DATASET_KEY = 'ninapro_db7'
DATASET_NAME = 'Ninapro_DB7'
N_SUBJECTS = 22


# ============================================================================
# IMPORT VALIDATION ENGINE (after path setup)
# ============================================================================
def _import_engine():
    """Import validate_engine and data_loaders with flexible path resolution."""
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, VALIDATION_DIR)
    try:
        from validation.validate_engine import load_config, process_dataset
    except ImportError:
        from validate_engine import load_config, process_dataset
    try:
        from validation.data_loaders import load_ninapro_db
    except ImportError:
        from data_loaders import load_ninapro_db
    return load_config, process_dataset, load_ninapro_db


# ============================================================================
# PROGRESS TRACKING
# ============================================================================
def _load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {'phase': '', 'experiments': [], 'started_at': datetime.now().isoformat()}


def _save_progress(progress_file, entry):
    progress = _load_progress(progress_file)
    progress['experiments'].append(entry)
    progress['last_update'] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2, default=str)


# ============================================================================
# PHASE 1: WINDOW SIZE ABLATION
# ============================================================================
def run_window_ablation(config, load_ninapro_db, process_dataset, skip_existing=True):
    """
    Phase 1: Test 7 window sizes x 4 classifiers on DB7.
    Total: 28 experiments, estimated 6-8 hours.
    """
    print("\n" + "=" * 80)
    print("  PHASE 1/3: WINDOW SIZE ABLATION (DB7)")
    print("  " + "-" * 76)
    print(f"  Window sizes: {WINDOW_SIZES_MS}")
    print(f"  Classifiers:  {WINDOW_CLASSIFIERS}")
    print(f"  Total:        {len(WINDOW_SIZES_MS)} x {len(WINDOW_CLASSIFIERS)} = "
          f"{len(WINDOW_SIZES_MS) * len(WINDOW_CLASSIFIERS)} experiments")
    print("=" * 80 + "\n")

    progress_file = os.path.join(RESULTS_DIR, "_day2_window_progress.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    completed = 0
    failed = 0
    skipped = 0
    t_phase_start = time.time()

    for i, window_ms in enumerate(WINDOW_SIZES_MS):
        for j, clf_name in enumerate(WINDOW_CLASSIFIERS):
            exp_num = i * len(WINDOW_CLASSIFIERS) + j + 1
            total = len(WINDOW_SIZES_MS) * len(WINDOW_CLASSIFIERS)

            clf_lower = clf_name.lower()
            result_filename = f"DB7_window_{window_ms}_{clf_lower}_results.json"
            result_path = os.path.join(RESULTS_DIR, result_filename)

            # ── Skip check ──
            if skip_existing and os.path.exists(result_path):
                try:
                    with open(result_path, 'r') as f:
                        existing = json.load(f)
                    if existing.get('success', False) and existing.get('classification'):
                        print(f"  [{exp_num}/{total}] SKIP: window={window_ms}ms, {clf_name} (already done)",
                              flush=True)
                        skipped += 1
                        completed += 1
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass  # File corrupted, re-run

            print(f"\n  [{'='*10}] [{exp_num}/{total}] window={window_ms}ms, clf={clf_name} [{'='*10}]",
                  flush=True)

            t0 = time.time()
            try:
                cfg = copy.deepcopy(config)

                # Set window size in BOTH base and adaptive config
                cfg['processing']['window_size_ms'] = window_ms
                if 'dataset_adaptive_configs' not in cfg:
                    cfg['dataset_adaptive_configs'] = {}
                if DATASET_KEY not in cfg['dataset_adaptive_configs']:
                    cfg['dataset_adaptive_configs'][DATASET_KEY] = {}
                if 'processing' not in cfg['dataset_adaptive_configs'][DATASET_KEY]:
                    cfg['dataset_adaptive_configs'][DATASET_KEY]['processing'] = {}

                cfg['dataset_adaptive_configs'][DATASET_KEY]['processing']['window_size_ms'] = window_ms
                cfg['dataset_adaptive_configs'][DATASET_KEY]['processing']['overlap'] = 0.5
                cfg['processing']['overlap'] = 0.5

                # Set classifier
                cfg['classification']['classifier'] = clf_name

                # Load DB7
                db7_path = cfg['datasets'][DATASET_KEY]['path']
                if not os.path.exists(db7_path):
                    raise FileNotFoundError(f"DB7 data path not found: {db7_path}")

                loader = load_ninapro_db(
                    db_version='DB7', data_path=db7_path,
                    subjects=None, movement_map=None, remove_class_zero=False,
                )

                # Run pipeline
                checkpoint = type('', (), {'get': lambda *a: [], 'update': lambda *a: None})()
                result = process_dataset(loader, DATASET_NAME, DATASET_KEY, cfg, checkpoint,
                                         quick=False, resume=False)

                if result is None:
                    raise RuntimeError("process_dataset returned None")

                output_data, trained_models, X_tests, feat_names = result
                elapsed = time.time() - t0

                # Build result
                classification = output_data.get('classification')
                if classification is not None:
                    acc, std, cm = classification
                    clf_list = [float(acc), float(std), cm.tolist() if hasattr(cm, 'tolist') else cm]
                else:
                    clf_list = None

                experiment_result = {
                    'success': True,
                    'ablation_type': 'window_size',
                    'dataset': DATASET_KEY,
                    'classifier': clf_name,
                    'window_size_ms': window_ms,
                    'overlap': 0.5,
                    'n_subjects': output_data.get('n_subjects'),
                    'n_channels': output_data.get('n_channels'),
                    'sampling_rate': output_data.get('sampling_rate'),
                    'n_movements': output_data.get('n_movements'),
                    'classification': clf_list,
                    'per_subject_accuracy': output_data.get('per_subject_accuracy', []),
                    'elapsed_seconds': round(elapsed, 1),
                    'timestamp': datetime.now().isoformat(),
                }

                with open(result_path, 'w') as f:
                    json.dump(experiment_result, f, indent=2, default=str)

                _save_progress(progress_file, {
                    'phase': 'window', 'window_ms': window_ms, 'classifier': clf_name,
                    'status': 'completed',
                    'accuracy': float(clf_list[0]) if clf_list else None,
                    'elapsed_seconds': elapsed,
                })

                mean_acc = clf_list[0] if clf_list else 'N/A'
                print(f"  DONE: window={window_ms}ms, {clf_name}: acc={mean_acc:.4f} ({elapsed:.0f}s)",
                      flush=True)
                completed += 1

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  FAILED: window={window_ms}ms, {clf_name}: {e}", flush=True)
                traceback.print_exc()

                error_result = {
                    'success': False, 'ablation_type': 'window_size', 'dataset': DATASET_KEY,
                    'classifier': clf_name, 'window_size_ms': window_ms,
                    'error': str(e), 'elapsed_seconds': round(elapsed, 1),
                    'timestamp': datetime.now().isoformat(),
                }
                with open(result_path, 'w') as f:
                    json.dump(error_result, f, indent=2, default=str)

                _save_progress(progress_file, {
                    'phase': 'window', 'window_ms': window_ms, 'classifier': clf_name,
                    'status': 'failed', 'error': str(e), 'elapsed_seconds': elapsed,
                })
                failed += 1

    t_phase = time.time() - t_phase_start
    total_exp = len(WINDOW_SIZES_MS) * len(WINDOW_CLASSIFIERS)
    print(f"\n{'='*80}")
    print(f"  PHASE 1 COMPLETE: Window Ablation")
    print(f"  Completed: {completed}/{total_exp}  |  Skipped: {skipped}  |  Failed: {failed}")
    print(f"  Time: {t_phase:.0f}s ({t_phase/60:.1f}min)")
    print(f"{'='*80}\n")

    return completed, failed


# ============================================================================
# PHASE 2: FEATURE GROUP ABLATION
# ============================================================================
def run_feature_ablation(config, load_ninapro_db, process_dataset, skip_existing=True):
    """
    Phase 2: Test 5 feature configs x 2 classifiers on DB7.
    Total: 10 experiments, estimated 4-6 hours.
    """
    print("\n" + "=" * 80)
    print("  PHASE 2/3: FEATURE GROUP ABLATION (DB7)")
    print("  " + "-" * 76)
    print("  Feature configs:")
    for cfg_item in FEATURE_CONFIGS:
        print(f"    - {cfg_item['name']:10s}: {cfg_item['description']}")
    print(f"  Classifiers:  {FEATURE_CLASSIFIERS}")
    print(f"  Total:        {len(FEATURE_CONFIGS)} x {len(FEATURE_CLASSIFIERS)} = "
          f"{len(FEATURE_CONFIGS) * len(FEATURE_CLASSIFIERS)} experiments")
    print("=" * 80 + "\n")

    progress_file = os.path.join(RESULTS_DIR, "_day2_feature_progress.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    completed = 0
    failed = 0
    skipped = 0
    t_phase_start = time.time()

    for i, feat_cfg in enumerate(FEATURE_CONFIGS):
        for j, clf_name in enumerate(FEATURE_CLASSIFIERS):
            exp_num = i * len(FEATURE_CLASSIFIERS) + j + 1
            total = len(FEATURE_CONFIGS) * len(FEATURE_CLASSIFIERS)

            clf_lower = clf_name.lower()
            result_filename = f"DB7_feat_{feat_cfg['name']}_{clf_lower}_results.json"
            result_path = os.path.join(RESULTS_DIR, result_filename)

            # ── Skip check ──
            if skip_existing and os.path.exists(result_path):
                try:
                    with open(result_path, 'r') as f:
                        existing = json.load(f)
                    if existing.get('success', False) and existing.get('classification'):
                        print(f"  [{exp_num}/{total}] SKIP: feat={feat_cfg['name']}, {clf_name} (already done)",
                              flush=True)
                        skipped += 1
                        completed += 1
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass

            print(f"\n  [{'='*10}] [{exp_num}/{total}] feat={feat_cfg['name']}, clf={clf_name} [{'='*10}]",
                  flush=True)
            print(f"    Description: {feat_cfg['description']}", flush=True)

            t0 = time.time()
            try:
                cfg = copy.deepcopy(config)

                # Apply feature overrides to adaptive config
                adaptive_proc = cfg.setdefault('dataset_adaptive_configs', {}). \
                    setdefault(DATASET_KEY, {}). \
                    setdefault('processing', {})

                for flag_key, flag_val in feat_cfg['overrides'].items():
                    print(f"    Setting {flag_key} = {flag_val}", flush=True)
                    adaptive_proc[flag_key] = flag_val
                    cfg['processing'][flag_key] = flag_val

                # Ensure EA stays ON and overlap=0.5
                adaptive_proc['euclidean_alignment'] = True
                adaptive_proc['overlap'] = 0.5
                cfg['processing']['overlap'] = 0.5

                # Set classifier
                cfg['classification']['classifier'] = clf_name

                # Load DB7
                db7_path = cfg['datasets'][DATASET_KEY]['path']
                if not os.path.exists(db7_path):
                    raise FileNotFoundError(f"DB7 data path not found: {db7_path}")

                loader = load_ninapro_db(
                    db_version='DB7', data_path=db7_path,
                    subjects=None, movement_map=None, remove_class_zero=False,
                )

                # Run pipeline
                checkpoint = type('', (), {'get': lambda *a: [], 'update': lambda *a: None})()
                result = process_dataset(loader, DATASET_NAME, DATASET_KEY, cfg, checkpoint,
                                         quick=False, resume=False)

                if result is None:
                    raise RuntimeError("process_dataset returned None")

                output_data, trained_models, X_tests, feat_names = result
                elapsed = time.time() - t0

                # Build result
                classification = output_data.get('classification')
                if classification is not None:
                    acc, std, cm = classification
                    clf_list = [float(acc), float(std), cm.tolist() if hasattr(cm, 'tolist') else cm]
                else:
                    clf_list = None

                experiment_result = {
                    'success': True,
                    'ablation_type': 'feature_group',
                    'dataset': DATASET_KEY,
                    'classifier': clf_name,
                    'feature_config': feat_cfg['name'],
                    'feature_label': feat_cfg['label'],
                    'feature_overrides': feat_cfg['overrides'],
                    'n_subjects': output_data.get('n_subjects'),
                    'n_channels': output_data.get('n_channels'),
                    'sampling_rate': output_data.get('sampling_rate'),
                    'n_movements': output_data.get('n_movements'),
                    'classification': clf_list,
                    'per_subject_accuracy': output_data.get('per_subject_accuracy', []),
                    'elapsed_seconds': round(elapsed, 1),
                    'timestamp': datetime.now().isoformat(),
                }

                with open(result_path, 'w') as f:
                    json.dump(experiment_result, f, indent=2, default=str)

                _save_progress(progress_file, {
                    'phase': 'feature', 'feature_config': feat_cfg['name'],
                    'classifier': clf_name, 'status': 'completed',
                    'accuracy': float(clf_list[0]) if clf_list else None,
                    'elapsed_seconds': elapsed,
                })

                mean_acc = clf_list[0] if clf_list else 'N/A'
                print(f"  DONE: feat={feat_cfg['name']}, {clf_name}: acc={mean_acc:.4f} ({elapsed:.0f}s)",
                      flush=True)
                completed += 1

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  FAILED: feat={feat_cfg['name']}, {clf_name}: {e}", flush=True)
                traceback.print_exc()

                error_result = {
                    'success': False, 'ablation_type': 'feature_group', 'dataset': DATASET_KEY,
                    'classifier': clf_name, 'feature_config': feat_cfg['name'],
                    'feature_label': feat_cfg['label'], 'error': str(e),
                    'elapsed_seconds': round(elapsed, 1), 'timestamp': datetime.now().isoformat(),
                }
                with open(result_path, 'w') as f:
                    json.dump(error_result, f, indent=2, default=str)

                _save_progress(progress_file, {
                    'phase': 'feature', 'feature_config': feat_cfg['name'],
                    'classifier': clf_name, 'status': 'failed',
                    'error': str(e), 'elapsed_seconds': elapsed,
                })
                failed += 1

    t_phase = time.time() - t_phase_start
    total_exp = len(FEATURE_CONFIGS) * len(FEATURE_CLASSIFIERS)
    print(f"\n{'='*80}")
    print(f"  PHASE 2 COMPLETE: Feature Ablation")
    print(f"  Completed: {completed}/{total_exp}  |  Skipped: {skipped}  |  Failed: {failed}")
    print(f"  Time: {t_phase:.0f}s ({t_phase/60:.1f}min)")
    print(f"{'='*80}\n")

    return completed, failed


# ============================================================================
# PHASE 3: STATISTICAL ANALYSIS
# ============================================================================
def run_statistical_analysis():
    """
    Phase 3: Read all ablation JSON results and generate:
      - Friedman + Nemenyi for window ablation
      - Wilcoxon signed-rank + Cohen's d for feature ablation
      - CSV + LaTeX tables for both
    """
    from scipy.stats import f as f_dist, norm, rankdata, wilcoxon, studentized_range

    print("\n" + "=" * 80)
    print("  PHASE 3/3: STATISTICAL ANALYSIS")
    print("=" * 80 + "\n")

    # ── Helper: Friedman test ──
    def friedman_test(acc_matrix):
        n_subjects, n_conditions = acc_matrix.shape
        if n_subjects < 3 or n_conditions < 3:
            return np.nan, np.nan, n_subjects, n_conditions, np.zeros(n_conditions)
        ranks = np.zeros_like(acc_matrix)
        for i in range(n_subjects):
            ranks[i] = rankdata(acc_matrix[i])
        mean_ranks = ranks.mean(axis=0)
        SS_between = n_subjects * np.sum((mean_ranks - (n_conditions + 1) / 2.0) ** 2)
        chi2_r = 12.0 * n_subjects / (n_conditions * (n_conditions + 1)) * SS_between
        F_stat = (chi2_r * (n_subjects - 1)) / (n_subjects * (n_conditions - 1) - chi2_r)
        df1 = n_conditions - 1
        df2 = (n_subjects - 1) * df1
        if df2 <= 0 or F_stat < 0:
            return chi2_r, 1.0, n_subjects, n_conditions, mean_ranks
        p_value = 1.0 - f_dist.cdf(F_stat, df1, df2)
        return chi2_r, p_value, n_subjects, n_conditions, mean_ranks

    # ── Helper: Nemenyi post-hoc ──
    def nemenyi_posthoc(mean_ranks, n_subjects, n_conditions, alpha=0.05):
        q_alpha = studentized_range.ppf(1 - alpha, n_conditions, np.inf)
        cd = q_alpha * np.sqrt(n_conditions * (n_conditions + 1) / (6.0 * n_subjects))
        comparisons = []
        for i in range(n_conditions):
            for j in range(i + 1, n_conditions):
                diff = abs(mean_ranks[i] - mean_ranks[j])
                comparisons.append({
                    'cond_i': int(i), 'cond_j': int(j),
                    'rank_diff': round(float(diff), 4),
                    'critical_distance': round(float(cd), 4),
                    'significant': diff > cd,
                })
        return cd, comparisons

    # ── Helper: Wilcoxon + Cohen's d ──
    def wilcoxon_test(a, b):
        a, b = np.array(a, float), np.array(b, float)
        diff = a - b
        mean_diff = float(np.mean(diff))
        try:
            W, p_val = wilcoxon(diff, zero_method='wilcox', alternative='two-sided')
        except ValueError:
            return 0.0, 1.0, 0.0, mean_diff, 0.0
        n_nz = int(np.count_nonzero(diff))
        r = 1 - (2.0 * W) / (n_nz * (n_nz + 1)) if n_nz > 0 else 0.0
        std_d = np.std(diff, ddof=1)
        d = float(np.mean(diff) / std_d) if std_d > 0 else 0.0
        return float(W), float(p_val), float(r), mean_diff, d

    # ── Helper: Bonferroni-Holm ──
    def holm_correction(p_values, alpha=0.05):
        n = len(p_values)
        if n == 0:
            return []
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * n
        for rank, (orig_idx, p) in enumerate(indexed):
            adjusted[orig_idx] = min(1.0, p * (n - rank))
        return list(zip(adjusted, [p < alpha for p in adjusted]))

    ALPHA = 0.05

    # ════════════════════════════════════════════════════════════════════
    # 3A: WINDOW ABLATION STATISTICS
    # ════════════════════════════════════════════════════════════════════
    print("  --- 3A: Window Ablation Statistics ---\n", flush=True)

    window_data = {}
    for clf in WINDOW_CLASSIFIERS:
        for wms in WINDOW_SIZES_MS:
            fname = f"DB7_window_{wms}_{clf.lower()}_results.json"
            fpath = os.path.join(RESULTS_DIR, fname)
            if not os.path.exists(fpath):
                print(f"    [WARN] Missing: {fname}", flush=True)
                continue
            try:
                with open(fpath, 'r') as f:
                    res = json.load(f)
                if not res.get('success') or not res.get('classification'):
                    continue
                window_data[(wms, clf)] = {
                    'acc': res['classification'][0], 'std': res['classification'][1],
                    'per_subject': res.get('per_subject_accuracy', []),
                }
            except (json.JSONDecodeError, KeyError) as e:
                print(f"    [WARN] Error reading {fname}: {e}", flush=True)

    # Per-classifier Friedman
    window_csv_rows = []
    for clf in WINDOW_CLASSIFIERS:
        available = sorted(set(wms for (wms, c) in window_data if c == clf))
        if len(available) < 3:
            continue

        acc_matrix = np.column_stack([
            np.array(window_data[(wms, clf)]['per_subject']) for wms in available
        ])
        chi2, p_val, n_subj, n_cond, mean_ranks = friedman_test(acc_matrix)

        sig = "***" if (not np.isnan(p_val) and p_val < 0.001) else \
              "**" if (not np.isnan(p_val) and p_val < 0.01) else \
              "*" if (not np.isnan(p_val) and p_val < 0.05) else "n.s."

        print(f"    {clf}: Friedman X2={chi2:.4f}, p={p_val:.4f} ({sig}) "
              f"[n={n_subj}, k={n_cond}]", flush=True)

        window_csv_rows.append({
            'classifier': clf, 'test': 'Friedman',
            'chi2': f"{chi2:.4f}" if not np.isnan(chi2) else "N/A",
            'p_value': f"{p_val:.4f}" if not np.isnan(p_val) else "N/A",
            'significant': sig, 'n_subjects': n_subj, 'n_conditions': n_cond,
        })

        # Nemenyi post-hoc if significant
        if not np.isnan(p_val) and p_val < ALPHA:
            cd, comps = nemenyi_posthoc(mean_ranks, n_subj, n_cond, ALPHA)
            print(f"      Nemenyi CD = {cd:.4f}", flush=True)
            for comp in comps:
                w_i, w_j = available[comp['cond_i']], available[comp['cond_j']]
                mark = "***" if comp['significant'] else ""
                print(f"        {w_i}ms vs {w_j}ms: rank_diff={comp['rank_diff']:.3f}, "
                      f"CD={comp['critical_distance']:.3f} {mark}", flush=True)

    # Save Window Stats CSV
    if window_csv_rows:
        csv_path = os.path.join(RESULTS_DIR, 'Table_window_stats.csv')
        import csv as csv_mod
        with open(csv_path, 'w', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(window_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(window_csv_rows)
        print(f"\n    [SAVED] {csv_path}", flush=True)

    # Save Window Stats LaTeX
    tex_path = os.path.join(RESULTS_DIR, 'Table_window_stats.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Window Ablation --- Friedman Test Results (DB7, 22-fold LOSO)}" + "\n")
        f.write(r"\label{tab:window_stats}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Classifier & $\chi^2$ & $p$-value & $n$ & $k$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        for row in window_csv_rows:
            sig = r"$^{\ast\ast}$" if row['significant'] == "***" else \
                  r"$^{\ast}$" if row['significant'] == "*" else ""
            f.write(f"{row['classifier']} & {row['chi2']} & {row['p_value']}{sig} "
                    f"& {row['n_subjects']} & {row['n_conditions']} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"    [SAVED] {tex_path}", flush=True)

    # ════════════════════════════════════════════════════════════════════
    # 3B: FEATURE ABLATION STATISTICS
    # ════════════════════════════════════════════════════════════════════
    print("\n  --- 3B: Feature Ablation Statistics ---\n", flush=True)

    feat_data = {}
    for feat_cfg in FEATURE_CONFIGS:
        for clf in FEATURE_CLASSIFIERS:
            fname = f"DB7_feat_{feat_cfg['name']}_{clf.lower()}_results.json"
            fpath = os.path.join(RESULTS_DIR, fname)
            if not os.path.exists(fpath):
                print(f"    [WARN] Missing: {fname}", flush=True)
                continue
            try:
                with open(fpath, 'r') as f:
                    res = json.load(f)
                if not res.get('success') or not res.get('classification'):
                    continue
                feat_data[(feat_cfg['name'], clf)] = {
                    'acc': res['classification'][0], 'std': res['classification'][1],
                    'per_subject': res.get('per_subject_accuracy', []),
                    'label': res.get('feature_label', feat_cfg['name']),
                }
            except (json.JSONDecodeError, KeyError) as e:
                print(f"    [WARN] Error reading {fname}: {e}", flush=True)

    feature_csv_rows = []
    for clf in FEATURE_CLASSIFIERS:
        full_key = ('Full', clf)
        if full_key not in feat_data:
            print(f"    {clf}: Full baseline missing, skipping", flush=True)
            continue

        full_acc = feat_data[full_key]['per_subject']
        full_mean = feat_data[full_key]['acc']
        print(f"    {clf}: Full baseline = {full_mean:.4f}", flush=True)

        raw_p_values = []
        comp_names = []

        for feat_cfg in FEATURE_CONFIGS:
            if feat_cfg['name'] == 'Full':
                continue
            key = (feat_cfg['name'], clf)
            if key not in feat_data:
                print(f"      {feat_cfg['name']}: MISSING", flush=True)
                continue

            ablation_acc = feat_data[key]['per_subject']
            W, p_val, r, mean_diff, cohens_d = wilcoxon_test(full_acc, ablation_acc)
            raw_p_values.append(p_val)
            comp_names.append(feat_cfg['name'])

            d_abs = abs(cohens_d)
            d_interp = "negligible" if d_abs < 0.2 else "small" if d_abs < 0.5 else \
                       "medium" if d_abs < 0.8 else "large"
            sig_mark = "***" if p_val < 0.001 else "**" if p_val < 0.01 else \
                       "*" if p_val < 0.05 else "n.s."
            direction = "+" if mean_diff > 0 else "-"

            print(f"      {feat_cfg['label']:>12s} vs Full: W={W:.0f}, p={p_val:.4f} {sig_mark}, "
                  f"d={direction}{abs(mean_diff)*100:.2f}%, d={cohens_d:.3f} ({d_interp})", flush=True)

            feature_csv_rows.append({
                'classifier': clf,
                'comparison': f"{feat_cfg['label']} vs Full",
                'config': feat_cfg['name'],
                'mean_full': f"{full_mean:.4f}",
                'mean_ablation': f"{feat_data[key]['acc']:.4f}",
                'mean_diff_pct': f"{mean_diff*100:.2f}",
                'W': f"{W:.0f}", 'p_value': f"{p_val:.4f}",
                'rank_biserial_r': f"{r:.4f}",
                'cohens_d': f"{cohens_d:.4f}",
                'effect_size': d_interp,
                'significant': sig_mark,
            })

        # Bonferroni-Holm correction
        if raw_p_values:
            corrected = holm_correction(raw_p_values, ALPHA)
            print(f"\n      Bonferroni-Holm corrected:", flush=True)
            for name, (adj_p, sig) in zip(comp_names, corrected):
                sig_mark = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else \
                           "*" if adj_p < 0.05 else "n.s."
                label = next((FEATURE_CONFIGS[i]['label'] for i, c in enumerate(FEATURE_CONFIGS)
                              if c['name'] == name), name)
                print(f"        {label:>12s}: adj_p={adj_p:.4f} {sig_mark}", flush=True)

            # Update CSV rows with corrected
            for row, name, (adj_p, sig) in zip(
                [r for r in feature_csv_rows if r['classifier'] == clf],
                comp_names, corrected
            ):
                row['p_value_corrected'] = f"{adj_p:.4f}"
                row['significant_corrected'] = "Yes" if sig else "No"

    # Save Feature Stats CSV
    if feature_csv_rows:
        csv_path = os.path.join(RESULTS_DIR, 'Table_feature_stats.csv')
        import csv as csv_mod
        with open(csv_path, 'w', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(feature_csv_rows[0].keys()),
                                        extrasaction='ignore')
            writer.writeheader()
            writer.writerows(feature_csv_rows)
        print(f"\n    [SAVED] {csv_path}", flush=True)

    # Save Feature Stats LaTeX
    tex_path = os.path.join(RESULTS_DIR, 'Table_feature_stats.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Feature Ablation --- Wilcoxon Signed-Rank Tests (DB7, 22-fold LOSO)}" + "\n")
        f.write(r"\label{tab:feature_stats}" + "\n")
        f.write(r"\begin{tabular}{llrrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Classifier & Config & $\Delta$Acc (\%) & $p$ & $p_{adj}$ & $d$ & Sig. \\" + "\n")
        f.write(r"\midrule" + "\n")
        for clf in FEATURE_CLASSIFIERS:
            clf_rows = [r for r in feature_csv_rows if r['classifier'] == clf]
            for row in clf_rows:
                d_val = float(row.get('cohens_d', 0))
                sig = row.get('significant_corrected', row.get('significant', ''))
                adj_p = row.get('p_value_corrected', row.get('p_value', '---'))
                f.write(f"{clf} & {row['comparison']} & {row['mean_diff_pct']} & "
                        f"{row['p_value']} & {adj_p} & {abs(d_val):.2f} & {sig} \\\\\n")
            if clf != FEATURE_CLASSIFIERS[-1]:
                f.write(r"\midrule" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"    [SAVED] {tex_path}", flush=True)

    print(f"\n{'='*80}")
    print("  PHASE 3 COMPLETE: Statistical Analysis")
    print(f"{'='*80}\n")


# ============================================================================
# SUMMARY GENERATION (CSV + LaTeX + PNG figures)
# ============================================================================
def generate_window_summary():
    """Generate CSV, LaTeX, and PNG for window ablation results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = {}
    for clf in WINDOW_CLASSIFIERS:
        for wms in WINDOW_SIZES_MS:
            fname = f"DB7_window_{wms}_{clf.lower()}_results.json"
            fpath = os.path.join(RESULTS_DIR, fname)
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, 'r') as f:
                    res = json.load(f)
                if not res.get('success') or not res.get('classification'):
                    continue
                data[(wms, clf)] = {'acc': res['classification'][0], 'std': res['classification'][1]}
            except:
                continue

    if not data:
        print("  [WARN] No window results for summary", flush=True)
        return

    # CSV
    csv_path = os.path.join(RESULTS_DIR, 'Table_window_ablation.csv')
    with open(csv_path, 'w') as f:
        f.write(','.join(['Window_ms'] + WINDOW_CLASSIFIERS + ['Best_Clf', 'Best_Acc']) + '\n')
        for wms in WINDOW_SIZES_MS:
            row = [str(wms)]
            best_acc, best_clf = -1, ''
            for clf in WINDOW_CLASSIFIERS:
                if (wms, clf) in data:
                    row.append(f"{data[(wms, clf)]['acc']:.4f} +/- {data[(wms, clf)]['std']:.4f}")
                    if data[(wms, clf)]['acc'] > best_acc:
                        best_acc = data[(wms, clf)]['acc']
                        best_clf = clf
                else:
                    row.append('---')
            row.extend([best_clf, f"{best_acc:.4f}" if best_acc > 0 else '---'])
            f.write(','.join(row) + '\n')
    print(f"  [SAVED] {csv_path}", flush=True)

    # LaTeX
    tex_path = os.path.join(RESULTS_DIR, 'Table_window_ablation.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n\centering\n")
        f.write(r"\caption{Window Size Ablation on DB7 (22-fold LOSO, 41 classes)}" + "\n")
        f.write(r"\label{tab:window_ablation}" + "\n")
        f.write(r"\begin{tabular}{l" + "c" * len(WINDOW_CLASSIFIERS) + "}\n")
        f.write(r"\toprule\nWindow (ms) & " + " & ".join(WINDOW_CLASSIFIERS) + r" \\" + "\n")
        f.write(r"\midrule\n")
        for wms in WINDOW_SIZES_MS:
            parts = [str(wms)]
            for clf in WINDOW_CLASSIFIERS:
                if (wms, clf) in data:
                    parts.append(f"{data[(wms,clf)]['acc']:.2f}\\% $\\pm$ {data[(wms,clf)]['std']:.2f}")
                else:
                    parts.append("---")
            f.write(" & ".join(parts) + r" \\" + "\n")
        f.write(r"\bottomrule\n\end{tabular}\n\end{table}\n")
    print(f"  [SAVED] {tex_path}", flush=True)

    # PNG
    fig_path = os.path.join(RESULTS_DIR, 'figure_window_ablation.png')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for clf in WINDOW_CLASSIFIERS:
        xs, ys, yerrs = [], [], []
        for wms in WINDOW_SIZES_MS:
            if (wms, clf) in data:
                xs.append(wms)
                ys.append(data[(wms, clf)]['acc'] * 100)
                yerrs.append(data[(wms, clf)]['std'] * 100)
        if xs:
            ax.errorbar(xs, ys, yerr=yerrs, label=clf,
                        color=PLOT_COLORS[clf], marker=PLOT_MARKERS[clf],
                        markersize=8, linewidth=2, capsize=4, capthick=1.5)
    ax.set_xlabel('Window Size (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Window Size Ablation --- NinaPro DB7 (22-fold LOSO, 41 classes)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(WINDOW_SIZES_MS)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {fig_path}", flush=True)

    # Console summary
    print(f"\n  WINDOW ABLATION SUMMARY:", flush=True)
    print(f"  {'Window':>8s}", end='')
    for clf in WINDOW_CLASSIFIERS:
        print(f"  {clf:>14s}", end='')
    print(flush=True)
    for wms in WINDOW_SIZES_MS:
        print(f"  {wms:>6d}ms", end='')
        for clf in WINDOW_CLASSIFIERS:
            if (wms, clf) in data:
                d = data[(wms, clf)]
                print(f"  {d['acc']*100:>6.2f}+/-{d['std']*100:.2f}", end='')
            else:
                print(f"  {'---':>14s}", end='')
        print(flush=True)


def generate_feature_summary():
    """Generate CSV, LaTeX, and PNG for feature ablation results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = {}
    for feat_cfg in FEATURE_CONFIGS:
        for clf in FEATURE_CLASSIFIERS:
            fname = f"DB7_feat_{feat_cfg['name']}_{clf.lower()}_results.json"
            fpath = os.path.join(RESULTS_DIR, fname)
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, 'r') as f:
                    res = json.load(f)
                if not res.get('success') or not res.get('classification'):
                    continue
                data[(feat_cfg['name'], clf)] = {
                    'acc': res['classification'][0], 'std': res['classification'][1],
                    'label': feat_cfg['label'],
                }
            except:
                continue

    if not data:
        print("  [WARN] No feature results for summary", flush=True)
        return

    # CSV
    csv_path = os.path.join(RESULTS_DIR, 'Table_feature_ablation.csv')
    with open(csv_path, 'w') as f:
        f.write(','.join(['Config', 'Label'] + FEATURE_CLASSIFIERS + ['Best_Clf', 'Best_Acc']) + '\n')
        for feat_cfg in FEATURE_CONFIGS:
            row = [feat_cfg['name'], feat_cfg['label']]
            best_acc, best_clf = -1, ''
            for clf in FEATURE_CLASSIFIERS:
                if (feat_cfg['name'], clf) in data:
                    d = data[(feat_cfg['name'], clf)]
                    row.append(f"{d['acc']:.4f} +/- {d['std']:.4f}")
                    if d['acc'] > best_acc:
                        best_acc = d['acc']
                        best_clf = clf
                else:
                    row.append('---')
            row.extend([best_clf, f"{best_acc:.4f}" if best_acc > 0 else '---'])
            f.write(','.join(row) + '\n')
    print(f"  [SAVED] {csv_path}", flush=True)

    # LaTeX
    tex_path = os.path.join(RESULTS_DIR, 'Table_feature_ablation.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n\centering\n")
        f.write(r"\caption{Feature Group Ablation on DB7 (22-fold LOSO, 41 classes)}" + "\n")
        f.write(r"\label{tab:feature_ablation}" + "\n")
        f.write(r"\begin{tabular}{l" + "c" * len(FEATURE_CLASSIFIERS) + "}\n")
        f.write(r"\toprule\nConfiguration & " + " & ".join(FEATURE_CLASSIFIERS) + r" \\" + "\n")
        f.write(r"\midrule\n")
        for feat_cfg in FEATURE_CONFIGS:
            parts = [feat_cfg['label']]
            for clf in FEATURE_CLASSIFIERS:
                if (feat_cfg['name'], clf) in data:
                    d = data[(feat_cfg['name'], clf)]
                    parts.append(f"{d['acc']:.2f}\\% $\\pm$ {d['std']:.2f}")
                else:
                    parts.append("---")
            f.write(" & ".join(parts) + r" \\" + "\n")
        f.write(r"\bottomrule\n\end{tabular}\n\end{table}\n")
    print(f"  [SAVED] {tex_path}", flush=True)

    # PNG
    fig_path = os.path.join(RESULTS_DIR, 'figure_feature_ablation.png')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    x = np.arange(len(FEATURE_CONFIGS))
    bar_width = 0.35
    for clf_idx, clf in enumerate(FEATURE_CLASSIFIERS):
        accs, errs = [], []
        for feat_cfg in FEATURE_CONFIGS:
            if (feat_cfg['name'], clf) in data:
                d = data[(feat_cfg['name'], clf)]
                accs.append(d['acc'] * 100)
                errs.append(d['std'] * 100)
            else:
                accs.append(0)
                errs.append(0)
        offset = (clf_idx - 0.5) * bar_width
        ax.bar(x + offset, accs, bar_width * 0.9, yerr=errs, label=clf,
               color=BAR_COLORS[clf_idx], edgecolor='white', linewidth=0.5,
               capsize=3, error_kw={'elinewidth': 1}, alpha=0.85)
    ax.set_xlabel('Feature Configuration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Feature Group Ablation --- NinaPro DB7 (22-fold LOSO, 41 classes)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([cfg['label'] for cfg in FEATURE_CONFIGS], fontsize=11)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {fig_path}", flush=True)

    # Console summary
    print(f"\n  FEATURE ABLATION SUMMARY:", flush=True)
    print(f"  {'Config':>10s}  {'Label':>12s}", end='')
    for clf in FEATURE_CLASSIFIERS:
        print(f"  {clf:>16s}", end='')
    print(flush=True)
    for feat_cfg in FEATURE_CONFIGS:
        print(f"  {feat_cfg['name']:>10s}  {feat_cfg['label']:>12s}", end='')
        for clf in FEATURE_CLASSIFIERS:
            if (feat_cfg['name'], clf) in data:
                d = data[(feat_cfg['name'], clf)]
                print(f"  {d['acc']*100:>6.2f}+/-{d['std']*100:.2f}", end='')
            else:
                print(f"  {'---':>16s}", end='')
        print(flush=True)


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Day 2 All-In-One: Window + Feature Ablation + Statistics (ONE COMMAND)"
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (auto-detected by default)')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['all', 'window', 'feature', 'stats'],
                        help='Which phase(s) to run (default: all)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip completed experiments (default: True)')
    parser.add_argument('--no-skip', dest='skip_existing', action='store_false',
                        help='Re-run even completed experiments')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only generate summary tables/plots from existing results')
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================
def main():
    args = parse_args()

    # ── Print banner ──
    print("=" * 80)
    print("  DAY 2 ALL-IN-ONE: Window Ablation + Feature Ablation + Statistics")
    print("  Paper 1: LOSO Cross-Validation Benchmark on NinaPro EMG")
    print("=" * 80)
    print(f"  Timestamp:     {datetime.now().isoformat()}")
    print(f"  Config path:   {CONFIG_PATH}")
    print(f"  Results dir:   {RESULTS_DIR}")
    print(f"  Phase:         {args.phase}")
    print(f"  Skip existing: {args.skip_existing}")
    print("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Summary-only mode ──
    if args.summary_only:
        print("\n  SUMMARY-ONLY MODE: Generating tables and plots from existing results...\n")
        generate_window_summary()
        generate_feature_summary()
        print("\n  DONE. Check:", RESULTS_DIR)
        return

    # ── Import engine ──
    print("\n  Importing validation engine...", flush=True)
    try:
        load_config, process_dataset, load_ninapro_db = _import_engine()
        print("  [OK] Engine imported successfully", flush=True)
    except ImportError as e:
        print(f"\n  [FATAL] Cannot import validation engine: {e}")
        print(f"  Make sure this script is in the validation/ folder alongside config.yaml")
        print(f"  Or run from project root: python validation/day2_all_in_one.py")
        sys.exit(1)

    # ── Load config ──
    config_path = args.config or CONFIG_PATH
    if not os.path.exists(config_path):
        print(f"\n  [FATAL] config.yaml not found at: {config_path}")
        sys.exit(1)

    print(f"\n  Loading config from: {config_path}", flush=True)
    config = load_config(config_path)
    print(f"  [OK] Config loaded", flush=True)

    # Override output_dir
    config['output_dir'] = RESULTS_DIR

    # Print DB7 info
    db7_cfg = config['datasets'].get(DATASET_KEY, {})
    db7_path = db7_cfg.get('path', 'NOT SET')
    print(f"  DB7 path: {db7_path}")
    if not os.path.exists(db7_path):
        print(f"  [WARNING] DB7 path does not exist! Experiments will fail.")

    t_grand_start = time.time()

    # ── PHASE 1: Window Ablation ──
    if args.phase in ('all', 'window'):
        run_window_ablation(config, load_ninapro_db, process_dataset, args.skip_existing)
        generate_window_summary()

    # ── PHASE 2: Feature Ablation ──
    if args.phase in ('all', 'feature'):
        run_feature_ablation(config, load_ninapro_db, process_dataset, args.skip_existing)
        generate_feature_summary()

    # ── PHASE 3: Statistical Analysis ──
    if args.phase in ('all', 'stats'):
        run_statistical_analysis()

    t_grand = time.time() - t_grand_start

    # ── Final report ──
    print("\n" + "=" * 80)
    print("  DAY 2 COMPLETE!")
    print(f"  Total time: {t_grand:.0f}s ({t_grand/3600:.1f}h {t_grand%3600/60:.0f}m)")
    print(f"  Results in: {RESULTS_DIR}")
    print("=" * 80)
    print("\n  Generated files:")
    print("    Window:   Table_window_ablation.csv/.tex, figure_window_ablation.png")
    print("    Feature:  Table_feature_ablation.csv/.tex, figure_feature_ablation.png")
    print("    Stats:    Table_window_stats.csv/.tex, Table_feature_stats.csv/.tex")
    print("    Raw:      DB7_window_*.json (28), DB7_feat_*.json (10)")
    print("\n" + "=" * 80)

    # Save final summary
    summary_path = os.path.join(RESULTS_DIR, "day2_summary.json")
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': round(t_grand, 1),
        'phases_run': args.phase,
        'results_dir': RESULTS_DIR,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
