#!/usr/bin/env python3
"""
day1_all_in_one.py — THE COMPLETE DAY 1 SOLUTION
=================================================
Does EVERYTHING in ONE run:

1. Patches metrics.py (adds Macro F1 computation)
2. Patches validate_engine.py (saves per-classifier JSON + F1)
3. Runs ALL 4 classifiers × ALL 3 databases automatically
4. Generates ALL statistical analysis files

USAGE (works from anywhere):
  python validation/day1_all_in_one.py   # from project root
  cd validation && python day1_all_in_one.py  # from inside validation/

OUTPUT (in validation/paper1_results/):
  - *_results.json          (per classifier × per database = 12 files)
  - Table2_main_results.csv
  - TableS1_per_subject_results.csv
  - TableS2_statistical_tests.csv
  - TableS3_friedman_test.csv
  - TableS4_holm_sidak_correction.csv
  - tables_paper.tex
  - statistical_summary.txt

Author: Paper 1 — Day 1 Complete Automation
"""

import os
import re
import sys
import shutil
import json
import csv
import time
import traceback
import numpy as np
from scipy import stats
from itertools import combinations

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Auto-detect: script inside validation/ or in project root? ──
if os.path.exists(os.path.join(SCRIPT_DIR, "metrics.py")):
    # Script is INSIDE validation/ folder
    VALIDATION_DIR = SCRIPT_DIR
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    # Script is in project root
    PROJECT_ROOT = SCRIPT_DIR
    VALIDATION_DIR = os.path.join(PROJECT_ROOT, "validation")

METRICS_PATH = os.path.join(VALIDATION_DIR, "metrics.py")
VALIDATE_PATH = os.path.join(VALIDATION_DIR, "validate_engine.py")
CONFIG_PATH = os.path.join(VALIDATION_DIR, "config.yaml")
RESULTS_DIR = os.path.join(VALIDATION_DIR, "paper1_results")
BACKUP_DIR = os.path.join(VALIDATION_DIR, "paper1_backups")

CLASSIFIERS = ['XGBoost', 'LDA', 'LinearSVC', 'RandomForest']
DATASETS = ['ninapro_db7', 'ninapro_db3', 'ninapro_db2']
DB_LABELS = {
    'ninapro_db7': 'DB7 (mixed, 22 subjects)',
    'ninapro_db3': 'DB3 (amputee, 11 subjects)',
    'ninapro_db2': 'DB2 (intact, 40 subjects)',
}
DB_SHORT = {
    'ninapro_db7': 'DB7',
    'ninapro_db3': 'DB3',
    'ninapro_db2': 'DB2',
}


# ============================================================================
# STEP 1: PATCH metrics.py — Add Macro F1 (v2 — regex-based, robust)
# ============================================================================
def patch_metrics():
    """Patch metrics.py to compute Macro F1 per fold. Uses regex for resilience."""
    print("\n" + "=" * 60)
    print("STEP 1/3: Patching metrics.py (adding Macro F1)...")
    print("=" * 60)

    if not os.path.exists(METRICS_PATH):
        print(f"  [ERROR] {METRICS_PATH} not found!")
        return False

    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    applied = 0

    def _sub(pattern, replacement, flags=0):
        """Regex sub — uses re.subn to properly resolve backreferences like \\1."""
        nonlocal content, applied
        new_content, n = re.subn(pattern, replacement, content, count=1, flags=flags)
        if n > 0:
            content = new_content
            applied += n
            return True
        return False

    # ── 1a: Add f1_score import ──
    if 'f1_score' not in re.split(r'from sklearn\.metrics import', content)[0][-100:]:
        if _sub(
            r'from sklearn\.metrics import confusion_matrix, accuracy_score',
            'from sklearn.metrics import confusion_matrix, accuracy_score, f1_score'
        ):
            print("  [OK] 1a: Added f1_score import")
        else:
            print("  [SKIP] 1a: import pattern not found")
    else:
        print("  [SKIP] 1a: f1_score already imported")

    # ── 1b: Add macro_f1 in _loso_fold_worker return ──
    if 'macro_f1' not in re.split(r'return _yt, _yp, acc, subject_id', content)[-1][:30]:
        # Add computation before return, add to return value
        if _sub(
            r"(    acc = float\(accuracy_score\(y_test, y_pred\)\))\n",
            r"\1\n    macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))\n"
        ):
            print("  [OK] 1b-i: Added macro_f1 computation")
        if _sub(
            r'return _yt, _yp, acc, subject_id, None, None\n',
            'return _yt, _yp, acc, subject_id, None, None, macro_f1\n'
        ):
            print("  [OK] 1b-ii: Updated _loso_fold_worker return (7 values)")
    else:
        print("  [SKIP] 1b: macro_f1 already in fold worker")

    # ── 1c: LOSO section — init + collect per_subject_f1 ──
    # Change: for yt, yp, acc, subj, model, Xt in results → add mf1
    # Also add per_subject_f1 = [] before the loop
    if 'per_subject_f1' not in content or content.count('per_subject_f1 = []') < 1:
        # Find the LOSO for-loop and add per_subject_f1 init before it
        loso_for_re = (
            r'(    y_true_all, y_pred_all, accs = \[\], \[\], \[\]\n'
            r'    per_subject_acc = \[\]\n'
            r'    trained_models, X_tests = \[\], \[\]\n\n'
            r'    for) yt, yp, acc, subj, model, Xt in results:'
        )
        if _sub(
            loso_for_re,
            r"\1 yt, yp, acc, subj, model, Xt, mf1 in results:",
            flags=re.MULTILINE
        ):
            print("  [OK] 1c-i: LOSO loop unpacks 7 values")

        # Add per_subject_f1 = [] init (in LOSO section only)
        loso_init_pattern = (
            r'(    per_subject_acc = \[\]\n)'
            r'(    trained_models, X_tests = \[\], \[\])'
        )
        if _sub(
            loso_init_pattern,
            r'\1    per_subject_f1 = []\n\2'
        ):
            print("  [OK] 1c-ii: LOSO init per_subject_f1 = []")

        # Add per_subject_f1.append in LOSO loop
        if _sub(
            r"(        per_subject_acc\.append\(\{'subject': subj, 'accuracy': float\(acc\)\}\)\n)"
            r"(        if return_models and model is not None:)",
            r"\1        per_subject_f1.append({'subject': subj, 'macro_f1': float(mf1)})\n\2"
        ):
            print("  [OK] 1c-iii: LOSO collects per_subject_f1")

    # ── 1d: LOSO return — include per_subject_f1 ──
    # The LOSO return is followed by blank lines then # Within-subject section
    if _sub(
        r'(    return y_true_all, y_pred_all, accs, per_subject_acc, trained_models, X_tests)'
        r'(\n\n+)(# =+.*?Within-subject)',
        r'    return y_true_all, y_pred_all, accs, per_subject_acc, per_subject_f1, trained_models, X_tests\2\3',
        flags=re.DOTALL
    ):
        print("  [OK] 1d: LOSO return includes per_subject_f1")
    else:
        print("  [SKIP] 1d: LOSO return pattern not found")

    # ── 1e: Within-subject section — init per_subject_f1 ──
    # Find the SECOND occurrence of the init block (first is LOSO, already patched)
    ws_init_pattern = (
        r'(    per_subject_acc = \[\]\n)'
        r'(    trained_models, X_tests = \[\], \[\])'
    )
    count_init = content.count('per_subject_acc = []')
    if count_init < 2 or 'per_subject_f1 = []' not in content.split('Within-subject')[1][:300] if 'Within-subject' in content else True:
        # Add per_subject_f1 = [] in within-subject section
        # Use the second occurrence of the pattern
        parts = content.split('# Within-subject')
        if len(parts) >= 2:
            ws_part = parts[1]
            ws_init_re = r'(    per_subject_acc = \[\]\n)(    trained_models, X_tests = \[\], \[\])'
            ws_new, n = re.subn(ws_init_re, r'\1    per_subject_f1 = []\n\2', ws_part, count=1)
            if n > 0:
                content = parts[0] + '# Within-subject' + ws_new
                applied += 1
                print("  [OK] 1e: Within-subject init per_subject_f1 = []")

    # ── 1f: Within-subject — collect macro_f1 per subject ──
    ws_f1_pattern = r"        per_subject_acc\.append\(\{'subject': int\(subj\), 'accuracy': float\(acc\)\}\)"
    if _sub(
        ws_f1_pattern + r'\n',
        ("        per_subject_acc.append({'subject': int(subj), 'accuracy': float(acc)})\n"
         "        _f1_val = float(f1_score(y_test, y_pred, average='macro', zero_division=0))\n"
         "        per_subject_f1.append({'subject': int(subj), 'macro_f1': _f1_val})\n")
    ):
        print("  [OK] 1f: Within-subject collects macro_f1")
    else:
        print("  [SKIP] 1f: within-subject append pattern not found")

    # ── 1g: Within-subject return — include per_subject_f1 ──
    # After within-subject loop comes: return ... then # Main Public API section
    ws_sections = content.split('# Main Public API')
    if len(ws_sections) >= 2:
        ws_code = ws_sections[0]
        ws_return_re = r'(    return y_true_all, y_pred_all, accs, per_subject_acc, trained_models, X_tests)(\n)'
        ws_new, n = re.subn(ws_return_re,
                             r'    return y_true_all, y_pred_all, accs, per_subject_acc, per_subject_f1, trained_models, X_tests\2',
                             ws_code, count=1)
        if n > 0:
            content = ws_new + '# Main Public API' + ws_sections[1]
            applied += 1
            print("  [OK] 1g: Within-subject return includes per_subject_f1")
        else:
            print("  [SKIP] 1g: within-subject return not found")
    else:
        print("  [SKIP] 1g: '# Main Public API' section not found")

    # ── 1h: evaluate_model — LOSO unpack 7 values ──
    if _sub(
        r'        \(y_true_all, y_pred_all, accs, per_subject_acc,\n         trained_models, X_tests\) = _evaluate_loso_parallel\(',
        '        (y_true_all, y_pred_all, accs, per_subject_acc, per_subject_f1,\n         trained_models, X_tests) = _evaluate_loso_parallel('
    ):
        print("  [OK] 1h: evaluate_model unpacks 7 values (LOSO)")

    # ── 1i: evaluate_model — within-subject unpack 7 values ──
    if _sub(
        r'        \(y_true_all, y_pred_all, accs, per_subject_acc,\n         trained_models, X_tests\) = _evaluate_within_subject\(',
        '        (y_true_all, y_pred_all, accs, per_subject_acc, per_subject_f1,\n         trained_models, X_tests) = _evaluate_within_subject('
    ):
        print("  [OK] 1i: evaluate_model unpacks 7 values (within-subject)")

    # ── 1j: evaluate_model — return per_subject_f1 in both paths ──
    if _sub(
        r'(        return mean_acc, std_acc, cm, acc_values, )\[\], \[\]',
        r'\1per_subject_f1, [], []'
    ):
        print("  [OK] 1j-i: evaluate_model return (no models) includes per_subject_f1")
    if _sub(
        r'(    return mean_acc, std_acc, cm, acc_values, )trained_models, X_tests',
        r'\1per_subject_f1, trained_models, X_tests'
    ):
        print("  [OK] 1j-ii: evaluate_model return (with models) includes per_subject_f1")

    if applied > 0:
        with open(METRICS_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n  >>> metrics.py: {applied} patches applied successfully")
    else:
        print("\n  >>> metrics.py: already fully patched (no changes needed)")

    return True


# ============================================================================
# STEP 2: PATCH validate_engine.py — Save per-classifier JSON
# ============================================================================
def patch_validate_engine():
    """Patch validate_engine.py to save per-classifier JSON with F1."""
    print("\n" + "=" * 60)
    print("STEP 2/3: Patching validate_engine.py (per-classifier JSON)...")
    print("=" * 60)

    if not os.path.exists(VALIDATE_PATH):
        print(f"  [ERROR] {VALIDATE_PATH} not found!")
        return False

    with open(VALIDATE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    applied = 0

    # Patch 2a: Unpack 7 values (line 655)
    search2a = "        acc, std, cm, per_subject_acc, trained_models, X_tests = result"
    if search2a in content:
        content = content.replace(
            search2a,
            "        acc, std, cm, per_subject_acc, per_subject_f1, trained_models, X_tests = result",
            1
        )
        applied += 1
        print("  [OK] Patch 2a: Unpacks 7 values from evaluate_model")

    # Patch 2b: Add per_subject_macro_f1 to output_data
    search2b = "        'per_subject_accuracy': per_subject_acc,"
    if search2b in content and "'per_subject_macro_f1'" not in content:
        content = content.replace(
            search2b,
            "        'per_subject_accuracy': per_subject_acc,\n        'per_subject_macro_f1': per_subject_f1,",
            1
        )
        applied += 1
        print("  [OK] Patch 2b: Adds per_subject_macro_f1 to output_data")

    # Patch 2c: Add classifier name to output_data
    search2c = "        'classification': classification_result,"
    if search2c in content and "'classifier_name'" not in content:
        content = content.replace(
            search2c,
            "        'classifier': clf_cfg.get('classifier', 'unknown'),\n        'classification': classification_result,",
            1
        )
        applied += 1
        print("  [OK] Patch 2c: Adds classifier name to output_data")

    # Patch 2d: Save per-classifier JSON file
    # Insert JSON saving code after output_data is built, before the return
    search2d = "    return output_data, trained_models, X_tests, feat_names"
    json_save_code = """    # ── Save per-classifier JSON (Day 1 automation) ──
    clf_name = clf_cfg.get('classifier', 'unknown').lower()
    json_filename = f"{dataset_name}_{clf_name}_results.json"
    json_path = os.path.join(output_dir, json_filename)
    try:
        with open(json_path, 'w', encoding='utf-8') as _jf:
            json.dump(output_data, _jf, indent=2, default=str)
        print(f"[saved] {json_path}", flush=True)
    except Exception as _je:
        print(f"[warn] Failed to save JSON: {_je}", flush=True)

    """
    if search2d in content and "paper1_results" not in content:
        # Also redirect output to paper1_results
        content = content.replace(
            search2d,
            json_save_code + "return output_data, trained_models, X_tests, feat_names",
            1
        )
        applied += 1
        print("  [OK] Patch 2d: Saves per-classifier JSON")

    if applied > 0:
        with open(VALIDATE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n  >>> validate_engine.py: {applied} patches applied")
    else:
        print("\n  >>> validate_engine.py: already patched")

    return True


# ============================================================================
# STEP 3: RUN ALL EXPERIMENTS
# ============================================================================
def run_all_experiments():
    """Run all 4 classifiers × 3 databases."""
    print("\n" + "=" * 60)
    print("STEP 3/3: Running all experiments...")
    print("=" * 60)
    print(f"  Classifiers: {CLASSIFIERS}")
    print(f"  Databases:   {DATASETS}")
    print(f"  Total:       {len(CLASSIFIERS)} × {len(DATASETS)} = {len(CLASSIFIERS) * len(DATASETS)} runs")
    print()

    # Import validate_engine
    sys.path.insert(0, PROJECT_ROOT)
    try:
        from validation.validate_engine import load_config, process_dataset, main
    except ImportError:
        from validate_engine import load_config, process_dataset, main

    if not os.path.exists(CONFIG_PATH):
        print(f"  [ERROR] config.yaml not found at: {CONFIG_PATH}")
        return {}

    config = load_config(CONFIG_PATH)

    # Override output_dir to paper1_results
    config['output_dir'] = RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Import data loaders
    sys.path.insert(0, VALIDATION_DIR)
    try:
        from data_loaders import load_ninapro_db
    except ImportError:
        from validation.data_loaders import load_ninapro_db

    total_start = time.time()
    results = {}  # {(clf, db): output_data}

    for clf_name in CLASSIFIERS:
        for ds_key in DATASETS:
            run_start = time.time()
            print(f"\n{'='*60}")
            print(f"  RUNNING: {clf_name} × {ds_key.upper()}")
            print(f"{'='*60}", flush=True)

            # Set classifier in config
            config['classification']['classifier'] = clf_name

            # Build dataset loader
            db_version = ds_key.replace('ninapro_', '').upper()
            dataset_cfg = config['datasets'].get(ds_key, {})
            path = dataset_cfg.get('path', '')
            movement_map = None
            if db_version == 'DB3':
                movement_map = config.get('db3_to_db7_movement_map')

            try:
                loader = load_ninapro_db(
                    db_version=db_version,
                    data_path=path,
                    subjects=None,
                    movement_map=movement_map,
                    remove_class_zero=False
                )

                # Create dummy args
                class Args:
                    pass
                args = Args()
                args.quick = False
                args.resume = False
                args.subjects = None

                # Use process_dataset directly (it calls evaluate_model internally)
                result = process_dataset(
                    loader, f'Ninapro_{db_version}', ds_key, config,
                    checkpoint=None, quick=False, resume=False
                )

                if result is not None:
                    res, models, X_tests, feat_names = result
                    results[(clf_name, ds_key)] = res
                    print(f"\n  [DONE] {clf_name} × {db_version}: "
                          f"Acc={res.get('classification', (0,0,[]))[0]:.4f} "
                          f"({time.time() - run_start:.1f}s)", flush=True)
                else:
                    print(f"\n  [FAIL] {clf_name} × {db_version}: No results", flush=True)

            except Exception as e:
                print(f"\n  [ERROR] {clf_name} × {db_version}: {e}", flush=True)
                traceback.print_exc()

            # Save intermediate results after each run
            _save_intermediate_json(results)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS DONE in {total_time/60:.1f} minutes")
    print(f"  Successful: {len(results)}/{len(CLASSIFIERS)*len(DATASETS)}")
    print(f"{'='*60}")

    return results


def _save_intermediate_json(results):
    """Save intermediate results so we don't lose data on crash."""
    summary = {}
    for (clf, db), res in results.items():
        clf_data = res.get('classification', (0, 0, []))
        summary[f"{clf}_{db}"] = {
            'accuracy': clf_data[0] if clf_data else 0,
            'std': clf_data[1] if clf_data else 0,
            'n_subjects': len(res.get('per_subject_accuracy', [])),
        }
    path = os.path.join(RESULTS_DIR, "_run_progress.json")
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)


# ============================================================================
# STEP 4: STATISTICAL ANALYSIS
# ============================================================================
def run_statistical_analysis():
    """Compute all statistical tests from saved JSON results."""
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    # Load all JSON results
    all_results = {}  # {(clf, db): {'per_subject_accuracy': [...], 'per_subject_macro_f1': [...]}}

    for clf in CLASSIFIERS:
        for ds in DATASETS:
            db_version = ds.replace('ninapro_', '').upper()
            json_file = os.path.join(RESULTS_DIR, f"Ninapro_{db_version}_{clf.lower()}_results.json")

            if not os.path.exists(json_file):
                print(f"  [MISSING] {json_file}")
                continue

            with open(json_file, 'r') as f:
                data = json.load(f)

            all_results[(clf, ds)] = data
            n_subj = len(data.get('per_subject_accuracy', []))
            n_f1 = len(data.get('per_subject_macro_f1', []))
            print(f"  Loaded: {clf:12s} × {ds:15s} ({n_subj} acc, {n_f1} f1)")

    if len(all_results) < 4:
        print(f"\n  [ERROR] Only {len(all_results)} results found. Need at least 4.")
        print("  Cannot run statistical analysis without real data.")
        return False

    # Compute statistics
    print(f"\n  Computing statistics for {len(all_results)} classifier-database pairs...")

    # === MAIN RESULTS TABLE ===
    main_rows = []
    for ds in DATASETS:
        for clf in CLASSIFIERS:
            key = (clf, ds)
            if key not in all_results:
                continue
            data = all_results[key]

            acc_list = [item['accuracy'] for item in data.get('per_subject_accuracy', [])]
            f1_list = [item['macro_f1'] for item in data.get('per_subject_macro_f1', [])]

            if not acc_list:
                continue

            acc_arr = np.array(acc_list)
            f1_arr = np.array(f1_list) if f1_list else np.zeros_like(acc_arr)

            acc_mean = np.mean(acc_arr)
            acc_std = np.std(acc_arr, ddof=1) if len(acc_arr) > 1 else 0
            f1_mean = np.mean(f1_arr)
            f1_std = np.std(f1_arr, ddof=1) if len(f1_arr) > 1 else 0

            # 95% CI (t-distribution)
            n = len(acc_arr)
            if n > 1:
                t_val = stats.t.ppf(0.975, n - 1)
                ci_lo = acc_mean - t_val * acc_std / np.sqrt(n)
                ci_hi = acc_mean + t_val * acc_std / np.sqrt(n)
            else:
                ci_lo, ci_hi = acc_mean, acc_mean

            main_rows.append({
                'Classifier': clf,
                'Database': ds,
                'DB_Short': DB_SHORT[ds],
                'DB_Label': DB_LABELS[ds],
                'N_Subjects': n,
                'Acc_Mean': acc_mean,
                'Acc_Std': acc_std,
                'Acc_CI_Lo': ci_lo,
                'Acc_CI_Hi': ci_hi,
                'F1_Mean': f1_mean,
                'F1_Std': f1_std,
            })

    # === PAIRWISE TESTS ===
    test_rows = []
    for ds in DATASETS:
        for clf_a, clf_b in combinations(CLASSIFIERS, 2):
            key_a = (clf_a, ds)
            key_b = (clf_b, ds)
            if key_a not in all_results or key_b not in all_results:
                continue

            acc_a = [item['accuracy'] for item in all_results[key_a].get('per_subject_accuracy', [])]
            acc_b = [item['accuracy'] for item in all_results[key_b].get('per_subject_accuracy', [])]
            f1_a = [item['macro_f1'] for item in all_results[key_a].get('per_subject_macro_f1', [])]
            f1_b = [item['macro_f1'] for item in all_results[key_b].get('per_subject_macro_f1', [])]

            if not acc_a or not acc_b:
                continue

            # Wilcoxon (accuracy)
            try:
                W, p = stats.wilcoxon(acc_a, acc_b, zero_method='wilcox', alternative='two-sided')
            except (ValueError, Exception):
                W, p = 0.0, 1.0

            # Effect size r
            diff = np.array(acc_a) - np.array(acc_b)
            n_nz = np.count_nonzero(diff)
            if n_nz > 0:
                r = 1 - (2 * W) / (n_nz * (n_nz + 1))
            else:
                r = 0.0

            # Cohen's d (paired)
            if len(diff) > 1 and np.std(diff, ddof=1) > 0:
                d = np.mean(diff) / np.std(diff, ddof=1)
            else:
                d = 0.0

            # Interpret
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            d_mag = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"

            # Wilcoxon on F1
            try:
                W_f1, p_f1 = stats.wilcoxon(f1_a, f1_b, zero_method='wilcox', alternative='two-sided')
            except (ValueError, Exception):
                W_f1, p_f1 = 0.0, 1.0
            f1_sig = "***" if p_f1 < 0.001 else "**" if p_f1 < 0.01 else "*" if p_f1 < 0.05 else "n.s."

            test_rows.append({
                'Database': ds, 'DB_Short': DB_SHORT[ds],
                'Classifier_A': clf_a, 'Classifier_B': clf_b,
                'N': len(acc_a),
                'Diff_Acc': np.mean(acc_a) - np.mean(acc_b),
                'Wilcoxon_W': W, 'Wilcoxon_p': p, 'Wilcoxon_r': r,
                'Wilcoxon_Sig': sig, 'Cohens_d': d, 'Cohens_d_Interp': d_mag,
                'F1_Wilcoxon_p': p_f1, 'F1_Wilcoxon_Sig': f1_sig,
            })

    # === FRIEDMAN TEST ===
    friedman_rows = []
    for ds in DATASETS:
        groups = []
        for clf in CLASSIFIERS:
            key = (clf, ds)
            if key in all_results:
                groups.append([item['accuracy'] for item in all_results[key].get('per_subject_accuracy', [])])
        if len(groups) >= 3:
            try:
                chi2, p = stats.friedmanchisquare(*groups)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                friedman_rows.append({
                    'Database': ds, 'DB_Label': DB_LABELS[ds],
                    'Chi2': chi2, 'p': p, 'Sig': sig, 'N_Classifiers': len(groups),
                })
            except (ValueError, Exception):
                pass

    # === HOLM-SIDAK CORRECTION ===
    holm_rows = []
    for ds in DATASETS:
        db_tests = [r for r in test_rows if r['Database'] == ds]
        p_values = [r['Wilcoxon_p'] for r in db_tests]
        n = len(p_values)

        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adj_p = [None] * n
        for rank, (orig_idx, p) in enumerate(indexed):
            alpha_adj = 1 - (1 - 0.05) ** (1 / (n - rank))
            adj_p[orig_idx] = min(1.0, p * (n - rank))
            sig = "***" if adj_p[orig_idx] < 0.001 else "**" if adj_p[orig_idx] < 0.01 else "*" if adj_p[orig_idx] < 0.05 else "n.s."
            comp = f"{db_tests[orig_idx]['Classifier_A']} vs {db_tests[orig_idx]['Classifier_B']}"
            holm_rows.append({
                'Database': ds, 'DB_Short': DB_SHORT[ds],
                'Comparison': comp, 'Raw_p': p, 'Adj_p': adj_p[orig_idx],
                'Rejected': bool(p <= alpha_adj), 'Sig': sig,
            })

    # === PER-SUBJECT CSV ===
    subject_rows = []
    for ds in DATASETS:
        for clf in CLASSIFIERS:
            key = (clf, ds)
            if key not in all_results:
                continue
            data = all_results[key]
            acc_items = data.get('per_subject_accuracy', [])
            f1_items = data.get('per_subject_macro_f1', [])
            for i, item in enumerate(acc_items):
                f1_val = f1_items[i]['macro_f1'] if i < len(f1_items) else 0.0
                subject_rows.append({
                    'Classifier': clf, 'Database': DB_SHORT[ds],
                    'Subject': item.get('subject', i + 1),
                    'Accuracy': item['accuracy'],
                    'Macro_F1': f1_val,
                })

    # === SAVE ALL FILES ===
    print("\n  Saving output files...")

    def save_csv(rows, filepath):
        if not rows:
            return
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"    Saved: {filepath} ({len(rows)} rows)")

    save_csv(main_rows, os.path.join(RESULTS_DIR, "Table2_main_results.csv"))
    save_csv(subject_rows, os.path.join(RESULTS_DIR, "TableS1_per_subject_results.csv"))
    save_csv(test_rows, os.path.join(RESULTS_DIR, "TableS2_statistical_tests.csv"))
    save_csv(friedman_rows, os.path.join(RESULTS_DIR, "TableS3_friedman_test.csv"))
    save_csv(holm_rows, os.path.join(RESULTS_DIR, "TableS4_holm_sidak_correction.csv"))

    # LaTeX tables
    latex = _generate_latex(main_rows, test_rows, friedman_rows, holm_rows)
    with open(os.path.join(RESULTS_DIR, "tables_paper.tex"), 'w') as f:
        f.write(latex)
    print(f"    Saved: tables_paper.tex")

    # Summary
    summary = _generate_summary(main_rows, test_rows, friedman_rows, holm_rows)
    with open(os.path.join(RESULTS_DIR, "statistical_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"    Saved: statistical_summary.txt")

    print(f"\n{summary}")
    return True


def _generate_latex(main_rows, test_rows, friedman_rows, holm_rows):
    """Generate LaTeX tables."""
    lines = []

    # Table 2: Main Results
    lines.append(r"% ===== TABLE 2: Main LOSO Results =====")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\caption{LOSO cross-validation results. Accuracy and Macro $F_1$ as "
                 r"mean $\pm$ std with 95\% CI. Best per database in \textbf{bold}.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Classifier & Database & $N$ & Accuracy (\%) & Macro $F_1$ (\%) & 95\% CI \\")
    lines.append(r"\midrule")

    for db in DATASETS:
        db_rows = [r for r in main_rows if r['Database'] == db]
        if not db_rows:
            continue
        best = max(db_rows, key=lambda x: x['Acc_Mean'])
        lines.append(r"\multicolumn{6}{l}{" + DB_LABELS[db] + r"} \\")
        lines.append(r"\midrule")
        for row in db_rows:
            is_best = abs(row['Acc_Mean'] - best['Acc_Mean']) < 0.0001
            acc_s = f"{row['Acc_Mean']*100:.2f} $\\pm$ {row['Acc_Std']*100:.2f}"
            f1_s = f"{row['F1_Mean']*100:.2f} $\\pm$ {row['F1_Std']*100:.2f}"
            ci_s = f"[{row['Acc_CI_Lo']*100:.2f}, {row['Acc_CI_Hi']*100:.2f}]"
            if is_best:
                lines.append(f"\\textbf{{{row['Classifier']}}} & {row['DB_Short']} & {row['N_Subjects']} "
                             f"& \\textbf{{{acc_s}}} & \\textbf{{{f1_s}}} & {ci_s} \\\\ ")
            else:
                lines.append(f"{row['Classifier']} & {row['DB_Short']} & {row['N_Subjects']} "
                             f"& {acc_s} & {f1_s} & {ci_s} \\\\ ")
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append("")

    # Table 3: Friedman
    if friedman_rows:
        lines.append(r"% ===== TABLE 3: Friedman Test =====")
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\caption{Friedman test for overall classifier differences per database.}")
        lines.append(r"\label{tab:friedman}")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"Database & $N$ & $\chi^{2}_r$ & $p$-value & Significance \\")
        lines.append(r"\midrule")
        for row in friedman_rows:
            n = next((r['N_Subjects'] for r in main_rows if r['Database'] == row['Database']), '?')
            lines.append(f"{row['DB_Label']} & {n} & {row['Chi2']:.3f} & {row['p']:.4f} & {row['Sig']} \\\\ ")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    return "\n".join(lines)


def _generate_summary(main_rows, test_rows, friedman_rows, holm_rows):
    """Generate console summary."""
    lines = []
    lines.append("=" * 72)
    lines.append("  DAY 1 STATISTICAL ANALYSIS — PAPER 1 (REAL DATA)")
    lines.append("  LOSO Cross-Validation: 4 Classifiers × 3 Databases")
    lines.append("=" * 72)

    for db in DATASETS:
        db_rows = [r for r in main_rows if r['Database'] == db]
        if not db_rows:
            continue
        best = max(db_rows, key=lambda x: x['Acc_Mean'])
        gap = (best['Acc_Mean'] - min(db_rows, key=lambda x: x['Acc_Mean'])['Acc_Mean']) * 100

        lines.append(f"\n  {DB_LABELS[db]}:")
        lines.append(f"  {'Classifier':<15} {'Accuracy':>22} {'Macro F1':>22} {'95% CI':>24}")
        lines.append(f"  {'-'*15} {'-'*22} {'-'*22} {'-'*24}")
        for row in db_rows:
            marker = " *" if abs(row['Acc_Mean'] - best['Acc_Mean']) < 0.0001 else "  "
            lines.append(f"  {row['Classifier']:<15}{marker} "
                         f"{row['Acc_Mean']*100:.2f}+/-{row['Acc_Std']*100:.2f}%   "
                         f"{row['F1_Mean']*100:.2f}+/-{row['F1_Std']*100:.2f}%   "
                         f"[{row['Acc_CI_Lo']*100:.2f}%, {row['Acc_CI_Hi']*100:.2f}%]")
        lines.append(f"\n  Gap: {gap:.2f}%")

    if friedman_rows:
        lines.append(f"\n{'='*72}")
        lines.append("  FRIEDMAN TEST")
        lines.append(f"{'='*72}")
        for row in friedman_rows:
            lines.append(f"  {row['DB_Label']:40s}  X2={row['Chi2']:7.3f}  p={row['p']:.4f} ({row['Sig']})")

    lines.append(f"\n{'='*72}")
    lines.append("  ALL FILES SAVED TO:", RESULTS_DIR)
    lines.append("=" * 72)

    return "\n".join(lines)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    print("=" * 70)
    print("  DAY 1 ALL-IN-ONE: Patch → Run Experiments → Statistical Analysis")
    print("  Paper 1: LOSO Cross-Validation Benchmark")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Backup original files
    for src, name in [(METRICS_PATH, "metrics.py"), (VALIDATE_PATH, "validate_engine.py")]:
        if os.path.exists(src):
            dst = os.path.join(BACKUP_DIR, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  Backed up {name}")

    # Step 1: Patch metrics.py
    ok1 = patch_metrics()

    # Step 2: Patch validate_engine.py
    ok2 = patch_validate_engine()

    if not ok1 or not ok2:
        print("\n  [WARNING] Some patches failed. Check messages above.")

    # Step 3: Run experiments
    print("\n" + "=" * 70)
    print("  READY TO RUN ALL EXPERIMENTS")
    print("=" * 70)
    print(f"  Output directory: {RESULTS_DIR}")
    print(f"  Backup directory: {BACKUP_DIR}")
    print()

    results = run_all_experiments()

    # Step 4: Statistical analysis
    if results and len(results) >= 4:
        run_statistical_analysis()
    else:
        print("\n  [INFO] Less than 4 experiments completed.")
        print("  Running analysis on whatever JSON files are available...")
        run_statistical_analysis()

    print("\n" + "=" * 70)
    print("  DONE! Check:", RESULTS_DIR)
    print("=" * 70)


if __name__ == '__main__':
    main()
