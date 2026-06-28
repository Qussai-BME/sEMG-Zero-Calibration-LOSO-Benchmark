#!/usr/bin/env python3
"""
run_all_classifiers.py — Multi-Classifier Comparison under Strict LOSO
=======================================================================

Runs XGBoost, RandomForest, ExtraTrees, and SVM on the SAME hand-crafted
features, under IDENTICAL LOSO protocol. Outputs comparison table + JSON.

Usage:
    # All subjects, multiple datasets:
    python validation/run_all_classifiers.py --config config.yaml --datasets ninapro_db3 ninapro_db2 ninapro_db7

    # Quick test (4 subjects only):
    python validation/run_all_classifiers.py --config config.yaml --datasets ninapro_db3 --subjects 1 2 3 4

    # Specific classifiers only:
    python validation/run_all_classifiers.py --config config.yaml --datasets ninapro_db3 --classifiers XGBoost ExtraTrees
"""

import sys
import os
import gc
import time
import json
import argparse
import numpy as np
from datetime import datetime

# ── Add project root to path ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import yaml

# ── Import existing modules ───────────────────────────────────────────
try:
    from validation.data_loaders import load_ninapro_db
except ImportError:
    from data_loaders import load_ninapro_db

try:
    from validation.process_engine import extract_features_per_channel
except ImportError:
    from process_engine import extract_features_per_channel

try:
    from validation.metrics import evaluate_model
except ImportError:
    from metrics import evaluate_model

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Classifier LOSO Comparison"
    )
    p.add_argument('--config', type=str,
                   default=os.path.join(SCRIPT_DIR, 'config.yaml'))
    p.add_argument('--datasets', nargs='+', required=True,
                   choices=['ninapro_db2', 'ninapro_db3', 'ninapro_db7'])
    p.add_argument('--subjects', nargs='+', type=int, default=None)
    p.add_argument('--classifiers', nargs='+', default=None,
                   help='Classifiers to run (default: all). '
                        'Choices: XGBoost, RandomForest, ExtraTrees, SVM')
    p.add_argument('--output', type=str, default=None,
                   help='Output JSON path (default: ./multi_clf_results.json)')
    return p.parse_args()


# =====================================================================
# Helpers
# =====================================================================
def load_config(path):
    """Load config with fallback: try CWD-relative if absolute path fails."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    # Fallback: try relative to SCRIPT_DIR
    fallback = os.path.join(SCRIPT_DIR, os.path.basename(path))
    if os.path.exists(fallback):
        with open(fallback, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(
        f"Config not found: {path} (also tried {fallback})"
    )


def build_proc_config(base_proc, dataset_key, all_configs):
    """Merge base processing config with dataset-adaptive overrides."""
    proc = dict(base_proc)
    adaptive = all_configs.get('dataset_adaptive_configs', {}).get(dataset_key, {})
    if 'processing' in adaptive:
        for k, v in adaptive['processing'].items():
            proc[k] = v
    return proc


def load_and_extract(db_version, data_path, subjects, proc_cfg, movement_map=None):
    """
    Load raw EMG and extract hand-crafted features for all subjects.

    Returns:
        X : np.ndarray (N_windows, n_features)
        y : np.ndarray (N_windows,)
        groups : np.ndarray (N_windows,)  — subject IDs for LOSO
        n_subjects : int
        feat_names : list
    """
    all_features, all_labels, all_groups = [], [], []
    n_subjects = 0
    feat_names = None

    loader = load_ninapro_db(
        db_version=db_version,
        data_path=data_path,
        subjects=subjects,
        movement_map=movement_map,
        remove_class_zero=False  # handled after aggregation
    )

    for emg, labels, meta in loader:
        subj_id = meta['subject_id']
        print(f"  [Subject {subj_id}] EMG shape: {emg.shape} ...", end=' ', flush=True)
        t0 = time.time()

        # Extract features
        features_flat, windows, snr, feature_names = \
            extract_features_per_channel(emg, proc_cfg)

        # Assign labels to windows (midpoint method)
        if len(windows) > 0:
            starts = np.array([s for s, e in windows])
            ends = np.array([e for s, e in windows])
            mids = np.clip((starts + ends) // 2, 0, len(labels) - 1)
            win_labels = labels[mids]
        else:
            win_labels = np.array([])

        dt = time.time() - t0
        print(f"features: {features_flat.shape} ({dt:.1f}s)", flush=True)

        if feat_names is None:
            feat_names = feature_names

        all_features.append(features_flat)
        all_labels.append(win_labels)
        all_groups.append(np.full(len(win_labels), n_subjects, dtype=np.int32))

        n_subjects += 1
        del emg, labels
        gc.collect()

    if not all_features:
        return None, None, None, 0, None

    # Aggregate
    total_rows = sum(f.shape[0] for f in all_features)
    n_cols = all_features[0].shape[1]
    X = np.empty((total_rows, n_cols), dtype=np.float32)
    y = np.empty(total_rows, dtype=all_labels[0].dtype)
    groups = np.empty(total_rows, dtype=np.int32)
    offset = 0
    for feat, lab, grp in zip(all_features, all_labels, all_groups):
        n = feat.shape[0]
        X[offset:offset + n] = feat
        y[offset:offset + n] = lab
        groups[offset:offset + n] = grp
        offset += n

    del all_features, all_labels, all_groups
    gc.collect()

    return X, y, groups, n_subjects, feat_names


def run_single_classifier(X, y, groups, clf_name, dataset_adaptive_cfg,
                          feat_names=None):
    """Run LOSO with a single classifier and return results dict."""
    print(f"\n{'='*60}", flush=True)
    print(f"  Classifier: {clf_name}", flush=True)
    print(f"{'='*60}", flush=True)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Get adaptive classification settings
    clf_cfg = dataset_adaptive_cfg.get('classification', {})

    t0 = time.time()
    # NOTE: evaluate_model() reads max_depth, learning_rate, subsample,
    # colsample_bytree, early_stopping_rounds from dataset_config automatically.
    # Do NOT pass them as direct kwargs — they are NOT in the function signature.
    result = evaluate_model(
        X, y, groups,
        strategy='loso',
        classifier=clf_name,
        n_estimators=clf_cfg.get('n_estimators', 200),
        n_top_features=clf_cfg.get('n_top_features', 250),
        feature_selection=clf_cfg.get('feature_selection', 'hybrid'),
        return_models=False,
        use_ensemble=False,
        dataset_config=dataset_adaptive_cfg,
        feature_names=feat_names,
    )
    dt = time.time() - t0

    mean_acc, std_acc, cm, per_subj_acc = result[0], result[1], result[2], result[3]

    return {
        'classifier': clf_name,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'per_subject_accuracy': per_subj_acc,
        'confusion_matrix': cm,
        'time_seconds': dt,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()

    # Classifiers to run
    ALL_CLASSIFIERS = ['XGBoost', 'RandomForest', 'ExtraTrees', 'SVM']
    if args.classifiers:
        classifiers = [c for c in args.classifiers if c in ALL_CLASSIFIERS]
        if not classifiers:
            print(f"Unknown classifiers. Choose from: {ALL_CLASSIFIERS}")
            return
    else:
        classifiers = ALL_CLASSIFIERS

    # Load config
    config = load_config(args.config)
    output_path = args.output or os.path.join(PROJECT_ROOT, 'multi_clf_results.json')

    all_results = {}  # {dataset: {classifier: results}}
    summary_table = []

    print("=" * 70, flush=True)
    print("  MULTI-CLASSIFIER COMPARISON UNDER STRICT LOSO", flush=True)
    print(f"  Classifiers: {classifiers}", flush=True)
    print(f"  Datasets:    {args.datasets}", flush=True)
    print(f"  Subjects:    {args.subjects or 'ALL'}", flush=True)
    print(f"  Timestamp:   {datetime.now().isoformat()}", flush=True)
    print("=" * 70, flush=True)

    for ds_key in args.datasets:
        ds_cfg = config.get('datasets', {}).get(ds_key, {})
        data_path = ds_cfg.get('path', '')
        fs = ds_cfg.get('sampling_rate', 2000)
        remove_zero = ds_cfg.get('remove_class_zero', False)

        # Build processing config
        proc_cfg = build_proc_config(config['processing'].copy(), ds_key, config)
        proc_cfg['sampling_rate'] = fs

        # Movement map for DB3
        movement_map = None
        if ds_key == 'ninapro_db3':
            movement_map = config.get('db3_to_db7_movement_map')

        # Subject filter
        subjects = args.subjects

        print(f"\n{'#'*70}", flush=True)
        print(f"  DATASET: {ds_key.upper()}", flush=True)
        print(f"  Path: {data_path}", flush=True)
        print(f"  Subjects: {subjects or 'ALL'}", flush=True)
        print(f"{'#'*70}", flush=True)

        # Load + extract features
        t_load = time.time()
        X, y, groups, n_subjects, feat_names = load_and_extract(
            db_version=ds_key.replace('ninapro_', '').upper(),
            data_path=data_path,
            subjects=subjects,
            proc_cfg=proc_cfg,
            movement_map=movement_map,
        )

        if X is None:
            print(f"  [SKIP] No data for {ds_key}", flush=True)
            continue

        print(f"\n  Aggregated: X={X.shape}, y unique={len(np.unique(y))}, "
              f"subjects={n_subjects} ({time.time()-t_load:.1f}s)", flush=True)

        # Remove class 0 if needed
        if remove_zero:
            mask = y != 0
            X, y, groups = X[mask], y[mask], groups[mask]
            print(f"  Removed class 0: {mask.sum()} samples remaining", flush=True)

        # Get adaptive config for this dataset
        dataset_adaptive = config.get('dataset_adaptive_configs', {}).get(ds_key, {})

        # Run each classifier
        ds_results = {}
        for clf_name in classifiers:
            clf_result = run_single_classifier(
                X, y, groups, clf_name, dataset_adaptive, feat_names
            )
            ds_results[clf_name] = clf_result

            summary_table.append({
                'dataset': ds_key,
                'classifier': clf_name,
                'mean': clf_result['mean_accuracy'],
                'std': clf_result['std_accuracy'],
                'n_subjects': n_subjects,
            })

        all_results[ds_key] = ds_results

        # Free memory
        del X, y, groups
        gc.collect()

    # ── Print Summary Table ────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("  RESULTS SUMMARY", flush=True)
    print("=" * 70, flush=True)

    # Header
    print(f"  {'Dataset':<16} {'Classifier':<16} {'Mean Acc':>10} {'Std':>8} {'Subjects':>8}",
          flush=True)
    print("  " + "-" * 60, flush=True)

    for row in summary_table:
        print(f"  {row['dataset']:<16} {row['classifier']:<16} "
              f"{row['mean']*100:>9.2f}% {row['std']*100:>7.2f}% "
              f"{row['n_subjects']:>8}", flush=True)

    print("  " + "-" * 60, flush=True)

    # Best per dataset
    for ds_key in args.datasets:
        if ds_key in all_results:
            ds_rows = [r for r in summary_table if r['dataset'] == ds_key]
            best = max(ds_rows, key=lambda r: r['mean'])
            print(f"  Best ({ds_key}): {best['classifier']} = "
                  f"{best['mean']*100:.2f} +/- {best['std']*100:.2f}%", flush=True)

    print("=" * 70, flush=True)

    # ── Save JSON ──────────────────────────────────────────────────────
    output = {
        'timestamp': datetime.now().isoformat(),
        'classifiers_run': classifiers,
        'datasets': args.datasets,
        'subjects_filter': args.subjects,
        'summary_table': summary_table,
        'detailed_results': {},
    }

    # Convert numpy types to native Python for JSON serialization
    for ds_key, ds_res in all_results.items():
        output['detailed_results'][ds_key] = {}
        for clf_name, clf_res in ds_res.items():
            output['detailed_results'][ds_key][clf_name] = {
                'mean_accuracy': float(clf_res['mean_accuracy']),
                'std_accuracy': float(clf_res['std_accuracy']),
                'per_subject_accuracy': [float(x) for x in clf_res['per_subject_accuracy']],
                'time_seconds': float(clf_res['time_seconds']),
                # confusion_matrix can be large — skip in JSON for readability
            }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {output_path}", flush=True)


if __name__ == '__main__':
    main()
