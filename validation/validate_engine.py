#!/usr/bin/env python3
"""
validate_engine.py — v12.0 (SELECTIVE INTELLIGENCE — Adaptive Engine)

v12.0 changes:
  - v33.0: class_zero removed ONLY in process_dataset (not in loaders)
  - v33.0: UCI sampling_rate enforced from config (no inference drift)
  - v33.0: process_one_subject always passes remove_class_zero=False to loaders

v11.0 changes preserved:
  - Applies dataset_adaptive_configs from config.yaml per dataset
  - Passes adaptive processing overrides (overlap, corr_channels, etc.)
  - Passes adaptive classification settings to evaluate_model
  - Passes feature_names through pipeline for correlation protection
  - Supports use_ensemble flag per dataset
  - Passes dataset_config to metrics.evaluate_model for full adaptive behavior

Critical fixes preserved from v10.3:
  1. FIXED: n_jobs NOT passed to evaluate_model
  2. FIXED: return_models=True wasted ~3GB on DB7/DB2
  3. Works with metrics.py v33.0 (Hybrid FS + Adaptive Engine)
"""

import sys
import os
import time
import gc
import argparse
import yaml
import pickle
import hashlib
import json
import traceback
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Try to import from validation package, fallback to direct imports
try:
    from validation.checkpoint import Checkpoint
except ImportError:
    Checkpoint = None

try:
    from validation.data_loaders import (
        load_uci_gesture, load_ninapro_db,
        load_cemhsey, load_uci_physical_action
    )
except ImportError:
    from data_loaders import (
        load_uci_gesture, load_ninapro_db,
        load_cemhsey, load_uci_physical_action
    )

try:
    from validation.process_engine import (
        extract_features_per_channel, verify_feature_dimension
    )
except ImportError:
    from process_engine import (
        extract_features_per_channel, verify_feature_dimension
    )

try:
    from validation.metrics import evaluate_model, feature_statistics
except ImportError:
    from metrics import evaluate_model, feature_statistics

try:
    from validation.report_generator import generate_report
except ImportError:
    try:
        from report_generator import generate_report
    except ImportError:
        generate_report = None


# =====================================================================
# CLI argument parsing
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EMG Analysis Engine v11.0 — Adaptive Validation Suite"
    )
    default_config = os.path.join(os.path.dirname(__file__), 'config.yaml')
    parser.add_argument('--config', type=str, default=default_config)
    parser.add_argument(
        '--datasets', nargs='+',
        choices=['uci', 'ninapro_db2', 'ninapro_db3', 'ninapro_db7',
                 'cemhsey', 'uci_physical'],
        required=True
    )
    parser.add_argument(
        '--subjects', nargs='+', type=int, default=None,
        help='Filter to specific subject IDs (e.g., --subjects 1 2 3 4 5)'
    )
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--cross-db', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_config_hash(proc_config):
    """Generate hash of processing config for cache invalidation."""
    config_str = json.dumps(proc_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


# =====================================================================
# v11.0: Build adaptive processing config for a dataset
# =====================================================================

def _build_adaptive_proc_config(base_proc, dataset_key, all_configs):
    """
    Merge base processing config with dataset-adaptive overrides.

    This allows each dataset to have its own overlap, corr_channels,
    downsample settings, etc. without breaking the shared pipeline.
    """
    proc = dict(base_proc)

    adaptive = all_configs.get('dataset_adaptive_configs', {}).get(dataset_key, {})
    if 'processing' in adaptive:
        for k, v in adaptive['processing'].items():
            proc[k] = v

    return proc


def _build_adaptive_clf_config(base_clf, dataset_key, all_configs):
    """
    Merge base classification config with dataset-adaptive overrides.
    Returns only the classification-related keys that evaluate_model accepts.
    """
    clf = dict(base_clf)

    adaptive = all_configs.get('dataset_adaptive_configs', {}).get(dataset_key, {})
    if 'classification' in adaptive:
        for k, v in adaptive['classification'].items():
            clf[k] = v

    # Also check legacy dataset_clf_overrides
    legacy = all_configs.get('dataset_clf_overrides', {}).get(dataset_key, {})
    clf.update(legacy)

    return clf


# =====================================================================
# Per-subject processing (called by parallel workers or sequentially)
# =====================================================================

def process_one_subject(args):
    """
    Process a single subject: load data, extract features, return results.
    """
    (subject_key, dataset_name, data_path, day,
     proc_config, cache_dir, disable_cache) = args

    try:
        _t0 = time.time()
        config_hash = get_config_hash(proc_config)

        # ── Check cache ─────────────────────────────────────────────────
        if not disable_cache:
            cache_file = os.path.join(
                cache_dir, f"{subject_key}_{config_hash}_features.npy"
            )
            label_file = os.path.join(
                cache_dir, f"{subject_key}_{config_hash}_labels.npy"
            )
            meta_file = os.path.join(
                cache_dir, f"{subject_key}_{config_hash}_meta.pkl"
            )

            if all(os.path.exists(x) for x in [cache_file, label_file, meta_file]):
                features = np.load(cache_file)
                window_labels = np.load(label_file)
                with open(meta_file, 'rb') as f:
                    mc = pickle.load(f)

                expected = mc.get('total_features')
                if expected is not None and features.shape[1] != expected:
                    for stale in [cache_file, label_file, meta_file]:
                        if os.path.exists(stale):
                            os.remove(stale)
                else:
                    print(
                        f"[cache] {subject_key}  {features.shape}",
                        flush=True
                    )
                    return {
                        'features': features,
                        'labels': window_labels,
                        'subject_key': subject_key,
                        'feature_names': mc['feature_names'],
                        'n_channels': mc['n_channels'],
                        'sampling_rate': mc.get('sampling_rate'),
                        'active_channels_mask': mc.get('active_channels_mask'),
                        'success': True
                    }

        # ── Load data ───────────────────────────────────────────────────
        remove_class_zero = proc_config.get('remove_class_zero', False)

        if dataset_name.startswith('Ninapro_'):
            db_version = dataset_name.split('_')[1]
            try:
                subj_id = int(subject_key)
            except ValueError:
                subj_id = subject_key
            # v12.0: ALWAYS pass remove_class_zero=False to loader.
            # Class zero removal is handled centrally in process_dataset (single point).
            # Double-removal was redundant and could mask data issues.
            loader = load_ninapro_db(
                db_version=db_version,
                data_path=data_path,
                subjects=[subj_id],
                movement_map=None,
                remove_class_zero=False
            )
            try:
                emg, labels, meta = next(loader)
            except StopIteration:
                raise ValueError(
                    f"No data loaded for subject {subj_id}"
                )

        elif dataset_name == 'UCI_Gesture':
            # v12.0: Enforce sampling_rate from config (prevents inference drift)
            loader = load_uci_gesture(
                data_path,
                subjects=[subject_key],
                sampling_rate=proc_config.get('sampling_rate')
            )
            try:
                emg, labels, meta = next(loader)
            except StopIteration:
                raise ValueError(
                    f"No data loaded for subject {subject_key}"
                )

        elif dataset_name == 'CEMHSEY':
            loader = load_cemhsey(
                data_path,
                subjects=[subject_key],
                days=[day] if day else None
            )
            try:
                emg, labels, meta = next(loader)
            except StopIteration:
                raise ValueError(
                    f"No data loaded for subject {subject_key}"
                )

        elif dataset_name == 'UCI_Physical':
            loader = load_uci_physical_action(
                data_path, subjects=[subject_key]
            )
            try:
                emg, labels, meta = next(loader)
            except StopIteration:
                raise ValueError(
                    f"No data loaded for subject {subject_key}"
                )

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # ── Extract features ────────────────────────────────────────────
        _t_load = time.time()
        print(
            f"[perf] {subject_key}: loaded {emg.shape} "
            f"in {_t_load - _t0:.1f}s",
            file=sys.stderr, flush=True
        )
        features_flat, windows, snr, feature_names = \
            extract_features_per_channel(emg, proc_config)
        _t_feat = time.time()
        print(
            f"[perf] {subject_key}: extracted {features_flat.shape} "
            f"features in {_t_feat - _t_load:.1f}s "
            f"(total {_t_feat - _t0:.1f}s)",
            file=sys.stderr, flush=True
        )

        # ── Assign labels to windows (v34.0: vectorized midpoints) ───────
        if len(windows) > 0:
            _starts = np.array([s for s, e in windows])
            _ends = np.array([e for s, e in windows])
            _mids = np.clip((_starts + _ends) // 2, 0, len(labels) - 1)
            win_labels = labels[_mids]
        else:
            win_labels = np.array([])

        # ── Cache results ───────────────────────────────────────────────
        if not disable_cache:
            os.makedirs(cache_dir, exist_ok=True)
            try:
                np.save(cache_file, features_flat)
                np.save(label_file, win_labels)
                mc = {
                    'feature_names': feature_names,
                    'n_channels': emg.shape[1],
                    'total_features': features_flat.shape[1],
                    'sampling_rate': meta.get('sampling_rate'),
                    'active_channels_mask': meta.get('active_channels_mask')
                }
                with open(meta_file, 'wb') as f:
                    pickle.dump(mc, f)
                print(
                    f"[cached] {subject_key}  {features_flat.shape}",
                    flush=True
                )
            except OSError as e:
                print(
                    f"[cache WRITE FAILED] {subject_key}: {e}",
                    flush=True
                )

        _t_total = time.time() - _t0
        print(
            f"[done] {subject_key}  features={features_flat.shape}  "
            f"({_t_total:.1f}s: load={_t_load - _t0:.1f}s, "
            f"feat={_t_feat - _t_load:.1f}s)",
            flush=True
        )

        # Capture values before freeing raw signal data
        _n_channels = emg.shape[1]
        _sampling_rate = meta.get('sampling_rate')
        _active_mask = meta.get('active_channels_mask')

        # Free per-subject memory (raw signal can be 100MB+)
        del emg, labels
        gc.collect()

        return {
            'features': features_flat,
            'labels': win_labels,
            'subject_key': subject_key,
            'feature_names': feature_names,
            'n_channels': _n_channels,
            'sampling_rate': _sampling_rate,
            'active_channels_mask': _active_mask,
            'success': True
        }

    except MemoryError:
        print(
            f"[error] {subject_key}: MemoryError — insufficient RAM",
            flush=True
        )
        return {
            'subject_key': subject_key,
            'success': False,
            'error': 'MemoryError'
        }

    except Exception as e:
        print(
            f"[error] {subject_key}: {e}", flush=True
        )
        traceback.print_exc()
        return {
            'subject_key': subject_key,
            'success': False,
            'error': str(e)
        }


# =====================================================================
# Dataset-level processing and evaluation (v11.0 — Adaptive Engine)
# =====================================================================

def process_dataset(loader, dataset_name, dataset_key, config, checkpoint,
                    quick=False, resume=False, proc_overrides=None):
    """
    Process all subjects in a dataset and run classification.

    v11.0: Applies dataset-adaptive configs for both processing and
    classification. Passes feature_names and dataset_config to
    evaluate_model for full adaptive behavior.
    """
    # v11.0: Build adaptive processing config
    proc_cfg = _build_adaptive_proc_config(
        config['processing'].copy(), dataset_key, config
    )
    if proc_overrides:
        proc_cfg.update(proc_overrides)

    # Apply dataset-specific settings
    ds_cfg = config.get('datasets', {}).get(dataset_key, {})
    proc_cfg['sampling_rate'] = ds_cfg.get(
        'sampling_rate', proc_cfg.get('sampling_rate')
    )
    proc_cfg['remove_class_zero'] = ds_cfg.get('remove_class_zero', False)

    # v11.0: Build adaptive classification config
    clf_cfg = _build_adaptive_clf_config(
        config['classification'].copy(), dataset_key, config
    )

    # v11.0: Get full dataset adaptive config for metrics
    dataset_adaptive = config.get('dataset_adaptive_configs', {}).get(
        dataset_key, {}
    )

    output_dir = config['output_dir']
    disable_cache = config.get('disable_cache', False)
    cache_dir = os.path.join(output_dir, 'cache', dataset_name)

    if not disable_cache:
        os.makedirs(cache_dir, exist_ok=True)

    # Resume from checkpoint
    processed_keys = (
        checkpoint.get(dataset_name, []) if checkpoint else []
    )
    processed_set = set(processed_keys)

    all_features, all_labels, all_groups = [], [], []
    n_subjects = 0
    feat_names = None
    active_masks = []

    # Build task list
    tasks = []
    for emg, labels, meta in loader:
        subject_key = str(meta.get('subject_id') or meta.get('subject'))

        if resume and not disable_cache and subject_key in processed_set:
            tqdm.write(f"[skip] {subject_key}")
            continue

        data_path = config['datasets'][dataset_key]['path']
        day = meta.get('day', None)
        tasks.append((
            subject_key, dataset_name, data_path, day,
            proc_cfg, cache_dir, disable_cache
        ))

        if quick:
            break

    if not tasks:
        print("No subjects to process.", flush=True)
        return None

    # v12.0: Force sequential processing for large datasets (DB7, DB2)
    # Parallel processing of feature extraction for 22+ subjects causes OOM.
    # Feature extraction is I/O bound (loading .mat), not CPU bound.
    parallel = config.get('parallel_processing', False)
    max_workers = config.get('max_parallel_workers', 2)
    _raw_n_jobs = config.get('n_jobs', max_workers)
    if _raw_n_jobs < 1:
        _raw_n_jobs = max_workers
    n_jobs = min(_raw_n_jobs, len(tasks), max_workers)

    # Override: use sequential for datasets with many subjects
    if len(tasks) > 10:
        parallel = False
        n_jobs = 1

    if parallel and len(tasks) > 1:
        print(
            f"Parallel: {len(tasks)} subjects / {n_jobs} workers "
            f"(max={max_workers})",
            flush=True
        )

        results = []
        try:
            with Pool(processes=n_jobs) as pool:
                results = list(tqdm(
                    pool.imap(process_one_subject, tasks),
                    total=len(tasks), desc=dataset_name
                ))
        except Exception as pool_err:
            print(
                f"[warn] Parallel pool failed ({pool_err}), "
                f"retrying sequentially...",
                flush=True
            )
            results = [
                process_one_subject(t)
                for t in tqdm(tasks, desc=f"{dataset_name} (sequential retry)")
            ]
    else:
        results = [
            process_one_subject(t)
            for t in tqdm(tasks, desc=dataset_name)
        ]

    # ── Aggregate results ───────────────────────────────────────────────
    failed_subjects = []
    for res in results:
        if res['success']:
            all_features.append(res['features'])
            all_labels.append(res['labels'])
            all_groups.append(np.full(len(res['labels']), n_subjects))

            if feat_names is None:
                feat_names = res['feature_names']

            if res.get('active_channels_mask') is not None:
                active_masks.append(res['active_channels_mask'])

            if not disable_cache and checkpoint:
                processed_set.add(res['subject_key'])
                try:
                    checkpoint.update(dataset_name, list(processed_set))
                except OSError:
                    pass

            n_subjects += 1
        else:
            failed_subjects.append(res.get('subject_key', 'unknown'))

    if failed_subjects:
        print(
            f"[warn] {len(failed_subjects)} subjects failed: "
            f"{failed_subjects[:5]}{'...' if len(failed_subjects) > 5 else ''}",
            flush=True
        )

    if not all_features:
        print("No features collected.", flush=True)
        return None

    # v12.0: Memory-safe aggregation with progress
    _agg_t0 = time.time()
    _total_rows = sum(f.shape[0] for f in all_features)
    _total_mb = sum(f.nbytes for f in all_features) / (1024 ** 2)
    print(f"[aggregate] Stacking {n_subjects} subjects, "
          f"{_total_rows:,} rows, {_total_mb:.0f}MB...", flush=True)

    # Pre-allocate for efficiency (avoids repeated reallocation)
    _n_cols = all_features[0].shape[1]
    X = np.empty((_total_rows, _n_cols), dtype=np.float32)
    y = np.empty(_total_rows, dtype=all_labels[0].dtype)
    groups = np.empty(_total_rows, dtype=np.int32)
    _offset = 0
    for feat, lab, grp in zip(all_features, all_labels, all_groups):
        _n = feat.shape[0]
        X[_offset:_offset + _n] = feat
        y[_offset:_offset + _n] = lab
        groups[_offset:_offset + _n] = grp
        _offset += _n
    del all_features, all_labels, all_groups  # Free memory immediately
    gc.collect()

    # Remove class zero if configured
    remove_zero = proc_cfg.get('remove_class_zero', False)
    if remove_zero:
        mask = y != 0
        X, y, groups = X[mask], y[mask], groups[mask]
        print(f"[aggregate] Removed class 0: {mask.sum()} samples remaining", flush=True)

    print(f"[aggregate] Done in {time.time() - _agg_t0:.1f}s, "
          f"X={X.shape}, y unique={len(np.unique(y))}", flush=True)

    # ── Feature dimension verification ──────────────────────────────────
    C = results[0].get('n_channels', X.shape[1])
    do_ar = proc_cfg.get('compute_ar', False)
    ar_order = proc_cfg.get('ar_order', 6) if do_ar else 0
    do_hjorth = proc_cfg.get('compute_hjorth', True)
    do_freq = proc_cfg.get('compute_freq_features', True)
    do_wavelet = proc_cfg.get('compute_wavelet', False)
    wav_level = proc_cfg.get('wavelet_level', 4)

    # v16.0: Pass ALL feature flags including TKEO bands and covariance
    do_tkeo_bands = proc_cfg.get('compute_tkeo_bands', False)
    do_cov_features = proc_cfg.get('compute_covariance_features', False)
    actual_features = X.shape[1]
    expected_dim, desc = verify_feature_dimension(
        C=C, ar_order=ar_order, do_hjorth=do_hjorth,
        do_freq=do_freq, do_wavelet=do_wavelet, wav_level=wav_level,
        do_tkeo_bands=do_tkeo_bands, do_cov_features=do_cov_features
    )
    if actual_features != expected_dim:
        print(
            f"WARNING: Feature dimension mismatch! "
            f"Expected {expected_dim}, got {actual_features}",
            flush=True
        )
    else:
        print(f"Feature dimension verified: {desc}", flush=True)

    # ── Feature statistics ─────────────────────────────────────────────
    class_names = sorted(np.unique(y).tolist())
    feat_stats = feature_statistics(X, y, feat_names, max_features=30)

    # ── Classification (v11.0: adaptive + feature_names) ───────────────
    classification_result = None
    per_subject_acc = None
    trained_models = None
    X_tests = None

    if len(class_names) > 1:
        val_cfg = config.get('validation', {})
        strategy = val_cfg.get('strategy', 'loso')
        train_ratio = val_cfg.get('train_ratio', 0.7)
        random_state = val_cfg.get('random_state', 42)

        # Auto-switch to within_subject if only 1 subject
        if n_subjects < 2:
            print(
                f"Only {n_subjects} subject(s). "
                f"Switching to 'within_subject' validation.",
                flush=True
            )
            strategy = 'within_subject'

        t0 = time.time()

        # v11.0: Pass ALL adaptive parameters + feature_names + dataset_config
        result = evaluate_model(
            X, y, groups,
            strategy=strategy,
            train_ratio=train_ratio,
            random_state=random_state,
            classifier=clf_cfg.get('classifier', 'XGBoost'),
            svm_c=clf_cfg.get('svm_c', 1.0),
            svm_gamma=clf_cfg.get('svm_gamma', 'scale'),
            class_weight=clf_cfg.get('class_weight', None),
            pca_components=clf_cfg.get('pca_components', None),
            n_estimators=clf_cfg.get('n_estimators', 200),
            n_top_features=clf_cfg.get('n_top_features', 250),
            feature_selection=clf_cfg.get('feature_selection', 'hybrid'),
            return_models=config.get('run_shap', False),
            learning_rate=clf_cfg.get('learning_rate', 0.1),
            subsample=clf_cfg.get('subsample', 0.9),
            colsample_bytree=clf_cfg.get('colsample_bytree', 0.9),
            n_jobs=config.get('n_jobs', -1),
            early_stopping_rounds=clf_cfg.get('early_stopping_rounds', 25),
            # v11.0: NEW parameters
            use_ensemble=clf_cfg.get('use_ensemble', False),
            dataset_config=dataset_adaptive,
            feature_names=feat_names,
        )

        # Unpack 6 return values
        acc, std, cm, per_subject_acc, per_subject_f1, trained_models, X_tests = result
        print(
            f"Classification ({strategy}): "
            f"{acc:.4f} +/- {std:.4f}  ({time.time() - t0:.1f}s)",
            flush=True
        )
        classification_result = (acc, std, cm)

    # ── Build output ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        'n_subjects': n_subjects,
        'n_channels': results[0].get('n_channels') if results else None,
        'sampling_rate': (
            results[0].get('sampling_rate') if results else None
        ),
        'n_movements': len(class_names),
        'class_names': [str(c) for c in class_names],
        'feature_stats': {str(k): v for k, v in feat_stats.items()},
        'classifier': clf_cfg.get('classifier', 'unknown'),
        'classification': classification_result,
        'per_subject_accuracy': per_subject_acc,
        'per_subject_macro_f1': per_subject_f1,
        'active_channel_masks': active_masks if active_masks else None,
        'issues': []
    }

    # ── Save per-classifier JSON (Day 1 automation) ──
    clf_name = clf_cfg.get('classifier', 'unknown').lower()
    json_filename = f"{dataset_name}_{clf_name}_results.json"
    json_path = os.path.join(output_dir, json_filename)
    try:
        with open(json_path, 'w', encoding='utf-8') as _jf:
            json.dump(output_data, _jf, indent=2, default=str)
        print(f"[saved] {json_path}", flush=True)
    except Exception as _je:
        print(f"[warn] Failed to save JSON: {_je}", flush=True)

    return output_data, trained_models, X_tests, feat_names


# =====================================================================
# Ablation study
# =====================================================================

def run_ablation(config, args):
    """Run ablation study over all presets in config."""
    ablation_cfg = config.get('ablation', {})
    if not ablation_cfg.get('enabled', False):
        print("Ablation not enabled in config.")
        return

    presets = ablation_cfg.get('presets', [])
    for preset in presets:
        print(f"\n=== Ablation: {preset['name']} ===")
        main(args, proc_overrides=preset['overrides'])


# =====================================================================
# Main entry point
# =====================================================================

def main(args, proc_overrides=None):
    """Main validation pipeline. v11.0 — Adaptive Engine."""

    # v13.0: Subject filtering from CLI (--subjects 1 2 3 4 5)
    target_subjects = None
    if getattr(args, 'subjects', None) and args.subjects:
        target_subjects = [int(s) for s in args.subjects]
        print(f"[filter] Processing only subjects: {target_subjects}", flush=True)

    config_path = args.config
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.yaml')

    config = load_config(config_path)

    # Ensure dataset entries exist
    for ds in args.datasets:
        if ds not in config.get('datasets', {}):
            config.setdefault('datasets', {})[ds] = {}

    disable_cache = config.get('disable_cache', False)
    chk_path = os.path.join(config['output_dir'], 'checkpoint.pkl')

    if not disable_cache:
        os.makedirs(config['output_dir'], exist_ok=True)

    chk = None
    if not disable_cache and Checkpoint is not None:
        try:
            chk = Checkpoint(chk_path)
        except Exception:
            chk = type('', (), {
                'get': lambda *a: [],
                'update': lambda *a: None
            })()
    else:
        chk = type('', (), {
            'get': lambda *a: [],
            'update': lambda *a: None
        })()

    for ds in args.datasets:
        ds_key = ds.lower()

        if ds_key == 'uci':
            path = config['datasets']['uci']['path']
            loader = load_uci_gesture(
                path, subjects=target_subjects if target_subjects else ([1] if args.quick else None)
            )
            result = process_dataset(
                loader, 'UCI_Gesture', 'uci', config, chk,
                quick=args.quick, resume=args.resume,
                proc_overrides=proc_overrides
            )
            if result is None:
                print("Skipping UCI Gesture (no data).")
                continue
            res, models, X_tests, feat_names = result
            if generate_report:
                generate_report(
                    'UCI_Gesture', config['processing'], res,
                    config['output_dir']
                )
            if config.get('run_shap', False) and models:
                from shap_analysis import run_shap_analysis
                run_shap_analysis(
                    models, X_tests, feat_names,
                    config['output_dir'], 'UCI_Gesture'
                )

        elif ds_key.startswith('ninapro_'):
            db_version = ds_key.replace('ninapro_', '').upper()
            dataset_cfg = config['datasets'][ds_key]
            path = dataset_cfg['path']
            movement_map = None
            if db_version == 'DB3':
                movement_map = config.get('db3_to_db7_movement_map')

            # v12.0: Always pass remove_class_zero=False to loader.
            # Removal happens in process_dataset after aggregation.
            loader = load_ninapro_db(
                db_version=db_version,
                data_path=path,
                subjects=target_subjects if target_subjects else ([1] if args.quick else None),
                movement_map=movement_map,
                remove_class_zero=False
            )
            result = process_dataset(
                loader, f'Ninapro_{db_version}', ds_key, config, chk,
                quick=args.quick, resume=args.resume,
                proc_overrides=proc_overrides
            )
            if result is None:
                print(f"Skipping {ds_key} (no data).")
                continue
            res, models, X_tests, feat_names = result
            if generate_report:
                generate_report(
                    f'Ninapro_{db_version}', config['processing'], res,
                    config['output_dir']
                )
            if config.get('run_shap', False) and models:
                from shap_analysis import run_shap_analysis
                channel_mask = res.get('active_channel_masks')
                if channel_mask and len(channel_mask) > 0:
                    channel_mask = channel_mask[0]
                else:
                    channel_mask = None
                run_shap_analysis(
                    models, X_tests, feat_names,
                    config['output_dir'], f'Ninapro_{db_version}',
                    channel_mask=channel_mask
                )

        elif ds_key == 'cemhsey':
            path = config['datasets'].get('cemhsey', {}).get('path', '')
            loader = load_cemhsey(
                path, subjects=target_subjects if target_subjects else ([1] if args.quick else None)
            )
            result = process_dataset(
                loader, 'CEMHSEY', 'cemhsey', config, chk,
                quick=args.quick, resume=args.resume,
                proc_overrides=proc_overrides
            )
            if result is None:
                print("Skipping CEMHSEY (no data).")
                continue
            res, _, _, _ = result
            if generate_report:
                generate_report(
                    'CEMHSEY', config['processing'], res,
                    config['output_dir']
                )

        elif ds_key == 'uci_physical':
            path = config['datasets']['uci_physical']['path']
            loader = load_uci_physical_action(
                path, subjects=target_subjects if target_subjects else ([1] if args.quick else None)
            )
            result = process_dataset(
                loader, 'UCI_Physical', 'uci_physical', config, chk,
                quick=args.quick, resume=args.resume,
                proc_overrides=proc_overrides
            )
            if result is None:
                print("Skipping UCI Physical (no data).")
                continue
            res, _, _, _ = result
            if generate_report:
                generate_report(
                    'UCI_Physical', config['processing'], res,
                    config['output_dir']
                )

    if args.cross_db:
        print("\n=== Cross-DB Transferability ===")
        cross_cfg = config.get('cross_db_experiments', [])
        for exp in cross_cfg:
            print(f"  Experiment: {exp['name']}")
            print(f"    Source: {exp['source']} -> Target: {exp['target']}")


if __name__ == '__main__':
    args = parse_args()
    if args.ablation:
        config = load_config(args.config)
        run_ablation(config, args)
    else:
        main(args)
