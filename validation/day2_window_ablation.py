#!/usr/bin/env python3
"""
day2_window_ablation.py — Window Size Ablation Study on NinaPro DB7

Paper 1, Day 2: Ablation Studies
=================================
Tests 7 window sizes (100, 150, 200, 250, 300, 400, 500 ms) × 4 classifiers
(XGBoost, LDA, LinearSVC, RandomForest) on DB7 (22-fold LOSO).

Total experiments: 28 (7 window sizes × 4 classifiers)
Default window (200ms) results already exist from Day 1 — re-run for consistency.

Usage:
    python day2_window_ablation.py                           # Run all 28 experiments
    python day2_window_ablation.py --skip-existing          # Skip completed experiments
    python day2_window_ablation.py --classifiers XGBoost LDA  # Run only 2 classifiers
    python day2_window_ablation.py --windows 100 200 300     # Run only 3 window sizes

Output files (in paper1_results/):
    DB7_window_{ms}_{clf}_results.json    — Per-experiment full results
    Table_window_ablation.csv             — Summary table
    Table_window_ablation.tex             — LaTeX table
    figure_window_ablation.png            — Line plot (x=window_ms, y=accuracy)

Author: Paper 1 Day 2 Ablation Pipeline
"""

import os
import sys
import json
import time
import copy
import argparse
import traceback
import numpy as np
from datetime import datetime

# ============================================================================
# Import handling — works from both project root and validation/ folder
# ============================================================================
def _find_project_root():
    """Find project root directory (where config.yaml lives)."""
    # Check if we're running from validation/ subfolder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        script_dir,
        os.path.join(script_dir, 'validation'),
        os.path.dirname(script_dir),
        os.path.join(os.path.dirname(script_dir), 'validation'),
    ]
    for candidate in candidates:
        config_path = os.path.join(candidate, 'config.yaml')
        if os.path.exists(config_path):
            return candidate, config_path
    # Fallback: return script dir (will error if config.yaml not found)
    return script_dir, os.path.join(script_dir, 'config.yaml')


_PROJECT_ROOT, _DEFAULT_CONFIG_PATH = _find_project_root()

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import pipeline modules
try:
    from validation.validate_engine import load_config, process_dataset
except ImportError:
    from validate_engine import load_config, process_dataset

try:
    from validation.data_loaders import load_ninapro_db
except ImportError:
    from data_loaders import load_ninapro_db

# ============================================================================
# Configuration
# ============================================================================
WINDOW_SIZES_MS = [100, 150, 200, 250, 300, 400, 500]
CLASSIFIERS = ['XGBoost', 'LDA', 'LinearSVC', 'RandomForest']

# Professional plot styling
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

DATASET_KEY = 'ninapro_db7'
DATASET_NAME = 'Ninapro_DB7'
N_SUBJECTS = 22


# ============================================================================
# Experiment runner
# ============================================================================
def run_single_experiment(config, clf_name, window_ms, output_dir, progress_file):
    """
    Run a single window-size × classifier experiment on DB7.

    Modifies config in-place to set window_size_ms and classifier, then calls
    process_dataset() which runs the full 22-fold LOSO pipeline.

    Parameters
    ----------
    config : dict — Full config (will be deep-copied and modified)
    clf_name : str — Classifier name (e.g., 'XGBoost')
    window_ms : int — Window size in milliseconds
    output_dir : str — Directory to save results
    progress_file : str — Path to progress JSON file

    Returns
    -------
    dict — Experiment result, or None if skipped/failed
    """
    clf_lower = clf_name.lower()
    result_filename = f"DB7_window_{window_ms}_{clf_lower}_results.json"
    result_path = os.path.join(output_dir, result_filename)

    # Check if already completed
    if os.path.exists(result_path):
        try:
            with open(result_path, 'r') as f:
                existing = json.load(f)
            if existing.get('success', False) and existing.get('classification'):
                print(f"  [SKIP] {result_filename} already exists", flush=True)
                return existing
        except (json.JSONDecodeError, KeyError):
            pass  # File corrupted, re-run

    print(f"\n{'='*70}", flush=True)
    print(f"  EXPERIMENT: window={window_ms}ms, classifier={clf_name}", flush=True)
    print(f"  Output: {result_filename}", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()

    try:
        # Deep copy config to avoid cross-experiment contamination
        cfg = copy.deepcopy(config)

        # ── Set window size in BOTH base processing AND adaptive config ──
        # The adaptive config builder in validate_engine.py reads from
        # dataset_adaptive_configs, so we must set it there too.
        cfg['processing']['window_size_ms'] = window_ms

        if 'dataset_adaptive_configs' not in cfg:
            cfg['dataset_adaptive_configs'] = {}
        if DATASET_KEY not in cfg['dataset_adaptive_configs']:
            cfg['dataset_adaptive_configs'][DATASET_KEY] = {}
        if 'processing' not in cfg['dataset_adaptive_configs'][DATASET_KEY]:
            cfg['dataset_adaptive_configs'][DATASET_KEY]['processing'] = {}

        cfg['dataset_adaptive_configs'][DATASET_KEY]['processing']['window_size_ms'] = window_ms

        # Keep overlap at 0.5 for all window sizes
        cfg['dataset_adaptive_configs'][DATASET_KEY]['processing']['overlap'] = 0.5
        cfg['processing']['overlap'] = 0.5

        # ── Set classifier ──
        cfg['classification']['classifier'] = clf_name

        # ── Create DB7 data loader ──
        db7_path = cfg['datasets'][DATASET_KEY]['path']
        if not os.path.exists(db7_path):
            raise FileNotFoundError(
                f"DB7 data path not found: {db7_path}\n"
                f"Please update config.yaml datasets.ninapro_db7.path"
            )

        loader = load_ninapro_db(
            db_version='DB7',
            data_path=db7_path,
            subjects=None,  # All subjects
            movement_map=None,
            remove_class_zero=False,
        )

        # ── Run full pipeline (feature extraction + LOSO classification) ──
        checkpoint = type('', (), {
            'get': lambda *a: [],
            'update': lambda *a: None,
        })()

        result = process_dataset(
            loader, DATASET_NAME, DATASET_KEY, cfg, checkpoint,
            quick=False, resume=False,
        )

        if result is None:
            raise RuntimeError("process_dataset returned None (no data loaded)")

        output_data, trained_models, X_tests, feat_names = result

        elapsed = time.time() - t0

        # ── Build experiment result ──
        classification = output_data.get('classification')
        if classification is not None:
            acc, std, cm = classification
            classification_list = [float(acc), float(std), cm.tolist() if hasattr(cm, 'tolist') else cm]
        else:
            classification_list = None

        per_subject_acc = output_data.get('per_subject_accuracy', [])

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
            'classification': classification_list,
            'per_subject_accuracy': per_subject_acc,
            'elapsed_seconds': round(elapsed, 1),
            'timestamp': datetime.now().isoformat(),
            'config_hash': f"window={window_ms},clf={clf_name}",
        }

        # ── Save result ──
        os.makedirs(output_dir, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(experiment_result, f, indent=2, default=str)

        # Update progress
        _save_progress(progress_file, window_ms, clf_name, 'completed',
                      acc=float(classification_list[0]) if classification_list else None,
                      elapsed=elapsed)

        mean_acc = classification_list[0] if classification_list else 'N/A'
        print(f"\n  DONE: window={window_ms}ms, {clf_name}: "
              f"accuracy={mean_acc:.4f} ({elapsed:.0f}s)", flush=True)

        return experiment_result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED: window={window_ms}ms, {clf_name}: {e}", flush=True)
        traceback.print_exc()

        error_result = {
            'success': False,
            'ablation_type': 'window_size',
            'dataset': DATASET_KEY,
            'classifier': clf_name,
            'window_size_ms': window_ms,
            'error': str(e),
            'elapsed_seconds': round(elapsed, 1),
            'timestamp': datetime.now().isoformat(),
        }

        # Save error result
        os.makedirs(output_dir, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(error_result, f, indent=2, default=str)

        # Update progress
        _save_progress(progress_file, window_ms, clf_name, 'failed',
                      error=str(e), elapsed=elapsed)

        return error_result


# ============================================================================
# Progress tracking
# ============================================================================
def _load_progress(progress_file):
    """Load progress from JSON file."""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {'experiments': [], 'started_at': datetime.now().isoformat()}


def _save_progress(progress_file, window_ms, clf_name, status,
                   acc=None, elapsed=None, error=None):
    """Append experiment result to progress file."""
    progress = _load_progress(progress_file)
    entry = {
        'window_ms': window_ms,
        'classifier': clf_name,
        'status': status,
        'timestamp': datetime.now().isoformat(),
    }
    if acc is not None:
        entry['accuracy'] = acc
    if elapsed is not None:
        entry['elapsed_seconds'] = elapsed
    if error is not None:
        entry['error'] = error

    progress['experiments'].append(entry)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2, default=str)


# ============================================================================
# Summary generation
# ============================================================================
def generate_summary(results_dir):
    """
    Read all window ablation JSON files and generate:
    1. Summary CSV
    2. LaTeX table
    3. PNG line plot
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Collect results ──
    data = {}  # (window_ms, clf_name) -> {acc, std, per_subject_acc}

    for clf in CLASSIFIERS:
        clf_lower = clf.lower()
        for wms in WINDOW_SIZES_MS:
            fname = f"DB7_window_{wms}_{clf_lower}_results.json"
            fpath = os.path.join(results_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, 'r') as f:
                    res = json.load(f)
                if not res.get('success', False):
                    continue
                clf_info = res.get('classification')
                if clf_info is None:
                    continue
                data[(wms, clf)] = {
                    'acc': clf_info[0],
                    'std': clf_info[1],
                    'per_subject': res.get('per_subject_accuracy', []),
                }
            except (json.JSONDecodeError, IndexError, KeyError):
                continue

    if not data:
        print("WARNING: No successful experiments found for summary generation.", flush=True)
        return

    # ── 1. CSV Summary ──
    csv_path = os.path.join(results_dir, 'Table_window_ablation.csv')
    with open(csv_path, 'w') as f:
        header = ['Window_ms'] + CLASSIFIERS + ['Best_Clf', 'Best_Acc']
        f.write(','.join(header) + '\n')

        for wms in WINDOW_SIZES_MS:
            row = [str(wms)]
            best_acc = -1
            best_clf = ''
            for clf in CLASSIFIERS:
                key = (wms, clf)
                if key in data:
                    acc_str = f"{data[key]['acc']:.4f} ± {data[key]['std']:.4f}"
                    row.append(acc_str)
                    if data[key]['acc'] > best_acc:
                        best_acc = data[key]['acc']
                        best_clf = clf
                else:
                    row.append('—')
                    # Try to find best from available
            row.append(best_clf)
            row.append(f"{best_acc:.4f}" if best_acc > 0 else '—')
            f.write(','.join(row) + '\n')

    print(f"  [summary] CSV saved: {csv_path}", flush=True)

    # ── 2. LaTeX Table ──
    tex_path = os.path.join(results_dir, 'Table_window_ablation.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Window Size Ablation on DB7 (22-fold LOSO, 41 classes)}" + "\n")
        f.write(r"\label{tab:window_ablation}" + "\n")
        f.write(r"\begin{tabular}{l" + "c" * len(CLASSIFIERS) + "}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write("Window (ms) & " + " & ".join(CLASSIFIERS) + r" \\" + "\n")
        f.write(r"\midrule" + "\n")

        for wms in WINDOW_SIZES_MS:
            row_parts = [str(wms)]
            for clf in CLASSIFIERS:
                key = (wms, clf)
                if key in data:
                    row_parts.append(f"{data[key]['acc']:.2f}\\% $\\pm$ {data[key]['std']:.2f}")
                else:
                    row_parts.append("—")
            f.write(" & ".join(row_parts) + r" \\" + "\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"  [summary] LaTeX saved: {tex_path}", flush=True)

    # ── 3. PNG Line Plot ──
    fig_path = os.path.join(results_dir, 'figure_window_ablation.png')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for clf in CLASSIFIERS:
        xs, ys, yerrs = [], [], []
        for wms in WINDOW_SIZES_MS:
            key = (wms, clf)
            if key in data:
                xs.append(wms)
                ys.append(data[key]['acc'] * 100)  # Convert to percentage
                yerrs.append(data[key]['std'] * 100)

        if xs:
            ax.errorbar(
                xs, ys, yerr=yerrs,
                label=clf,
                color=PLOT_COLORS.get(clf, '#333333'),
                marker=PLOT_MARKERS.get(clf, 'o'),
                markersize=8,
                linewidth=2,
                capsize=4,
                capthick=1.5,
                elinewidth=1,
            )

    ax.set_xlabel('Window Size (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Window Size Ablation — NinaPro DB7 (22-fold LOSO, 41 classes)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(WINDOW_SIZES_MS)
    ax.set_xticklabels([str(w) for w in WINDOW_SIZES_MS], fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 2))

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  [summary] Figure saved: {fig_path}", flush=True)

    # ── Print summary to console ──
    print(f"\n{'='*70}", flush=True)
    print("  WINDOW ABLATION SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Window':>8s}", end='', flush=True)
    for clf in CLASSIFIERS:
        print(f"  {clf:>14s}", end='', flush=True)
    print(flush=True)
    print(f"  {'-'*8}", end='', flush=True)
    for _ in CLASSIFIERS:
        print(f"  {'-'*14}", end='', flush=True)
    print(flush=True)

    for wms in WINDOW_SIZES_MS:
        print(f"  {wms:>6d}ms", end='', flush=True)
        for clf in CLASSIFIERS:
            key = (wms, clf)
            if key in data:
                print(f"  {data[key]['acc']*100:>6.2f}±{data[key]['std']*100:.2f}", end='', flush=True)
            else:
                print(f"  {'---':>14s}", end='', flush=True)
        print(flush=True)
    print(f"{'='*70}\n", flush=True)


# ============================================================================
# CLI argument parsing
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Day 2 — Window Size Ablation Study on DB7"
    )
    parser.add_argument(
        '--config', type=str, default=_DEFAULT_CONFIG_PATH,
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save results (default: paper1_results/ next to script)'
    )
    parser.add_argument(
        '--windows', nargs='+', type=int, default=None,
        help='Window sizes to test (default: all 7)'
    )
    parser.add_argument(
        '--classifiers', nargs='+', type=str, default=None,
        help='Classifiers to test (default: all 4)'
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip experiments that already have result files'
    )
    parser.add_argument(
        '--summary-only', action='store_true',
        help='Only generate summary tables/plots from existing results'
    )
    return parser.parse_args()


# ============================================================================
# Main entry point
# ============================================================================
def main():
    args = parse_args()

    # ── Resolve paths ──
    config_path = args.config
    if not os.path.exists(config_path):
        # Try relative to script
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            print(f"ERROR: config.yaml not found at {config_path} or {alt_path}", flush=True)
            sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper1_results')
    os.makedirs(output_dir, exist_ok=True)

    progress_file = os.path.join(output_dir, '_window_ablation_progress.json')

    # ── Load config ──
    print(f"Loading config from: {config_path}", flush=True)
    config = load_config(config_path)

    # ── Determine experiment matrix ──
    window_sizes = args.windows if args.windows else WINDOW_SIZES_MS
    classifiers = args.classifiers if args.classifiers else CLASSIFIERS

    # Validate
    for w in window_sizes:
        if w < 50:
            print(f"WARNING: Window size {w}ms is very small (fs=2000Hz → {int(w*2000/1000)} samples)", flush=True)

    total_experiments = len(window_sizes) * len(classifiers)

    print(f"\n{'='*70}", flush=True)
    print(f"  Day 2: Window Size Ablation Study", flush=True)
    print(f"  Dataset: {DATASET_KEY} ({config['datasets'][DATASET_KEY]['path']})", flush=True)
    print(f"  Window sizes: {window_sizes}", flush=True)
    print(f"  Classifiers: {classifiers}", flush=True)
    print(f"  Total experiments: {total_experiments}", flush=True)
    print(f"  Output directory: {output_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # ── Summary-only mode ──
    if args.summary_only:
        print("Generating summary from existing results...", flush=True)
        generate_summary(output_dir)
        return

    # ── Run experiments ──
    t_total_start = time.time()
    completed = 0
    failed = 0

    for i, window_ms in enumerate(window_sizes):
        for clf_name in classifiers:
            exp_num = i * len(classifiers) + classifiers.index(clf_name) + 1
            print(f"\n[{exp_num}/{total_experiments}] "
                  f"window={window_ms}ms, clf={clf_name}", flush=True)

            result = run_single_experiment(
                config, clf_name, window_ms, output_dir, progress_file
            )

            if result and result.get('success', False):
                completed += 1
            else:
                failed += 1

    t_total = time.time() - t_total_start

    # ── Generate summary ──
    print(f"\n{'='*70}", flush=True)
    print(f"  All experiments completed!", flush=True)
    print(f"  Successful: {completed}/{total_experiments}", flush=True)
    print(f"  Failed: {failed}/{total_experiments}", flush=True)
    print(f"  Total time: {t_total:.0f}s ({t_total/60:.1f}min)", flush=True)
    print(f"{'='*70}\n", flush=True)

    generate_summary(output_dir)


if __name__ == '__main__':
    main()
