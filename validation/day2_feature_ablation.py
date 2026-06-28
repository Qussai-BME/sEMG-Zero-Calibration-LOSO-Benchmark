#!/usr/bin/env python3
"""
day2_feature_ablation.py — Feature Group Ablation Study on NinaPro DB7

Paper 1, Day 2: Ablation Studies
=================================
Tests 5 feature configurations × 2 classifiers (XGBoost, LDA) on DB7 (22-fold LOSO).

Feature configurations:
  1. Full      — All features ON (baseline, from Day 1 — can be loaded)
  2. {-ICC}    — Remove Inter-Channel Correlation (66D)
  3. {-TKEO}   — Remove TKEO Band Energy features (48D)
  4. {-Hjorth} — Remove Hjorth parameters (3×12=36D)
  5. {-Freq}   — Remove Frequency features (7×12=84D)

Feature group breakdown (with all ON):
  - Base TD: 16 features × 12 channels = 192D
  - Hjorth:  3 features × 12 channels = 36D
  - Freq:    7 features × 12 channels = 84D
  - ICC:     C(12,2) = 66D
  - TKEO:    4 bands × 12 channels = 48D
  - Total:   192 + 36 + 84 + 66 + 48 = 426D (→ 420 after FS with hybrid MI+f_classif)

Total experiments: 10 (5 configs × 2 classifiers)

Usage:
    python day2_feature_ablation.py                        # Run all 10 experiments
    python day2_feature_ablation.py --skip-existing         # Skip completed
    python day2_feature_ablation.py --summary-only           # Generate summary only

Output files (in paper1_results/):
    DB7_feat_{config}_{clf}_results.json   — Per-experiment full results
    Table_feature_ablation.csv             — Summary table
    Table_feature_ablation.tex             — LaTeX table
    figure_feature_ablation.png            — Grouped bar chart

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
    return script_dir, os.path.join(script_dir, 'config.yaml')


_PROJECT_ROOT, _DEFAULT_CONFIG_PATH = _find_project_root()

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from validation.validate_engine import load_config, process_dataset
except ImportError:
    from validate_engine import load_config, process_dataset

try:
    from validation.data_loaders import load_ninapro_db
except ImportError:
    from data_loaders import load_ninapro_db


# ============================================================================
# Feature ablation configurations
# ============================================================================
# Each config defines which feature group flags to set to False.
# All other processing settings (EA, TKEO, etc.) remain at their
# adaptive defaults from config.yaml.
FEATURE_CONFIGS = [
    {
        'name': 'Full',
        'description': 'All features ON (baseline)',
        'label': 'Full (426D)',
        'overrides': {},  # Nothing disabled
    },
    {
        'name': 'noICC',
        'description': 'Remove Inter-Channel Correlation (−66D)',
        'label': '{−ICC}',
        'overrides': {
            'compute_inter_channel_corr': False,
        },
    },
    {
        'name': 'noTKEO',
        'description': 'Remove TKEO Band Energy features (−48D)',
        'label': '{−TKEO}',
        'overrides': {
            'compute_tkeo_bands': False,
        },
    },
    {
        'name': 'noHjorth',
        'description': 'Remove Hjorth parameters (−36D)',
        'label': '{−Hjorth}',
        'overrides': {
            'compute_hjorth': False,
        },
    },
    {
        'name': 'noFreq',
        'description': 'Remove Frequency features (−84D)',
        'label': '{−Freq}',
        'overrides': {
            'compute_freq_features': False,
        },
    },
]

CLASSIFIERS = ['XGBoost', 'LDA']

# Bar chart colors (one per feature config)
BAR_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
BAR_HATCH = ['/', '\\', 'x', '-', '+']

DATASET_KEY = 'ninapro_db7'
DATASET_NAME = 'Ninapro_DB7'


# ============================================================================
# Experiment runner
# ============================================================================
def run_single_experiment(config, clf_name, feat_config, output_dir, progress_file):
    """
    Run a single feature-config × classifier experiment on DB7.

    Modifies the adaptive processing config to disable specific feature groups,
    then calls process_dataset() for the full 22-fold LOSO pipeline.

    Parameters
    ----------
    config : dict — Full config (will be deep-copied and modified)
    clf_name : str — Classifier name
    feat_config : dict — Feature config from FEATURE_CONFIGS
    output_dir : str — Directory to save results
    progress_file : str — Path to progress JSON

    Returns
    -------
    dict — Experiment result, or None if skipped/failed
    """
    clf_lower = clf_name.lower()
    result_filename = f"DB7_feat_{feat_config['name']}_{clf_lower}_results.json"
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
            pass

    print(f"\n{'='*70}", flush=True)
    print(f"  EXPERIMENT: features={feat_config['name']}, classifier={clf_name}", flush=True)
    print(f"  Description: {feat_config['description']}", flush=True)
    print(f"  Output: {result_filename}", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()

    try:
        # Deep copy config
        cfg = copy.deepcopy(config)

        # ── Apply feature overrides to adaptive config ──
        # CRITICAL: Must override at dataset_adaptive_configs level so
        # the adaptive config builder in validate_engine.py picks them up.
        adaptive_proc = cfg.setdefault('dataset_adaptive_configs', {}). \
            setdefault(DATASET_KEY, {}). \
            setdefault('processing', {})

        for flag_key, flag_val in feat_config['overrides'].items():
            print(f"    Setting {flag_key} = {flag_val}", flush=True)
            adaptive_proc[flag_key] = flag_val
            # Also set in base processing as fallback
            cfg['processing'][flag_key] = flag_val

        # Ensure Euclidean Alignment stays ON (critical for DB7)
        adaptive_proc['euclidean_alignment'] = True

        # Ensure overlap stays at 0.5
        adaptive_proc['overlap'] = 0.5
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
            subjects=None,
            movement_map=None,
            remove_class_zero=False,
        )

        # ── Run full pipeline ──
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

        # ── Build result ──
        classification = output_data.get('classification')
        if classification is not None:
            acc, std, cm = classification
            classification_list = [float(acc), float(std),
                                   cm.tolist() if hasattr(cm, 'tolist') else cm]
        else:
            classification_list = None

        per_subject_acc = output_data.get('per_subject_accuracy', [])

        experiment_result = {
            'success': True,
            'ablation_type': 'feature_group',
            'dataset': DATASET_KEY,
            'classifier': clf_name,
            'feature_config': feat_config['name'],
            'feature_label': feat_config['label'],
            'feature_overrides': feat_config['overrides'],
            'n_subjects': output_data.get('n_subjects'),
            'n_channels': output_data.get('n_channels'),
            'sampling_rate': output_data.get('sampling_rate'),
            'n_movements': output_data.get('n_movements'),
            'classification': classification_list,
            'per_subject_accuracy': per_subject_acc,
            'elapsed_seconds': round(elapsed, 1),
            'timestamp': datetime.now().isoformat(),
        }

        # ── Save result ──
        os.makedirs(output_dir, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(experiment_result, f, indent=2, default=str)

        # Update progress
        _save_progress(progress_file, feat_config['name'], clf_name, 'completed',
                      acc=float(classification_list[0]) if classification_list else None,
                      elapsed=elapsed)

        mean_acc = classification_list[0] if classification_list else 'N/A'
        print(f"\n  DONE: feat={feat_config['name']}, {clf_name}: "
              f"accuracy={mean_acc:.4f} ({elapsed:.0f}s)", flush=True)

        return experiment_result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED: feat={feat_config['name']}, {clf_name}: {e}", flush=True)
        traceback.print_exc()

        error_result = {
            'success': False,
            'ablation_type': 'feature_group',
            'dataset': DATASET_KEY,
            'classifier': clf_name,
            'feature_config': feat_config['name'],
            'feature_label': feat_config['label'],
            'error': str(e),
            'elapsed_seconds': round(elapsed, 1),
            'timestamp': datetime.now().isoformat(),
        }

        os.makedirs(output_dir, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(error_result, f, indent=2, default=str)

        _save_progress(progress_file, feat_config['name'], clf_name, 'failed',
                      error=str(e), elapsed=elapsed)

        return error_result


# ============================================================================
# Progress tracking
# ============================================================================
def _load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {'experiments': [], 'started_at': datetime.now().isoformat()}


def _save_progress(progress_file, feat_name, clf_name, status,
                   acc=None, elapsed=None, error=None):
    progress = _load_progress(progress_file)
    entry = {
        'feature_config': feat_name,
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
    Read all feature ablation JSON files and generate:
    1. Summary CSV
    2. LaTeX table
    3. Grouped bar chart PNG
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Collect results ──
    data = {}  # (feat_name, clf_name) -> {acc, std, per_subject, label}

    for feat_cfg in FEATURE_CONFIGS:
        for clf in CLASSIFIERS:
            clf_lower = clf.lower()
            fname = f"DB7_feat_{feat_cfg['name']}_{clf_lower}_results.json"
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
                data[(feat_cfg['name'], clf)] = {
                    'acc': clf_info[0],
                    'std': clf_info[1],
                    'per_subject': res.get('per_subject_accuracy', []),
                    'label': feat_cfg['label'],
                }
            except (json.JSONDecodeError, IndexError, KeyError):
                continue

    if not data:
        print("WARNING: No successful experiments found for summary generation.", flush=True)
        return

    # ── 1. CSV Summary ──
    csv_path = os.path.join(results_dir, 'Table_feature_ablation.csv')
    with open(csv_path, 'w') as f:
        header = ['Feature_Config', 'Label'] + CLASSIFIERS + ['Best_Clf', 'Best_Acc']
        f.write(','.join(header) + '\n')

        for feat_cfg in FEATURE_CONFIGS:
            fname = feat_cfg['name']
            row = [fname, feat_cfg['label']]
            best_acc = -1
            best_clf = ''
            for clf in CLASSIFIERS:
                key = (fname, clf)
                if key in data:
                    acc_val = data[key]['acc']
                    acc_str = f"{acc_val:.4f} ± {data[key]['std']:.4f}"
                    row.append(acc_str)
                    if acc_val > best_acc:
                        best_acc = acc_val
                        best_clf = clf
                else:
                    row.append('—')
            row.append(best_clf)
            row.append(f"{best_acc:.4f}" if best_acc > 0 else '—')
            f.write(','.join(row) + '\n')

    print(f"  [summary] CSV saved: {csv_path}", flush=True)

    # ── 2. LaTeX Table ──
    tex_path = os.path.join(results_dir, 'Table_feature_ablation.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Feature Group Ablation on DB7 (22-fold LOSO, 41 classes)}" + "\n")
        f.write(r"\label{tab:feature_ablation}" + "\n")
        f.write(r"\begin{tabular}{l" + "c" * len(CLASSIFIERS) + "}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write("Configuration & " + " & ".join(CLASSIFIERS) + r" \\" + "\n")
        f.write(r"\midrule" + "\n")

        for feat_cfg in FEATURE_CONFIGS:
            fname = feat_cfg['name']
            row_parts = [feat_cfg['label']]
            for clf in CLASSIFIERS:
                key = (fname, clf)
                if key in data:
                    row_parts.append(
                        f"{data[key]['acc']:.2f}\\% $\\pm$ {data[key]['std']:.2f}"
                    )
                else:
                    row_parts.append("—")
            f.write(" & ".join(row_parts) + r" \\" + "\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"  [summary] LaTeX saved: {tex_path}", flush=True)

    # ── 3. Grouped Bar Chart ──
    fig_path = os.path.join(results_dir, 'figure_feature_ablation.png')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    n_configs = len(FEATURE_CONFIGS)
    n_clfs = len(CLASSIFIERS)
    bar_width = 0.35
    x = np.arange(n_configs)

    for clf_idx, clf in enumerate(CLASSIFIERS):
        accs = []
        errs = []
        for feat_cfg in FEATURE_CONFIGS:
            key = (feat_cfg['name'], clf)
            if key in data:
                accs.append(data[key]['acc'] * 100)
                errs.append(data[key]['std'] * 100)
            else:
                accs.append(0)
                errs.append(0)

        offset = (clf_idx - (n_clfs - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, accs, bar_width * 0.9,
            yerr=errs,
            label=clf,
            color=BAR_COLORS[clf_idx % len(BAR_COLORS)],
            edgecolor='white',
            linewidth=0.5,
            capsize=3,
            error_kw={'elinewidth': 1, 'capthick': 1},
            alpha=0.85,
        )

    ax.set_xlabel('Feature Configuration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Feature Group Ablation — NinaPro DB7 (22-fold LOSO, 41 classes)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([cfg['label'] for cfg in FEATURE_CONFIGS], fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Set reasonable y-axis range
    all_accs = []
    for feat_cfg in FEATURE_CONFIGS:
        for clf in CLASSIFIERS:
            key = (feat_cfg['name'], clf)
            if key in data:
                all_accs.append(data[key]['acc'] * 100)
    if all_accs:
        ymin = max(0, min(all_accs) - max(all_accs) * 0.1)
        ymax = min(100, max(all_accs) + max(all_accs) * 0.05)
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  [summary] Figure saved: {fig_path}", flush=True)

    # ── Print summary to console ──
    print(f"\n{'='*70}", flush=True)
    print("  FEATURE ABLATION SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Config':>10s}  {'Label':>10s}", end='', flush=True)
    for clf in CLASSIFIERS:
        print(f"  {clf:>16s}", end='', flush=True)
    print(flush=True)

    for feat_cfg in FEATURE_CONFIGS:
        fname = feat_cfg['name']
        print(f"  {fname:>10s}  {feat_cfg['label']:>10s}", end='', flush=True)
        for clf in CLASSIFIERS:
            key = (fname, clf)
            if key in data:
                print(f"  {data[key]['acc']*100:>6.2f}±{data[key]['std']*100:.2f}",
                      end='', flush=True)
            else:
                print(f"  {'---':>16s}", end='', flush=True)
        print(flush=True)
    print(f"{'='*70}\n", flush=True)


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Day 2 — Feature Group Ablation Study on DB7"
    )
    parser.add_argument(
        '--config', type=str, default=_DEFAULT_CONFIG_PATH,
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save results (default: paper1_results/)'
    )
    parser.add_argument(
        '--classifiers', nargs='+', type=str, default=None,
        help='Classifiers to test (default: XGBoost LDA)'
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
# Main
# ============================================================================
def main():
    args = parse_args()

    # ── Resolve paths ──
    config_path = args.config
    if not os.path.exists(config_path):
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

    progress_file = os.path.join(output_dir, '_feature_ablation_progress.json')

    # ── Load config ──
    print(f"Loading config from: {config_path}", flush=True)
    config = load_config(config_path)

    # ── Determine experiment matrix ──
    classifiers = args.classifiers if args.classifiers else CLASSIFIERS
    total_experiments = len(FEATURE_CONFIGS) * len(classifiers)

    print(f"\n{'='*70}", flush=True)
    print(f"  Day 2: Feature Group Ablation Study", flush=True)
    print(f"  Dataset: {DATASET_KEY} ({config['datasets'][DATASET_KEY]['path']})", flush=True)
    print(f"  Feature configs: {[c['name'] for c in FEATURE_CONFIGS]}", flush=True)
    print(f"  Classifiers: {classifiers}", flush=True)
    print(f"  Total experiments: {total_experiments}", flush=True)
    print(f"  Output directory: {output_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Print feature group info
    print("  Feature groups:", flush=True)
    print("    Base TD:  16 × 12ch = 192D", flush=True)
    print("    Hjorth:   3 × 12ch =  36D", flush=True)
    print("    Freq:     7 × 12ch =  84D", flush=True)
    print("    ICC:      C(12,2)  =  66D", flush=True)
    print("    TKEO:     4 × 12ch =  48D", flush=True)
    print("    Total:                  426D", flush=True)
    print()

    # ── Summary-only mode ──
    if args.summary_only:
        print("Generating summary from existing results...", flush=True)
        generate_summary(output_dir)
        return

    # ── Run experiments ──
    t_total_start = time.time()
    completed = 0
    failed = 0

    for i, feat_cfg in enumerate(FEATURE_CONFIGS):
        for clf_name in classifiers:
            exp_num = i * len(classifiers) + classifiers.index(clf_name) + 1
            print(f"\n[{exp_num}/{total_experiments}] "
                  f"feat={feat_cfg['name']}, clf={clf_name}", flush=True)

            result = run_single_experiment(
                config, clf_name, feat_cfg, output_dir, progress_file
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
