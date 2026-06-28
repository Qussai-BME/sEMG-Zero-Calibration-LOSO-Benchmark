#!/usr/bin/env python3
"""
Extract Confusion Matrices from Saved JSON Results
===================================================
Reads the already-computed LOSO-CV results from JSON files
(NO retraining needed) and generates publication-quality confusion
matrix figures.

Source: Ninapro_DB{2,3,7}_xgboost_results.json
  - classification[0] = mean accuracy
  - classification[1] = std accuracy
  - classification[2] = aggregate confusion matrix (41×41)

Output:
  - confusion_xgb_db2.png/pdf
  - confusion_xgb_db3.png/pdf
  - confusion_xgb_db7.png/pdf
  - Also: per-subject accuracy bar plots

Usage:
  python extract_confusion_matrices.py              # all DBs
  python extract_confusion_matrices.py --db db3      # DB3 only
  python extract_confusion_matrices.py --db all --pdf  # PDF output
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter

# ── Fonts (cross-platform) ────────────────────────────────────────────
import platform, os
import matplotlib.font_manager as fm
_DEJAVU_PATHS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',          # Linux
    'C:/Windows/Fonts/dejavu-sans-ttf/DejaVuSans.ttf',           # Windows (MSYS2)
    'C:/Windows/Fonts/DejaVuSans.ttf',                           # Windows (native)
    os.path.expanduser('~/.local/share/fonts/DejaVuSans.ttf'),   # Linux user
]
for _fp in _DEJAVU_PATHS:
    if os.path.isfile(_fp):
        try:
            fm.fontManager.addfont(_fp)
        except Exception:
            pass
plt.rcParams['font.family'] = 'DejaVu Sans' if any(os.path.isfile(p) for p in _DEJAVU_PATHS) else 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'paper1_results')
OUTPUT_DIR  = RESULTS_DIR

# Fallback: if paper1_results not found next to script, try current dir
if not os.path.isdir(RESULTS_DIR):
    RESULTS_DIR = os.path.join(os.getcwd(), 'paper1_results')
    OUTPUT_DIR  = RESULTS_DIR

# Movement names for NinaPro DB7 mapping (41 classes)
MOVEMENT_NAMES_DB7 = [
    'Rest',        # 0
    'Hand close',  # 1
    'Hand open',   # 2
    'Wrist flex',  # 3
    'Wrist ext',   # 4
    'Supination',  # 5
    'Pronation',   # 6
    'Tripod',      # 7
    'Lateral',     # 8
    'Finger pt',   # 9
    'Finger add',  # 10
    'Thumb opp',   # 11
    'Finger ext',  # 12
    'Finger flex', # 13
    'Thumb ext',   # 14
    'Thumb flex',  # 15
    'Thumb int',   # 16
    'Index flex',  # 17
    'Index ext',   # 18
    'Ring flex',   # 19
    'Little flex', # 20
    'Ring ext',    # 21
    'Little ext',  # 22
    'Index-LM opp',# 23
    'Finger snap', # 24
    'Thumb press', # 25
    'Finger tap',  # 26
    'N/a',         # 27
    'N/a',         # 28
    'N/a',         # 29
    'Wrist circ',  # 30 (if exists)
]
# Fill up to 41 if needed
while len(MOVEMENT_NAMES_DB7) < 41:
    MOVEMENT_NAMES_DB7.append(f'C{len(MOVEMENT_NAMES_DB7)}')


# ====================================================================
# Load JSON results
# ====================================================================
def load_json(db_name):
    """Load NinaPro XGBoost results from JSON."""
    db_upper = db_name.upper().replace('DB', 'DB')
    # JSON files use uppercase DB names: Ninapro_DB2_xgboost_results.json
    path = os.path.join(RESULTS_DIR, f'Ninapro_{db_upper}_xgboost_results.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    with open(path) as f:
        data = json.load(f)
    return data


# ====================================================================
# Plot confusion matrix
# ====================================================================
def plot_confusion_matrix(cm, class_names, db_name, accuracy,
                           output_dir, fmt='png', dpi=300):
    """
    Plot a publication-quality confusion matrix.

    Normalizes by row (true label) to show per-class recall rates.
    Uses percentage scale with subtle grid lines.
    """
    cm = np.array(cm, dtype=np.float64)

    # ── Normalize by row (true label) → recall per class ──
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums * 100  # percentage

    n_classes = cm.shape[0]
    short_labels = []
    for i in range(n_classes):
        if i == 0:
            short_labels.append('Rest')
        elif i < len(MOVEMENT_NAMES_DB7):
            name = MOVEMENT_NAMES_DB7[i]
            # Abbreviate long names
            if len(name) > 10:
                short_labels.append(name[:9] + '.')
            else:
                short_labels.append(name)
        else:
            short_labels.append(str(i))

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(14, 12))

    # Color map
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100, aspect='equal')

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Recall (%)', fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    # ── Annotate cells ──
    # Only annotate diagonal and cells with >2% for readability
    threshold = 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            if i == j:
                # Diagonal: bold white
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        fontsize=6.5, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.1', fc='navy', alpha=0.6, ec='none'))
            elif val > threshold:
                # Off-diagonal > threshold
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=5, color='black', alpha=0.8)
            # else: leave blank

    # ── Axis labels ──
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(short_labels, rotation=90, ha='center', fontsize=6.5)
    ax.set_yticklabels(short_labels, fontsize=6.5)

    ax.set_xlabel('Predicted Label', fontsize=13, labelpad=10)
    ax.set_ylabel('True Label', fontsize=13, labelpad=10)

    # Title
    db_pop = {'db2': 'Intact Subjects', 'db3': 'Transradial Amputees', 'db7': 'Intact Subjects'}
    title = (f'XGBoost Confusion Matrix — NinaPro DB{db_name[-1]} ({db_pop.get(db_name, "")})\n'
             f'LOSO-CV Accuracy: {accuracy:.2f}%  |  {n_classes} Classes')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Grid
    ax.set_xticks(np.arange(-.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_classes, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    plt.tight_layout()

    # Save
    out_path = os.path.join(output_dir, f'confusion_xgb_{db_name}.{fmt}')
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ====================================================================
# Plot per-subject accuracy
# ====================================================================
def plot_per_subject_accuracy(data, db_name, output_dir, fmt='png', dpi=300):
    """Bar plot of per-subject LOSO-CV accuracy."""
    n_sub = data['n_subjects']
    accs = np.array(data['per_subject_accuracy']) * 100
    f1s = [s['macro_f1'] * 100 for s in data['per_subject_macro_f1']]

    x = np.arange(n_sub)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, n_sub * 0.5), 6))

    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='steelblue', edgecolor='white')
    bars2 = ax.bar(x + width/2, f1s, width, label='Macro F1', color='coral', edgecolor='white')

    ax.set_xlabel('Test Subject (LOSO)', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'Per-Subject Performance — XGBoost on NinaPro DB{db_name[-1]}\n'
                 f'Mean Accuracy: {accs.mean():.2f}% ± {accs.std():.2f}%  |  '
                 f'Mean Macro F1: {np.mean(f1s):.2f}%',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i+1}' for i in range(n_sub)], fontsize=9)
    ax.legend(loc='best', fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add mean line
    ax.axhline(y=accs.mean(), color='navy', linestyle='--', alpha=0.5,
               label=f'Mean: {accs.mean():.1f}%')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'per_subject_xgb_{db_name}.{fmt}')
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ====================================================================
# Main
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description='Extract confusion matrices from JSON results')
    parser.add_argument('--db', default='all',
                        choices=['db2', 'db3', 'db7', 'all'],
                        help='Database(s) to process')
    parser.add_argument('--pdf', action='store_true', help='Output PDF instead of PNG')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Output directory')
    args = parser.parse_args()

    fmt = 'pdf' if args.pdf else 'png'
    os.makedirs(args.output_dir, exist_ok=True)

    db_list = ['db2', 'db3', 'db7'] if args.db == 'all' else [args.db]

    print("=" * 70)
    print("Extracting Confusion Matrices from Saved JSON Results")
    print("=" * 70)

    for db in db_list:
        print(f"\n{'─' * 50}")
        print(f"Processing {db.upper()}...")
        print(f"{'─' * 50}")

        # Load
        data = load_json(db)
        acc = data['classification'][0] * 100
        std = data['classification'][1] * 100
        cm = data['classification'][2]
        n_sub = data['n_subjects']
        n_mov = data['n_movements']
        class_names = data['class_names']

        print(f"  Subjects: {n_sub}")
        print(f"  Classes:  {n_mov}")
        print(f"  Accuracy: {acc:.2f}% ± {std:.2f}%")
        print(f"  CM shape:  {len(cm)}×{len(cm[0])}")
        print(f"  CM total:  {sum(sum(row) for row in cm):,} samples")

        # Per-subject stats
        per_sub = data['per_subject_accuracy']
        print(f"  Per-subject: min={min(per_sub)*100:.2f}%, max={max(per_sub)*100:.2f}%, "
              f"mean={np.mean(per_sub)*100:.2f}%")

        # Plot confusion matrix
        plot_confusion_matrix(cm, class_names, db, acc, args.output_dir, fmt=fmt, dpi=args.dpi)

        # Plot per-subject accuracy
        plot_per_subject_accuracy(data, db, args.output_dir, fmt=fmt, dpi=args.dpi)

    print(f"\n{'=' * 70}")
    print("Done! All confusion matrices saved to:", args.output_dir)
    print("=" * 70)


if __name__ == '__main__':
    main()
