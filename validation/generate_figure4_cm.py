#!/usr/bin/env python3
"""
Generate Figure 4: Confusion Matrices for Paper 1
=================================================
XGBoost confusion matrices for ALL 3 databases.
Reads directly from saved JSON (no retraining).

Output: Figure4_confusion_matrices.pdf + .png
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# ── Font setup (cross-platform) ──
_DEJAVU = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'C:/Windows/Fonts/dejavu-sans-ttf/DejaVuSans.ttf',
    'C:/Windows/Fonts/DejaVuSans.ttf',
]
for _f in _DEJAVU:
    if os.path.isfile(_f):
        try: fm.fontManager.addfont(_f)
        except: pass
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ── Paths (auto-detect) ──
SCRIPT_DIR = str(Path(__file__).resolve().parent)
BASE = Path(SCRIPT_DIR) / 'paper1_results'
OUT  = BASE
if not BASE.is_dir():
    BASE = Path.cwd() / 'paper1_results'
    OUT  = BASE

CMAP = 'Blues'

# Movement names (41 classes mapped to DB7 standard)
MOV_NAMES = [
    'Rest', 'H.Close', 'H.Open', 'W.Flex', 'W.Ext',
    'Supin', 'Pron', 'Tripod', 'Lat', 'F.Pt',
    'F.Add', 'Th.Opp', 'F.Ext', 'F.Flex', 'Th.Ext',
    'Th.Flex', 'Th.Int', 'Idx.Fl', 'Idx.Ext', 'R.Fl',
    'L.Fl', 'R.Ext', 'L.Ext', 'Idx-LM', 'F.Snap',
    'Th.Press', 'F.Tap', 'N/A', 'N/A', 'N/A',
    'W.Circ', 'N/A', 'N/A', 'N/A', 'N/A',
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
    'N/A',
]


def load_cm(db_label, classifier='xgboost'):
    """Load confusion matrix + accuracy from JSON."""
    fname = BASE / f"Ninapro_{db_label}_{classifier}_results.json"
    if not fname.exists():
        print(f"  [WARN] Not found: {fname}")
        return None, 0

    with open(fname) as f:
        data = json.load(f)

    clf = data.get('classification', [])
    if isinstance(clf, list) and len(clf) >= 3:
        cm = np.array(clf[2], dtype=np.float64)
        acc = clf[0] * 100
        return cm, acc
    return None, 0


def plot_cm(ax, cm, title, n_classes):
    """Plot single confusion matrix (normalized by row)."""
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_pct = cm / row_sum  # 0 to 1

    im = ax.imshow(cm_pct, cmap=CMAP, vmin=0, vmax=1, aspect='equal')

    n = len(cm_pct)
    for i in range(n):
        for j in range(n):
            val = cm_pct[i, j]
            if i == j:
                color = 'white' if val > 0.5 else '#1F2937'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=4.5, color=color, fontweight='bold')
            elif val > 0.05:
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=3.5, color='#64748B')

    # Short labels for ticks
    short = []
    for k in range(n):
        if k < len(MOV_NAMES):
            name = MOV_NAMES[k]
            if name == 'N/A':
                short.append(str(k))
            else:
                short.append(name)
        else:
            short.append(str(k))

    tick_step = max(1, n // 10)
    ticks = list(range(0, n, tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([short[t] for t in ticks], fontsize=5, rotation=45, ha='right')
    ax.set_yticklabels([short[t] for t in ticks], fontsize=5)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Predicted Class', fontsize=8)
    ax.set_ylabel('True Class', fontsize=8)
    ax.tick_params(length=2, width=0.5)

    return im


def main():
    print("=" * 60)
    print("  Generating Figure 4: XGBoost Confusion Matrices")
    print("=" * 60)

    # ALL 3 DBs use XGBoost (paper compares XGBoost vs CNN-1D)
    configs = [
        ("DB7", "DB7 — XGBoost (22 intact subjects)\n41 classes, LOSO-CV"),
        ("DB3", "DB3 — XGBoost (11 transradial amputees)\n41 classes, LOSO-CV"),
        ("DB2", "DB2 — XGBoost (40 intact subjects)\n41 classes, LOSO-CV"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('white')
    cbar = None

    for idx, (db_label, title) in enumerate(configs):
        ax = axes[idx]
        cm, acc = load_cm(db_label, 'xgboost')

        if cm is None:
            ax.text(0.5, 0.5, f'No data for {db_label}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='#94A3B8')
            continue

        n = cm.shape[0]
        cm_acc = cm.diagonal().sum() / cm.sum() * 100
        full_title = f"{title}\nAccuracy: {acc:.2f}%"
        print(f"  {db_label}: {n}x{n} CM, JSON acc={acc:.2f}%, CM acc={cm_acc:.2f}%")

        im = plot_cm(ax, cm, full_title, n)

        if idx == 0:
            cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, aspect=30)
            cbar.set_label('Row-Normalized Recall', fontsize=8)
            cbar.ax.tick_params(labelsize=7)

    plt.tight_layout(pad=2.0)

    for ext in ['pdf', 'png']:
        out_path = OUT / f'Figure4_confusion_matrices.{ext}'
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [SAVED] {out_path}")

    plt.close(fig)
    print("\n  Done! Figure4 regenerated from JSON.")


if __name__ == '__main__':
    main()
