#!/usr/bin/env python3
"""
day2_fix_all.py — FIX ALL GAPS FROM DAYS 1+2 (ZERO re-runs)
============================================================
Reads ALL existing JSON result files and generates everything
that was missing:

  FIX #1: Nemenyi pairwise comparisons → CSV + LaTeX
  FIX #2: Macro F1 added to Day 2 summary tables  
  FIX #3: Day 1 figures (main results bar chart + confusion matrices)
  FIX #4: Per-class F1 analysis from confusion matrices
  FIX #5: Comprehensive Nemenyi for Day 1 (inter-database)

USAGE (run from validation/ folder):
  python day2_fix_all.py

OUTPUT (in paper1_results/):
  Day 1:
    figure_main_results.png          — Grouped bar chart (4 clf × 3 DB)
    figure_cm_DB7_xgboost.png        — Confusion matrices (best clf per DB)
    figure_cm_DB3_randomforest.png
    figure_cm_DB2_xgboost.png
    figure_per_class_f1.png          — Per-class F1 bar chart
    Table_per_class_f1.csv           — Per-class F1 values
    TableS5_nemenyi_posthoc.csv      — Nemenyi pairwise comparisons
    TableS5_nemenyi_posthoc.tex
  Day 2:
    Table_window_ablation_FULL.csv   — Accuracy + Macro F1
    Table_window_ablation_FULL.tex
    Table_feature_ablation_FULL.csv  — Accuracy + Macro F1
    Table_feature_ablation_FULL.tex
    Table_window_nemenyi.csv         — Nemenyi pairwise per classifier
    Table_window_nemenyi.tex
    TableS6_window_per_class_f1.csv
    TableS6_feature_per_class_f1.csv
  All:
    ALL_SUMMARY.txt                  — Complete text summary

Runtime: < 30 seconds (no experiments, just I/O + matplotlib)
"""

import os
import json
import csv as csv_mod
import numpy as np
from datetime import datetime

# ============================================================================
# AUTO-DETECT PATHS
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(SCRIPT_DIR, "paper1_results")):
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "paper1_results")
elif os.path.exists(os.path.join(os.path.dirname(SCRIPT_DIR), "paper1_results")):
    RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "paper1_results")
else:
    RESULTS_DIR = SCRIPT_DIR  # fallback

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"[PATHS] Results dir: {RESULTS_DIR}")
print(f"[PATHS] Files found: {len(os.listdir(RESULTS_DIR))}")

# ============================================================================
# CONFIGURATION
# ============================================================================
DATABASES = ['DB7', 'DB3', 'DB2']
CLASSIFIERS = ['XGBoost', 'LDA', 'LinearSVC', 'RandomForest']
DB_FULL_NAMES = {'DB7': 'NinaPro DB7 (mixed)', 'DB3': 'NinaPro DB3 (amputee)',
                 'DB2': 'NinaPro DB2 (intact)'}
DB_SUBJECTS = {'DB7': 22, 'DB3': 11, 'DB2': 40}
WINDOW_SIZES = [100, 150, 200, 250, 300, 400, 500]
FEAT_CONFIGS = ['Full', 'noICC', 'noTKEO', 'noHjorth', 'noFreq']
FEAT_LABELS = {
    'Full': 'Full (420D)', 'noICC': '{-ICC}', 'noTKEO': '{-TKEO}',
    'noHjorth': '{-Hjorth}', 'noFreq': '{-Freq}'
}

PLOT_COLORS = {'XGBoost': '#1f77b4', 'LDA': '#ff7f0e', 'LinearSVC': '#2ca02c',
               'RandomForest': '#d62728'}
DB_COLORS = {'DB7': '#4e79a7', 'DB3': '#e15759', 'DB2': '#59a14f'}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_json(filepath):
    """Load JSON, return dict or None."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return None


def save_csv(filepath, rows, fieldnames):
    """Save list of dicts to CSV."""
    if not rows:
        return
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [SAVED] {filepath}")


def friedman_test(acc_matrix):
    """Iman-Davenport Friedman test."""
    n_subj, n_cond = acc_matrix.shape
    if n_subj < 3 or n_cond < 3:
        return np.nan, np.nan, n_subj, n_cond, np.zeros(n_cond)
    from scipy.stats import f as f_dist, rankdata
    ranks = np.zeros_like(acc_matrix)
    for i in range(n_subj):
        ranks[i] = rankdata(acc_matrix[i])
    mean_ranks = ranks.mean(axis=0)
    SS_between = n_subj * np.sum((mean_ranks - (n_cond + 1) / 2.0) ** 2)
    chi2_r = 12.0 * n_subj / (n_cond * (n_cond + 1)) * SS_between
    denom = n_subj * (n_cond - 1) - chi2_r
    if denom <= 0:
        return chi2_r, 1.0, n_subj, n_cond, mean_ranks
    F_stat = (chi2_r * (n_subj - 1)) / denom
    df1 = n_cond - 1
    df2 = (n_subj - 1) * df1
    if df2 <= 0 or F_stat < 0:
        return chi2_r, 1.0, n_subj, n_cond, mean_ranks
    p_value = 1.0 - f_dist.cdf(F_stat, df1, df2)
    return chi2_r, p_value, n_subj, n_cond, mean_ranks


def nemenyi_posthoc(mean_ranks, n_subjects, n_conditions, alpha=0.05):
    """Nemenyi post-hoc test."""
    from scipy.stats import studentized_range
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


def per_class_f1_from_cm(cm):
    """Compute per-class F1 from confusion matrix."""
    cm = np.array(cm, dtype=float)
    n = cm.shape[0]
    f1_scores = []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    return f1_scores


def macro_f1_from_cm(cm):
    """Compute macro F1 from confusion matrix."""
    f1s = per_class_f1_from_cm(cm)
    return float(np.mean(f1s))


# ============================================================================
# LOAD ALL EXISTING DATA
# ============================================================================
print("\n" + "=" * 80)
print("  LOADING ALL EXISTING RESULT FILES")
print("=" * 80 + "\n")

# Day 1 data
day1_data = {}
for db in DATABASES:
    for clf in CLASSIFIERS:
        fname = f"Ninapro_{db}_{clf.lower()}_results.json"
        fpath = os.path.join(RESULTS_DIR, fname)
        d = load_json(fpath)
        if d and d.get('classification'):
            day1_data[(db, clf)] = d
            acc = d['classification'][0]
            cm = d['classification'][2]
            mf1 = macro_f1_from_cm(cm) if cm else 0
            print(f"  [OK] {db}/{clf}: acc={acc:.4f}, macro_f1={mf1:.4f}, "
                  f"cm={len(cm)}x{len(cm[0]) if cm else '?'}")
        else:
            print(f"  [MISSING] {db}/{clf}")

# Day 2 Window data
day2_window = {}
for wms in WINDOW_SIZES:
    for clf in CLASSIFIERS:
        fname = f"DB7_window_{wms}_{clf.lower()}_results.json"
        fpath = os.path.join(RESULTS_DIR, fname)
        d = load_json(fpath)
        if d and d.get('success') and d.get('classification'):
            day2_window[(wms, clf)] = d
            cm = d['classification'][2]
            mf1 = macro_f1_from_cm(cm) if cm else 0
            print(f"  [OK] window={wms}/{clf}: acc={d['classification'][0]:.4f}, "
                  f"macro_f1={mf1:.4f}")
        else:
            print(f"  [MISSING] window={wms}/{clf}")

# Day 2 Feature data
day2_feature = {}
for feat in FEAT_CONFIGS:
    for clf in ['XGBoost', 'LDA']:
        fname = f"DB7_feat_{feat}_{clf.lower()}_results.json"
        fpath = os.path.join(RESULTS_DIR, fname)
        d = load_json(fpath)
        if d and d.get('success') and d.get('classification'):
            day2_feature[(feat, clf)] = d
            cm = d['classification'][2]
            mf1 = macro_f1_from_cm(cm) if cm else 0
            print(f"  [OK] feat={feat}/{clf}: acc={d['classification'][0]:.4f}, "
                  f"macro_f1={mf1:.4f}")
        else:
            print(f"  [MISSING] feat={feat}/{clf}")

n_day1 = len(day1_data)
n_window = len(day2_window)
n_feature = len(day2_feature)
print(f"\n  Summary: {n_day1} Day 1, {n_window} Window, {n_feature} Feature experiments loaded")


# ============================================================================
# FIX #3: DAY 1 FIGURES
# ============================================================================
print("\n" + "=" * 80)
print("  FIX #3: GENERATING DAY 1 FIGURES")
print("=" * 80 + "\n")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Use default system fonts (works on Windows, Linux, Mac)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
# ── Figure 1: Main Results Grouped Bar Chart ──
if day1_data:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Accuracy
    ax = axes[0]
    x = np.arange(len(DATABASES))
    width = 0.2
    for i, clf in enumerate(CLASSIFIERS):
        accs = [day1_data[(db, clf)]['classification'][0] * 100 for db in DATABASES
                if (db, clf) in day1_data]
        stds = [day1_data[(db, clf)]['classification'][1] * 100 for db in DATABASES
                if (db, clf) in day1_data]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, accs, width, label=clf, color=PLOT_COLORS[clf],
                      yerr=stds, capsize=3, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Database', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Classification Accuracy', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{db}\n(n={DB_SUBJECTS[db]})' for db in DATABASES])
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 85)
    ax.grid(axis='y', alpha=0.3)

    # Macro F1
    ax = axes[1]
    for i, clf in enumerate(CLASSIFIERS):
        f1s = []
        for db in DATABASES:
            if (db, clf) in day1_data:
                cm = day1_data[(db, clf)]['classification'][2]
                f1s.append(macro_f1_from_cm(cm) * 100)
            else:
                f1s.append(0)
        # Also get per-subject std from per_subject_macro_f1 if available
        offset = (i - 1.5) * width
        ax.bar(x + offset, f1s, width, label=clf, color=PLOT_COLORS[clf],
               edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Database', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro F1 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Macro F1 Score', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{db}\n(n={DB_SUBJECTS[db]})' for db in DATABASES])
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 40)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Paper 1: LOSO Cross-Validation Results — 4 Classifiers x 3 Databases',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fpath = os.path.join(RESULTS_DIR, 'figure_main_results.png')
    plt.savefig(fpath, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [SAVED] {fpath}")

    # ── Figure 2: Confusion Matrices (best classifier per DB) ──
    best_clf_per_db = {}
    for db in DATABASES:
        best_acc = -1
        best_clf = None
        for clf in CLASSIFIERS:
            if (db, clf) in day1_data:
                acc = day1_data[(db, clf)]['classification'][0]
                if acc > best_acc:
                    best_acc = acc
                    best_clf = clf
        best_clf_per_db[db] = best_clf
        print(f"  Best clf for {db}: {best_clf} ({best_acc:.4f})")

    for db in DATABASES:
        clf = best_clf_per_db[db]
        if (db, clf) not in day1_data:
            continue
        cm = np.array(day1_data[(db, clf)]['classification'][2], dtype=float)
        n = cm.shape[0]

        # Normalize per row
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=0.5, aspect='auto')
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        ax.set_title(f'Confusion Matrix — {DB_FULL_NAMES[db]}\n'
                     f'{clf} (Acc={day1_data[(db, clf)]["classification"][0]*100:.1f}%)',
                     fontsize=13, fontweight='bold')

        # Only show ticks every 5 classes
        tick_step = max(1, n // 10)
        ticks = list(range(0, n, tick_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Proportion')

        fpath = os.path.join(RESULTS_DIR, f'figure_cm_{db}_{clf.lower()}.png')
        plt.savefig(fpath, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  [SAVED] {fpath}")


# ============================================================================
# FIX #4: PER-CLASS F1 ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("  FIX #4: PER-CLASS F1 ANALYSIS")
print("=" * 80 + "\n")

# Day 1 per-class F1
all_per_class_f1 = {}
for db in DATABASES:
    class_names = None
    for clf in CLASSIFIERS:
        if (db, clf) not in day1_data:
            continue
        d = day1_data[(db, clf)]
        if class_names is None:
            class_names = d.get('class_names', [str(i) for i in range(d['n_movements'])])
        cm = d['classification'][2]
        f1s = per_class_f1_from_cm(cm)
        if clf not in all_per_class_f1:
            all_per_class_f1[clf] = {}
        all_per_class_f1[clf][db] = f1s

# Save per-class F1 table
if all_per_class_f1:
    rows = []
    n_classes = len(class_names) if class_names else 41
    for c_idx in range(n_classes):
        cname = class_names[c_idx] if class_names and c_idx < len(class_names) else str(c_idx)
        row = {'class': cname}
        for clf in CLASSIFIERS:
            for db in DATABASES:
                if clf in all_per_class_f1 and db in all_per_class_f1[clf]:
                    if c_idx < len(all_per_class_f1[clf][db]):
                        row[f'{clf}_{db}_F1'] = f"{all_per_class_f1[clf][db][c_idx]:.4f}"
                    else:
                        row[f'{clf}_{db}_F1'] = '---'
                else:
                    row[f'{clf}_{db}_F1'] = '---'
        rows.append(row)

    fieldnames = ['class'] + [f'{clf}_{db}_F1' for clf in CLASSIFIERS for db in DATABASES]
    save_csv(os.path.join(RESULTS_DIR, 'Table_per_class_f1.csv'), rows, fieldnames)

    # Per-class F1 figure (best clf per DB, grouped)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=200)
    for ax_idx, db in enumerate(DATABASES):
        ax = axes[ax_idx]
        clf = best_clf_per_db.get(db, 'XGBoost')
        if clf not in all_per_class_f1 or db not in all_per_class_f1[clf]:
            continue
        f1s = all_per_class_f1[clf][db]
        colors = ['#e15759' if f < 0.1 else '#f28e2b' if f < 0.3 else '#59a14f' for f in f1s]
        ax.bar(range(len(f1s)), [f * 100 for f in f1s], color=colors, edgecolor='white',
               linewidth=0.3)
        ax.set_xlabel('Class', fontsize=10)
        ax.set_ylabel('F1 (%)', fontsize=10)
        ax.set_title(f'{DB_FULL_NAMES[db]}\n{clf}', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        # Mark easy/hard threshold
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.suptitle('Per-Class F1 Scores (Best Classifier per Database)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fpath = os.path.join(RESULTS_DIR, 'figure_per_class_f1.png')
    plt.savefig(fpath, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  [SAVED] {fpath}")

    # Print hardest classes
    print("\n  Hardest classes (F1 < 10%):")
    for db in DATABASES:
        clf = best_clf_per_db.get(db)
        if clf in all_per_class_f1 and db in all_per_class_f1[clf]:
            f1s = all_per_class_f1[clf][db]
            hard = [(i, f) for i, f in enumerate(f1s) if f < 0.10]
            hard.sort(key=lambda x: x[1])
            cn = class_names if class_names else [str(i) for i in range(len(f1s))]
            for idx, f in hard[:5]:
                print(f"    {db}/{clf}: class {cn[idx] if idx < len(cn) else idx} F1={f:.4f}")


# ============================================================================
# FIX #5: NEMENYI FOR DAY 1 (INTER-DATABASE)
# ============================================================================
print("\n" + "=" * 80)
print("  FIX #5: NEMENYI POST-HOC FOR DAY 1")
print("=" * 80 + "\n")

nemenyi_rows = []
for db in DATABASES:
    available_clfs = [clf for clf in CLASSIFIERS if (db, clf) in day1_data]
    if len(available_clfs) < 3:
        continue

    # Build per-subject accuracy matrix
    all_subj_accs = [day1_data[(db, clf)]['per_subject_accuracy'] for clf in available_clfs]
    min_len = min(len(a) for a in all_subj_accs)
    acc_matrix = np.column_stack([np.array(a[:min_len]) for a in all_subj_accs])

    chi2, p_val, n_subj, n_cond, mean_ranks = friedman_test(acc_matrix)

    sig = "***" if (not np.isnan(p_val) and p_val < 0.001) else \
          "**" if (not np.isnan(p_val) and p_val < 0.01) else \
          "*" if (not np.isnan(p_val) and p_val < 0.05) else "n.s."

    print(f"  {db}: Friedman chi2={chi2:.4f}, p={p_val:.4f} ({sig}) "
          f"[n={n_subj}, k={n_cond}]")
    print(f"    Mean ranks: {', '.join(f'{clf}={r:.2f}' for clf, r in zip(available_clfs, mean_ranks))}")

    # Nemenyi if significant
    if not np.isnan(p_val) and p_val < 0.05:
        cd, comps = nemenyi_posthoc(mean_ranks, n_subj, n_cond, 0.05)
        print(f"    Nemenyi CD = {cd:.4f}")
        for comp in comps:
            clf_i = available_clfs[comp['cond_i']]
            clf_j = available_clfs[comp['cond_j']]
            mark = "***" if comp['significant'] else ""
            print(f"      {clf_i} vs {clf_j}: rank_diff={comp['rank_diff']:.3f}, "
                  f"CD={comp['critical_distance']:.3f} {mark}")

            nemenyi_rows.append({
                'database': db, 'classifier_i': clf_i, 'classifier_j': clf_j,
                'mean_rank_i': f"{mean_ranks[comp['cond_i']]:.3f}",
                'mean_rank_j': f"{mean_ranks[comp['cond_j']]:.3f}",
                'rank_diff': f"{comp['rank_diff']:.4f}",
                'critical_distance': f"{comp['critical_distance']:.4f}",
                'significant': 'Yes' if comp['significant'] else 'No',
                'friedman_p': f"{p_val:.4f}",
            })

if nemenyi_rows:
    fieldnames = ['database', 'classifier_i', 'classifier_j', 'mean_rank_i',
                  'mean_rank_j', 'rank_diff', 'critical_distance',
                  'significant', 'friedman_p']
    save_csv(os.path.join(RESULTS_DIR, 'TableS5_nemenyi_posthoc.csv'), nemenyi_rows, fieldnames)

    # LaTeX
    tex_path = os.path.join(RESULTS_DIR, 'TableS5_nemenyi_posthoc.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n\centering\n")
        f.write(r"\caption{Nemenyi Post-Hoc Pairwise Comparisons}" + "\n")
        f.write(r"\label{tab:nemenyi_posthoc}" + "\n")
        f.write(r"\begin{tabular}{lllrrrl}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write("DB & Clf $i$ & Clf $j$ & Rank $i$ & Rank $j$ & $\\Delta$Rank & CD \\\\\n")
        f.write(r"\midrule" + "\n")
        for row in nemenyi_rows:
            sig = r"$^{*}$" if row['significant'] == 'Yes' else ""
            f.write(f"{row['database']} & {row['classifier_i']} & {row['classifier_j']} & "
                    f"{row['mean_rank_i']} & {row['mean_rank_j']} & "
                    f"{row['rank_diff']}{sig} & {row['critical_distance']} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"  [SAVED] {tex_path}")


# ============================================================================
# FIX #1: NEMENYI PAIRWISE FOR DAY 2 WINDOW ABLATION
# ============================================================================
print("\n" + "=" * 80)
print("  FIX #1: NEMENYI PAIRWISE FOR WINDOW ABLATION")
print("=" * 80 + "\n")

if day2_window:
    window_nemenyi_rows = []
    for clf in CLASSIFIERS:
        available_wms = sorted(set(wms for (wms, c) in day2_window if c == clf))
        if len(available_wms) < 3:
            continue

        all_subj = [day2_window[(wms, clf)].get('per_subject_accuracy', []) for wms in available_wms]
        min_len = min(len(a) for a in all_subj)
        if min_len < 3:
            continue
        acc_matrix = np.column_stack([np.array(a[:min_len]) for a in all_subj])

        chi2, p_val, n_subj, n_cond, mean_ranks = friedman_test(acc_matrix)

        sig = "***" if (not np.isnan(p_val) and p_val < 0.001) else \
              "**" if (not np.isnan(p_val) and p_val < 0.01) else \
              "*" if (not np.isnan(p_val) and p_val < 0.05) else "n.s."

        print(f"  {clf}: Friedman chi2={chi2:.4f}, p={p_val:.4f} ({sig})")
        print(f"    Mean ranks: {', '.join(f'{w}ms={r:.2f}' for w, r in zip(available_wms, mean_ranks))}")

        if not np.isnan(p_val) and p_val < 0.05:
            cd, comps = nemenyi_posthoc(mean_ranks, n_subj, n_cond, 0.05)
            print(f"    Nemenyi CD = {cd:.4f}")
            for comp in comps:
                w_i = available_wms[comp['cond_i']]
                w_j = available_wms[comp['cond_j']]
                mark = "***" if comp['significant'] else ""
                print(f"      {w_i}ms vs {w_j}ms: diff={comp['rank_diff']:.3f}, "
                      f"CD={comp['critical_distance']:.3f} {mark}")
                window_nemenyi_rows.append({
                    'classifier': clf, 'window_i': f"{w_i}ms", 'window_j': f"{w_j}ms",
                    'mean_rank_i': f"{mean_ranks[comp['cond_i']]:.3f}",
                    'mean_rank_j': f"{mean_ranks[comp['cond_j']]:.3f}",
                    'rank_diff': f"{comp['rank_diff']:.4f}",
                    'critical_distance': f"{comp['critical_distance']:.4f}",
                    'significant': 'Yes' if comp['significant'] else 'No',
                    'friedman_chi2': f"{chi2:.4f}", 'friedman_p': f"{p_val:.4f}",
                })

    if window_nemenyi_rows:
        fieldnames = ['classifier', 'window_i', 'window_j', 'mean_rank_i', 'mean_rank_j',
                      'rank_diff', 'critical_distance', 'significant',
                      'friedman_chi2', 'friedman_p']
        save_csv(os.path.join(RESULTS_DIR, 'Table_window_nemenyi.csv'),
                 window_nemenyi_rows, fieldnames)

        # LaTeX
        tex = os.path.join(RESULTS_DIR, 'Table_window_nemenyi.tex')
        with open(tex, 'w', encoding='utf-8') as f:
            f.write(r"\begin{table}[htbp]" + "\n\centering\n")
            f.write(r"\caption{Window Ablation --- Nemenyi Post-Hoc (DB7)}" + "\n")
            f.write(r"\label{tab:window_nemenyi}" + "\n")
            f.write(r"\begin{tabular}{llllrrrl}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write("Clf & $W_i$ & $W_j$ & Rank$_i$ & Rank$_j$ & $\\Delta$Rank & CD & Sig \\\\\n")
            f.write(r"\midrule" + "\n")
            prev_clf = None
            for row in window_nemenyi_rows:
                if prev_clf and prev_clf != row['classifier']:
                    f.write(r"\midrule" + "\n")
                f.write(f"{row['classifier']} & {row['window_i']} & {row['window_j']} & "
                        f"{row['mean_rank_i']} & {row['mean_rank_j']} & "
                        f"{row['rank_diff']} & {row['critical_distance']} & "
                        f"{row['significant']} \\\\\n")
                prev_clf = row['classifier']
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table}" + "\n")
        print(f"  [SAVED] {tex}")
else:
    print("  [SKIP] No Day 2 window data found")


# ============================================================================
# FIX #2: MACRO F1 IN DAY 2 SUMMARY TABLES
# ============================================================================
print("\n" + "=" * 80)
print("  FIX #2: MACRO F1 IN DAY 2 SUMMARY TABLES")
print("=" * 80 + "\n")

# ── Window Ablation with Macro F1 ──
if day2_window:
    rows = []
    for wms in WINDOW_SIZES:
        row = {'window_ms': wms}
        for clf in CLASSIFIERS:
            key = (wms, clf)
            if key in day2_window:
                d = day2_window[key]
                acc = d['classification'][0]
                std = d['classification'][1]
                cm = d['classification'][2]
                mf1 = macro_f1_from_cm(cm) if cm else 0
                row[f'{clf}_acc'] = f"{acc*100:.2f}+/-{std*100:.2f}"
                row[f'{clf}_macro_f1'] = f"{mf1*100:.2f}"
            else:
                row[f'{clf}_acc'] = '---'
                row[f'{clf}_macro_f1'] = '---'
        rows.append(row)

    fieldnames = ['window_ms'] + [f'{clf}_{m}' for clf in CLASSIFIERS for m in ['acc', 'macro_f1']]
    save_csv(os.path.join(RESULTS_DIR, 'Table_window_ablation_FULL.csv'), rows, fieldnames)

    # LaTeX
    tex = os.path.join(RESULTS_DIR, 'Table_window_ablation_FULL.tex')
    with open(tex, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n\centering\n")
        f.write(r"\caption{Window Size Ablation on DB7 (Accuracy + Macro F1)}" + "\n")
        f.write(r"\label{tab:window_ablation_full}" + "\n")
        header_acc = " & ".join([f"{clf} Acc" for clf in CLASSIFIERS])
        header_f1 = " & ".join([f"{clf} F1" for clf in CLASSIFIERS])
        f.write(f"Window & {header_acc} \\\\\n")
        f.write(f"(ms) & {header_f1} \\\\\n")
        f.write(r"\midrule" + "\n")
        for row in rows:
            accs = [row.get(f'{clf}_acc', '---') for clf in CLASSIFIERS]
            f1s = [row.get(f'{clf}_macro_f1', '---') for clf in CLASSIFIERS]
            f.write(f"{row['window_ms']} & {' & '.join(accs)} \\\\\n")
            f.write(f"     & {' & '.join(f1s)} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"  [SAVED] {tex}")

# ── Feature Ablation with Macro F1 ──
if day2_feature:
    rows = []
    for feat in FEAT_CONFIGS:
        label = FEAT_LABELS.get(feat, feat)
        row = {'config': feat, 'label': label}
        for clf in ['XGBoost', 'LDA']:
            key = (feat, clf)
            if key in day2_feature:
                d = day2_feature[key]
                acc = d['classification'][0]
                std = d['classification'][1]
                cm = d['classification'][2]
                mf1 = macro_f1_from_cm(cm) if cm else 0
                row[f'{clf}_acc'] = f"{acc*100:.2f}+/-{std*100:.2f}"
                row[f'{clf}_macro_f1'] = f"{mf1*100:.2f}"
            else:
                row[f'{clf}_acc'] = '---'
                row[f'{clf}_macro_f1'] = '---'
        rows.append(row)

    fieldnames = ['config', 'label'] + [f'{clf}_{m}' for clf in ['XGBoost', 'LDA']
                                         for m in ['acc', 'macro_f1']]
    save_csv(os.path.join(RESULTS_DIR, 'Table_feature_ablation_FULL.csv'), rows, fieldnames)

    tex = os.path.join(RESULTS_DIR, 'Table_feature_ablation_FULL.tex')
    with open(tex, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n\centering\n")
        f.write(r"\caption{Feature Group Ablation on DB7 (Accuracy + Macro F1)}" + "\n")
        f.write(r"\label{tab:feature_ablation_full}" + "\n")
        f.write("Config & XGBoost Acc & XGBoost F1 & LDA Acc & LDA F1 \\\\\n")
        f.write(r"\midrule" + "\n")
        for row in rows:
            f.write(f"{row['label']} & {row.get('XGBoost_acc','---')} & "
                    f"{row.get('XGBoost_macro_f1','---')} & "
                    f"{row.get('LDA_acc','---')} & "
                    f"{row.get('LDA_macro_f1','---')} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"  [SAVED] {tex}")


# ============================================================================
# DAY 2 PER-CLASS F1 (bonus: best window + best feature config)
# ============================================================================
print("\n" + "=" * 80)
print("  BONUS: DAY 2 PER-CLASS F1 ANALYSIS")
print("=" * 80 + "\n")

# Best window per-class F1 (500ms + XGBoost)
best_window_key = (500, 'XGBoost')
if best_window_key in day2_window:
    cm = day2_window[best_window_key]['classification'][2]
    f1s = per_class_f1_from_cm(cm)
    rows = []
    for i, f in enumerate(f1s):
        rows.append({'class': str(i), 'F1': f"{f:.4f}", 'F1_pct': f"{f*100:.2f}"})
    save_csv(os.path.join(RESULTS_DIR, 'TableS6_window_per_class_f1.csv'), rows, ['class', 'F1', 'F1_pct'])
    print(f"  [SAVED] TableS6_window_per_class_f1.csv (500ms/XGBoost, macro_F1={np.mean(f1s):.4f})")

# Best feature per-class F1 (Full + XGBoost)
best_feat_key = ('Full', 'XGBoost')
if best_feat_key in day2_feature:
    cm = day2_feature[best_feat_key]['classification'][2]
    f1s = per_class_f1_from_cm(cm)
    rows = []
    for i, f in enumerate(f1s):
        rows.append({'class': str(i), 'F1': f"{f:.4f}", 'F1_pct': f"{f*100:.2f}"})
    save_csv(os.path.join(RESULTS_DIR, 'TableS6_feature_per_class_f1.csv'), rows, ['class', 'F1', 'F1_pct'])
    print(f"  [SAVED] TableS6_feature_per_class_f1.csv (Full/XGBoost, macro_F1={np.mean(f1s):.4f})")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("  ALL FIXES COMPLETE — SUMMARY")
print("=" * 80 + "\n")

summary_lines = [
    f"Paper 1 — Days 1+2 Complete Analysis Summary",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    "=" * 60,
    "DAY 1: MAIN RESULTS (4 Classifiers x 3 Databases)",
    "=" * 60,
]
for db in DATABASES:
    summary_lines.append(f"\n  {DB_FULL_NAMES[db]} (n={DB_SUBJECTS[db]}, 41 classes):")
    for clf in CLASSIFIERS:
        if (db, clf) in day1_data:
            d = day1_data[(db, clf)]
            acc = d['classification'][0] * 100
            std = d['classification'][1] * 100
            cm = d['classification'][2]
            mf1 = macro_f1_from_cm(cm) * 100
            psa = d.get('per_subject_accuracy', [])
            psf = d.get('per_subject_macro_f1', [])
            summary_lines.append(
                f"    {clf:14s}: Acc={acc:.2f}+/-{std:.2f}%, "
                f"MacroF1={mf1:.2f}%, "
                f"PerSubjAcc=[{','.join(f'{a*100:.1f}' for a in psa[:5])}{'...' if len(psa)>5 else ''}]"
            )

summary_lines.extend([
    "",
    "=" * 60,
    "DAY 1: FRIEDMAN + NEMENYI",
    "=" * 60,
])

if day1_data:
    for db in DATABASES:
        available_clfs = [clf for clf in CLASSIFIERS if (db, clf) in day1_data]
        if len(available_clfs) < 3:
            continue
        all_subj = [day1_data[(db, clf)]['per_subject_accuracy'] for clf in available_clfs]
        min_len = min(len(a) for a in all_subj)
        acc_matrix = np.column_stack([np.array(a[:min_len]) for a in all_subj])
        chi2, p_val, n_subj, n_cond, mean_ranks = friedman_test(acc_matrix)
        sig = "***" if (not np.isnan(p_val) and p_val < 0.001) else \
              "**" if (not np.isnan(p_val) and p_val < 0.01) else \
              "*" if (not np.isnan(p_val) and p_val < 0.05) else "n.s."
        gap = max(day1_data[(db, c)]['classification'][0] for c in available_clfs) - \
              min(day1_data[(db, c)]['classification'][0] for c in available_clfs)
        summary_lines.append(f"  {db}: Friedman chi2={chi2:.3f}, p={p_val:.4f} ({sig}), "
                             f"Gap={gap*100:.2f}%")
        summary_lines.append(f"    Ranks: {', '.join(f'{c}={r:.2f}' for c, r in zip(available_clfs, mean_ranks))}")

if day2_window:
    summary_lines.extend([
        "",
        "=" * 60,
        "DAY 2: WINDOW ABLATION (DB7, 7 sizes x 4 clf)",
        "=" * 60,
    ])
    for wms in WINDOW_SIZES:
        accs = [day2_window[(wms, clf)]['classification'][0] * 100
                for clf in CLASSIFIERS if (wms, clf) in day2_window]
        if accs:
            summary_lines.append(f"  {wms}ms: {', '.join(f'{a:.2f}%' for a in accs)}")

if day2_feature:
    summary_lines.extend([
        "",
        "=" * 60,
        "DAY 2: FEATURE ABLATION (DB7, 5 configs x 2 clf)",
        "=" * 60,
    ])
    for feat in FEAT_CONFIGS:
        label = FEAT_LABELS.get(feat, feat)
        accs = [day2_feature[(feat, clf)]['classification'][0] * 100
                for clf in ['XGBoost', 'LDA'] if (feat, clf) in day2_feature]
        if accs:
            summary_lines.append(f"  {label:14s}: {', '.join(f'{a:.2f}%' for a in accs)}")

summary_lines.extend([
    "",
    "=" * 60,
    "FILES GENERATED",
    "=" * 60,
])

# List all files generated
gen_files = [f for f in os.listdir(RESULTS_DIR) if f not in ['_run_progress.json',
            'feature_stats'] and not f.startswith('Ninapro_') and not f.startswith('DB7_')]
for f in sorted(gen_files):
    fpath = os.path.join(RESULTS_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    summary_lines.append(f"  {f} ({size_kb:.1f} KB)")

summary_text = "\n".join(summary_lines)
print(summary_text)

fpath = os.path.join(RESULTS_DIR, 'ALL_SUMMARY.txt')
with open(fpath, 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"\n  [SAVED] {fpath}")

print(f"\n{'='*80}")
print("  DONE — ALL FIXES APPLIED SUCCESSFULLY")
print(f"  Results directory: {RESULTS_DIR}")
print(f"{'='*80}")
