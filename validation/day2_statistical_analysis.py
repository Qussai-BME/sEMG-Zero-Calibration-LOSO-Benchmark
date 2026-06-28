#!/usr/bin/env python3
"""
day2_statistical_analysis.py — Statistical Analysis for Ablation Studies

Paper 1, Day 2: Ablation Studies
=================================
Reads JSON results from both window and feature ablation studies and generates:

1. Window Ablation:
   - Friedman test across window sizes (per classifier)
   - Nemenyi post-hoc test for pairwise window comparisons
   - Effect sizes (Cohen's d)
   - CSV + LaTeX table

2. Feature Ablation:
   - Wilcoxon signed-rank test: each ablation config vs Full (per classifier)
   - Cohen's d effect sizes
   - CSV + LaTeX table

All tests use per-subject accuracy (22 values for DB7).

Usage:
    python day2_statistical_analysis.py                        # Analyze both ablations
    python day2_statistical_analysis.py --window-only           # Window ablation only
    python day2_statistical_analysis.py --feature-only          # Feature ablation only
    python day2_statistical_analysis.py --results-dir ./paper1_results/

Output files (in paper1_results/):
    Table_window_stats.csv          — Friedman + Nemenyi results
    Table_window_stats.tex          — LaTeX version
    Table_feature_stats.csv         — Wilcoxon + Cohen's d results
    Table_feature_stats.tex         — LaTeX version

Author: Paper 1 Day 2 Ablation Pipeline
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

# ============================================================================
# Import handling
# ============================================================================
def _find_results_dir():
    """Find the paper1_results directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, 'paper1_results'),
        os.path.join(os.path.dirname(script_dir), 'paper1_results'),
        os.path.join(script_dir, 'validation', 'paper1_results'),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    # Fallback: use script dir
    return os.path.join(script_dir, 'paper1_results')


# ============================================================================
# Configuration
# ============================================================================
WINDOW_SIZES_MS = [100, 150, 200, 250, 300, 400, 500]
WINDOW_CLASSIFIERS = ['XGBoost', 'LDA', 'LinearSVC', 'RandomForest']

FEATURE_CONFIG_NAMES = ['Full', 'noICC', 'noTKEO', 'noHjorth', 'noFreq']
FEATURE_LABELS = {
    'Full': 'Full (426D)',
    'noICC': '{-ICC}',
    'noTKEO': '{-TKEO}',
    'noHjorth': '{-Hjorth}',
    'noFreq': '{-Freq}',
}
FEATURE_CLASSIFIERS = ['XGBoost', 'LDA']

ALPHA = 0.05
DATASET_KEY = 'ninapro_db7'


# ============================================================================
# Statistical test implementations
# ============================================================================

def friedman_test(accuracy_matrix):
    """
    Friedman test for repeated measures across multiple conditions.

    Parameters
    ----------
    accuracy_matrix : np.ndarray, shape (n_subjects, n_conditions)
        Per-subject accuracy for each condition (window size or feature config).

    Returns
    -------
    chi2 : float — Friedman chi-squared statistic
    p_value : float — p-value
    n_subjects : int
    n_conditions : int
    mean_ranks : np.ndarray — Mean rank per condition
    """
    n_subjects, n_conditions = accuracy_matrix.shape

    if n_subjects < 3 or n_conditions < 3:
        return np.nan, np.nan, n_subjects, n_conditions, np.zeros(n_conditions)

    # Rank within each subject (row)
    ranks = np.zeros_like(accuracy_matrix)
    for i in range(n_subjects):
        ranks[i] = _rankdata(accuracy_matrix[i])

    # Mean rank per condition (column)
    mean_ranks = ranks.mean(axis=0)

    # Friedman chi-squared (Iman-Davenport extension with F-distribution)
    # Standard Friedman statistic
    SS_total = n_conditions * (n_conditions + 1) * (n_conditions - 1) / 12.0
    SS_between = n_subjects * np.sum((mean_ranks - (n_conditions + 1) / 2.0) ** 2)

    chi2_r = 12.0 * n_subjects / (n_conditions * (n_conditions + 1)) * SS_between

    # Iman-Davenport F-statistic (more powerful for small samples)
    F_stat = (chi2_r * (n_subjects - 1)) / \
             (n_subjects * (n_conditions - 1) - chi2_r)

    from scipy.stats import f as f_dist

    df1 = n_conditions - 1
    df2 = (n_subjects - 1) * df1

    if df2 <= 0 or F_stat < 0:
        return chi2_r, 1.0, n_subjects, n_conditions, mean_ranks

    p_value = 1.0 - f_dist.cdf(F_stat, df1, df2)

    return chi2_r, p_value, n_subjects, n_conditions, mean_ranks


def _rankdata(arr):
    """Rank data, handling ties with average rank."""
    from scipy.stats import rankdata
    return rankdata(arr)


def nemenyi_posthoc(mean_ranks, n_subjects, n_conditions, alpha=0.05):
    """
    Nemenyi post-hoc test after Friedman.

    Computes critical distance and identifies significantly different pairs.

    Parameters
    ----------
    mean_ranks : np.ndarray — Mean ranks from Friedman test
    n_subjects : int — Number of subjects
    n_conditions : int — Number of conditions
    alpha : float — Significance level

    Returns
    -------
    cd : float — Critical distance
    comparisons : list of dict — Pairwise comparison results
    """
    # Critical distance (Nemenyi)
    from scipy.stats import studentized_range

    q_alpha = studentized_range.ppf(1 - alpha, n_conditions, np.inf)
    cd = q_alpha * np.sqrt(n_conditions * (n_conditions + 1) / (6.0 * n_subjects))

    comparisons = []
    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            significant = diff > cd
            comparisons.append({
                'cond_i': int(i),
                'cond_j': int(j),
                'rank_diff': round(float(diff), 4),
                'critical_distance': round(float(cd), 4),
                'significant': significant,
                'p_approx': round(float(min(1.0, 2.0 * (1.0 - norm_cdf(diff / (cd / 1.96))))), 4)
                if cd > 0 else 1.0,
            })

    return cd, comparisons


def norm_cdf(x):
    """Standard normal CDF."""
    from scipy.stats import norm
    return norm.cdf(x)


def wilcoxon_signed_rank(a, b, alpha=0.05):
    """
    Wilcoxon signed-rank test for paired samples.

    Parameters
    ----------
    a, b : array-like — Paired accuracy lists (per-subject)
    alpha : float

    Returns
    -------
    W : float — Wilcoxon statistic
    p_value : float
    r : float — Effect size (rank-biserial correlation)
    mean_diff : float — Mean difference
    cohens_d : float — Cohen's d
    """
    from scipy.stats import wilcoxon

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    diff = a - b
    mean_diff = float(np.mean(diff))

    # Wilcoxon test
    try:
        W, p_value = wilcoxon(diff, zero_method='wilcox', alternative='two-sided')
    except ValueError:
        # All differences are zero or too few non-zero
        return 0.0, 1.0, 0.0, mean_diff, 0.0

    # Rank-biserial correlation (effect size)
    n = int(np.count_nonzero(diff))
    if n > 0:
        total_rank_sum = n * (n + 1) / 2.0
        r = 1 - (2.0 * W) / total_rank_sum
    else:
        r = 0.0

    # Cohen's d
    cohens_d = _cohens_d(a, b)

    return float(W), float(p_value), float(r), mean_diff, cohens_d


def _cohens_d(a, b):
    """Compute Cohen's d for paired samples."""
    diff = np.array(a) - np.array(b)
    n = len(diff)
    if n < 2:
        return 0.0
    mean_d = np.mean(diff)
    std_d = np.std(diff, ddof=1)
    if std_d == 0:
        return 0.0
    return float(mean_d / std_d)


def bonferroni_correction(p_values, alpha=0.05):
    """
    Bonferroni-Holm step-up correction.

    Parameters
    ----------
    p_values : list of float
    alpha : float

    Returns
    -------
    list of (adjusted_p, significant) tuples
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n

    # Holm step-down
    cumulative_alpha = alpha
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_p = min(1.0, p * (n - rank))
        adjusted[orig_idx] = adjusted_p

    significant = [p < alpha for p in adjusted]
    return list(zip(adjusted, significant))


# ============================================================================
# Data loading helpers
# ============================================================================

def load_window_results(results_dir):
    """
    Load all window ablation results.

    Returns
    -------
    dict: (window_ms, clf_name) -> {'acc': float, 'std': float, 'per_subject': list}
    """
    data = {}
    for clf in WINDOW_CLASSIFIERS:
        clf_lower = clf.lower()
        for wms in WINDOW_SIZES_MS:
            fname = f"DB7_window_{wms}_{clf_lower}_results.json"
            fpath = os.path.join(results_dir, fname)
            if not os.path.exists(fpath):
                print(f"  [WARN] Missing: {fname}", flush=True)
                continue
            try:
                with open(fpath, 'r') as f:
                    res = json.load(f)
                if not res.get('success', False):
                    continue
                clf_info = res.get('classification')
                if clf_info is None:
                    continue
                per_subj = res.get('per_subject_accuracy', [])
                if len(per_subj) != 22:
                    print(f"  [WARN] {fname}: expected 22 subjects, got {len(per_subj)}", flush=True)
                data[(wms, clf)] = {
                    'acc': clf_info[0],
                    'std': clf_info[1],
                    'per_subject': per_subj,
                }
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                print(f"  [WARN] Error reading {fname}: {e}", flush=True)
    return data


def load_feature_results(results_dir):
    """
    Load all feature ablation results.

    Returns
    -------
    dict: (feat_name, clf_name) -> {'acc': float, 'std': float, 'per_subject': list, 'label': str}
    """
    data = {}
    for feat_name in FEATURE_CONFIG_NAMES:
        for clf in FEATURE_CLASSIFIERS:
            clf_lower = clf.lower()
            fname = f"DB7_feat_{feat_name}_{clf_lower}_results.json"
            fpath = os.path.join(results_dir, fname)
            if not os.path.exists(fpath):
                print(f"  [WARN] Missing: {fname}", flush=True)
                continue
            try:
                with open(fpath, 'r') as f:
                    res = json.load(f)
                if not res.get('success', False):
                    continue
                clf_info = res.get('classification')
                if clf_info is None:
                    continue
                per_subj = res.get('per_subject_accuracy', [])
                data[(feat_name, clf)] = {
                    'acc': clf_info[0],
                    'std': clf_info[1],
                    'per_subject': per_subj,
                    'label': res.get('feature_label', feat_name),
                }
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                print(f"  [WARN] Error reading {fname}: {e}", flush=True)
    return data


# ============================================================================
# Window Ablation Analysis
# ============================================================================

def analyze_window_ablation(results_dir):
    """Run Friedman + Nemenyi analysis on window ablation results."""
    print("\n" + "="*70, flush=True)
    print("  WINDOW ABLATION: STATISTICAL ANALYSIS", flush=True)
    print("="*70 + "\n", flush=True)

    data = load_window_results(results_dir)
    if not data:
        print("  ERROR: No window ablation results found. Run day2_window_ablation.py first.", flush=True)
        return

    csv_rows = []
    tex_rows = []
    summary_lines = []

    for clf in WINDOW_CLASSIFIERS:
        print(f"\n  --- Classifier: {clf} ---", flush=True)

        # Build accuracy matrix: (n_subjects, n_window_sizes)
        available_windows = sorted(set(
            wms for (wms, c) in data.keys() if c == clf
        ))

        if len(available_windows) < 3:
            print(f"    SKIP: Need >= 3 window sizes, have {len(available_windows)}", flush=True)
            continue

        acc_matrix = np.column_stack([
            np.array(data[(wms, clf)]['per_subject'])
            for wms in available_windows
        ])

        # Friedman test
        chi2, p_val, n_subj, n_cond, mean_ranks = friedman_test(acc_matrix)

        print(f"    Friedman χ² = {chi2:.4f}, p = {p_val:.4f} "
              f"(n={n_subj}, k={n_cond})", flush=True)

        if not np.isnan(p_val) and p_val < ALPHA:
            print(f"    *** SIGNIFICANT (p < {ALPHA}) ***", flush=True)

            # Nemenyi post-hoc
            cd, comparisons = nemenyi_posthoc(mean_ranks, n_subj, n_cond, ALPHA)
            print(f"    Nemenyi CD = {cd:.4f}", flush=True)

            for comp in comparisons:
                i, j = comp['cond_i'], comp['cond_j']
                w_i, w_j = available_windows[i], available_windows[j]
                sig_mark = "***" if comp['significant'] else ""
                print(f"      {w_i}ms vs {w_j}ms: "
                      f"rank_diff={comp['rank_diff']:.3f}, "
                      f"CD={comp['critical_distance']:.3f} "
                      f"{sig_mark}", flush=True)

        # Store for CSV/Tex
        csv_rows.append({
            'classifier': clf,
            'test': 'Friedman',
            'statistic': f"χ²={chi2:.4f}" if not np.isnan(chi2) else "N/A",
            'p_value': f"{p_val:.4f}" if not np.isnan(p_val) else "N/A",
            'significant': "Yes" if (not np.isnan(p_val) and p_val < ALPHA) else "No",
            'n_subjects': n_subj,
            'n_conditions': n_cond,
        })

        # Per-window summary
        for idx, wms in enumerate(available_windows):
            mean_r = mean_ranks[idx]
            mean_acc = np.mean(acc_matrix[:, idx])
            csv_rows.append({
                'classifier': clf,
                'test': f'Mean rank (w={wms}ms)',
                'statistic': f"{mean_r:.2f}",
                'p_value': '',
                'significant': '',
                'mean_accuracy': f"{mean_acc:.4f}",
            })

    # ── Save CSV ──
    csv_path = os.path.join(results_dir, 'Table_window_stats.csv')
    if csv_rows:
        fieldnames = sorted(set(k for row in csv_rows for k in row.keys()))
        with open(csv_path, 'w', newline='') as f:
            import csv as csv_mod
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  [saved] CSV: {csv_path}", flush=True)

    # ── Save LaTeX ──
    tex_path = os.path.join(results_dir, 'Table_window_stats.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Window Ablation — Friedman Test Results (DB7, 22-fold LOSO)}" + "\n")
        f.write(r"\label{tab:window_stats}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Classifier & $\chi^2$ & $p$-value & $n$ & $k$ \\" + "\n")
        f.write(r"\midrule" + "\n")

        for clf in WINDOW_CLASSIFIERS:
            available_windows = sorted(set(
                wms for (wms, c) in data.keys() if c == clf
            ))
            if len(available_windows) < 3:
                continue

            acc_matrix = np.column_stack([
                np.array(data[(wms, clf)]['per_subject'])
                for wms in available_windows
            ])
            chi2, p_val, n_subj, n_cond, _ = friedman_test(acc_matrix)

            sig = r"$^{\ast\ast}$" if (not np.isnan(p_val) and p_val < 0.01) else \
                  (r"$^{\ast}$" if (not np.isnan(p_val) and p_val < 0.05) else "")

            chi2_str = f"{chi2:.2f}" if not np.isnan(chi2) else "—"
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "—"
            f.write(f"{clf} & {chi2_str} & {p_str}{sig} & {n_subj} & {n_cond} \\\\\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"  [saved] LaTeX: {tex_path}", flush=True)


# ============================================================================
# Feature Ablation Analysis
# ============================================================================

def analyze_feature_ablation(results_dir):
    """Run Wilcoxon signed-rank tests for feature ablation results."""
    print("\n" + "="*70, flush=True)
    print("  FEATURE ABLATION: STATISTICAL ANALYSIS", flush=True)
    print("="*70 + "\n", flush=True)

    data = load_feature_results(results_dir)
    if not data:
        print("  ERROR: No feature ablation results found. Run day2_feature_ablation.py first.", flush=True)
        return

    csv_rows = []

    for clf in FEATURE_CLASSIFIERS:
        print(f"\n  --- Classifier: {clf} ---", flush=True)

        # Check if Full baseline exists
        full_key = ('Full', clf)
        if full_key not in data:
            print(f"    SKIP: Full baseline not found for {clf}", flush=True)
            continue

        full_acc = data[full_key]['per_subject']
        full_mean = data[full_key]['acc']

        print(f"    Full baseline: {full_mean:.4f} (± {data[full_key]['std']:.4f})", flush=True)

        # Collect p-values for Bonferroni correction
        raw_p_values = []
        comparison_names = []

        for feat_name in FEATURE_CONFIG_NAMES:
            if feat_name == 'Full':
                continue

            key = (feat_name, clf)
            if key not in data:
                print(f"    {feat_name}: MISSING", flush=True)
                continue

            ablation_acc = data[key]['per_subject']
            label = data[key].get('label', feat_name)

            # Wilcoxon signed-rank test
            W, p_val, r, mean_diff, cohens_d = wilcoxon_signed_rank(full_acc, ablation_acc)

            raw_p_values.append(p_val)
            comparison_names.append(feat_name)

            # Effect size interpretation
            d_abs = abs(cohens_d)
            if d_abs < 0.2:
                d_interp = "negligible"
            elif d_abs < 0.5:
                d_interp = "small"
            elif d_abs < 0.8:
                d_interp = "medium"
            else:
                d_interp = "large"

            direction = "+" if mean_diff > 0 else "−"
            sig_mark = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))

            print(f"    {label:>12s} vs Full: "
                  f"W={W:.0f}, p={p_val:.4f} {sig_mark}, "
                  f"Δ={direction}{abs(mean_diff)*100:.2f}%, "
                  f"d={cohens_d:.3f} ({d_interp})", flush=True)

            csv_rows.append({
                'classifier': clf,
                'comparison': f"{label} vs Full",
                'config': feat_name,
                'mean_full': f"{full_mean:.4f}",
                'mean_ablation': f"{data[key]['acc']:.4f}",
                'mean_diff_pct': f"{mean_diff*100:.2f}",
                'W': f"{W:.0f}",
                'p_value': f"{p_val:.4f}",
                'rank_biserial_r': f"{r:.4f}",
                'cohens_d': f"{cohens_d:.4f}",
                'effect_size': d_interp,
                'significant': sig_mark,
            })

        # Bonferroni-Holm correction
        if raw_p_values:
            corrected = bonferroni_correction(raw_p_values, ALPHA)
            print(f"\n    Bonferroni-Holm corrected p-values:", flush=True)
            for name, (adj_p, sig) in zip(comparison_names, corrected):
                sig_mark = "***" if adj_p < 0.001 else ("**" if adj_p < 0.01 else ("*" if adj_p < 0.05 else "ns"))
                print(f"      {FEATURE_LABELS.get(name, name):>12s}: "
                      f"adj_p={adj_p:.4f} {sig_mark}", flush=True)

            # Update CSV rows with corrected p-values
            for row, name, (adj_p, sig) in zip(
                [r for r in csv_rows if r['classifier'] == clf],
                comparison_names,
                corrected
            ):
                row['p_value_corrected'] = f"{adj_p:.4f}"
                row['significant_corrected'] = "Yes" if sig else "No"

    # ── Save CSV ──
    csv_path = os.path.join(results_dir, 'Table_feature_stats.csv')
    if csv_rows:
        fieldnames = sorted(set(k for row in csv_rows for k in row.keys()))
        with open(csv_path, 'w', newline='') as f:
            import csv as csv_mod
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  [saved] CSV: {csv_path}", flush=True)

    # ── Save LaTeX ──
    tex_path = os.path.join(results_dir, 'Table_feature_stats.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Feature Ablation — Wilcoxon Signed-Rank Tests (DB7, 22-fold LOSO)}" + "\n")
        f.write(r"\label{tab:feature_stats}" + "\n")
        f.write(r"\begin{tabular}{llrrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Classifier & Config & $\Delta$Acc (\%) & $p$ & $p_{adj}$ & $d$ & Sig. \\" + "\n")
        f.write(r"\midrule" + "\n")

        for clf in FEATURE_CLASSIFIERS:
            full_key = ('Full', clf)
            if full_key not in data:
                continue

            clf_rows = [r for r in csv_rows if r['classifier'] == clf]
            for row in clf_rows:
                d_val = float(row.get('cohens_d', 0))
                d_str = f"{abs(d_val):.2f}" + \
                    (r"$^{\dagger}$" if d_val < 0 else "")
                sig = row.get('significant_corrected', row.get('significant', ''))
                adj_p = row.get('p_value_corrected', row.get('p_value', '—'))

                f.write(f"{clf} & {row['comparison']} & {row['mean_diff_pct']} & "
                        f"{row['p_value']}{'' if sig == 'ns' else r'$^{\ast}$'} & "
                        f"{adj_p} & {d_str} & {sig} \\\\\n")

            # Separator between classifiers
            if clf != FEATURE_CLASSIFIERS[-1]:
                f.write(r"\midrule" + "\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"  [saved] LaTeX: {tex_path}", flush=True)


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Day 2 — Statistical Analysis for Ablation Studies"
    )
    parser.add_argument(
        '--results-dir', type=str, default=None,
        help='Directory containing ablation JSON results'
    )
    parser.add_argument(
        '--window-only', action='store_true',
        help='Only analyze window ablation results'
    )
    parser.add_argument(
        '--feature-only', action='store_true',
        help='Only analyze feature ablation results'
    )
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    # Resolve results directory
    results_dir = args.results_dir
    if results_dir is None:
        results_dir = _find_results_dir()

    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}", flush=True)
        print("Run the ablation scripts first to generate result files.", flush=True)
        sys.exit(1)

    print(f"Results directory: {results_dir}", flush=True)
    print(f"Timestamp: {datetime.now().isoformat()}", flush=True)

    # Count available result files
    window_files = [f for f in os.listdir(results_dir) if f.startswith('DB7_window_') and f.endswith('.json')]
    feature_files = [f for f in os.listdir(results_dir) if f.startswith('DB7_feat_') and f.endswith('.json')]
    print(f"Window ablation results: {len(window_files)} files", flush=True)
    print(f"Feature ablation results: {len(feature_files)} files", flush=True)

    run_window = not args.feature_only
    run_feature = not args.window_only

    if run_window:
        analyze_window_ablation(results_dir)

    if run_feature:
        analyze_feature_ablation(results_dir)

    print(f"\n{'='*70}", flush=True)
    print("  Statistical analysis complete!", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == '__main__':
    main()
