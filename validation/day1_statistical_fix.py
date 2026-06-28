#!/usr/bin/env python3
"""
day1_statistical_fix.py — Run statistical analysis on existing JSON results.
No need to re-run experiments! Just point to the results directory.

USAGE:
  python day1_statistical_fix.py
  python day1_statistical_fix.py "C:\path\to\paper1_results"
"""
import os
import sys
import json
import csv
import numpy as np
from scipy import stats
from itertools import combinations

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Auto-detect results directory
if os.path.exists(os.path.join(SCRIPT_DIR, "paper1_results")):
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "paper1_results")
elif os.path.exists(os.path.join(SCRIPT_DIR, "validation", "paper1_results")):
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "validation", "paper1_results")
elif len(sys.argv) > 1:
    RESULTS_DIR = sys.argv[1]
else:
    print("[ERROR] Cannot find paper1_results directory.")
    print("  Usage: python day1_statistical_fix.py <path_to_paper1_results>")
    sys.exit(1)

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


def _extract_acc_list(data):
    """Extract accuracy list from JSON — handles both float[] and dict[]."""
    items = data.get('per_subject_accuracy', [])
    if not items:
        return []
    # If first item is a dict with 'accuracy' key
    if isinstance(items[0], dict):
        return [item['accuracy'] for item in items]
    # If items are floats directly
    elif isinstance(items[0], (int, float)):
        return list(items)
    return []


def _extract_f1_list(data):
    """Extract macro_f1 list from JSON — handles both float[] and dict[]."""
    items = data.get('per_subject_macro_f1', [])
    if not items:
        return []
    if isinstance(items[0], dict):
        return [item['macro_f1'] for item in items]
    elif isinstance(items[0], (int, float)):
        return list(items)
    return []


def _extract_subject_ids(data, n):
    """Extract subject IDs from JSON."""
    items = data.get('per_subject_macro_f1', [])
    if items and isinstance(items[0], dict):
        return [item.get('subject', i) for i, item in enumerate(items)]
    # Try per_subject_accuracy
    items2 = data.get('per_subject_accuracy', [])
    if items2 and isinstance(items2[0], dict):
        return [item.get('subject', i) for i, item in enumerate(items2)]
    # Default: sequential
    return list(range(n))


def run_statistical_analysis():
    """Compute all statistical tests from saved JSON results."""
    print("=" * 70)
    print("  STATISTICAL ANALYSIS (FIXED) — PAPER 1")
    print("  LOSO Cross-Validation: 4 Classifiers × 3 Databases")
    print("=" * 70)
    print(f"  Results directory: {RESULTS_DIR}\n")

    # Load all JSON results
    all_results = {}
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
            acc_list = _extract_acc_list(data)
            f1_list = _extract_f1_list(data)
            print(f"  Loaded: {clf:15s} × {ds:15s} ({len(acc_list)} acc, {len(f1_list)} f1)")

    if len(all_results) < 4:
        print(f"\n  [ERROR] Only {len(all_results)} results found.")
        return False

    print(f"\n  Computing statistics for {len(all_results)} classifier-database pairs...\n")

    # ====================================================================
    # MAIN RESULTS TABLE
    # ====================================================================
    main_rows = []
    for ds in DATASETS:
        for clf in CLASSIFIERS:
            key = (clf, ds)
            if key not in all_results:
                continue
            data = all_results[key]
            acc_list = _extract_acc_list(data)
            f1_list = _extract_f1_list(data)
            if not acc_list:
                continue

            acc_arr = np.array(acc_list)
            f1_arr = np.array(f1_list) if f1_list else np.zeros_like(acc_arr)

            acc_mean = np.mean(acc_arr)
            acc_std = np.std(acc_arr, ddof=1) if len(acc_arr) > 1 else 0
            f1_mean = np.mean(f1_arr)
            f1_std = np.std(f1_arr, ddof=1) if len(f1_arr) > 1 else 0

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

    # ====================================================================
    # PAIRWISE WILCOXON TESTS
    # ====================================================================
    test_rows = []
    for ds in DATASETS:
        for clf_a, clf_b in combinations(CLASSIFIERS, 2):
            key_a = (clf_a, ds)
            key_b = (clf_b, ds)
            if key_a not in all_results or key_b not in all_results:
                continue

            acc_a = _extract_acc_list(all_results[key_a])
            acc_b = _extract_acc_list(all_results[key_b])
            f1_a = _extract_f1_list(all_results[key_a])
            f1_b = _extract_f1_list(all_results[key_b])

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
                'Diff_Acc': float(np.mean(acc_a) - np.mean(acc_b)),
                'Wilcoxon_W': float(W), 'Wilcoxon_p': float(p), 'Wilcoxon_r': float(r),
                'Wilcoxon_Sig': sig, 'Cohens_d': float(d), 'Cohens_d_Interp': d_mag,
                'F1_Wilcoxon_p': float(p_f1), 'F1_Wilcoxon_Sig': f1_sig,
            })

    # ====================================================================
    # FRIEDMAN TEST
    # ====================================================================
    friedman_rows = []
    for ds in DATASETS:
        groups = []
        for clf in CLASSIFIERS:
            key = (clf, ds)
            if key in all_results:
                groups.append(_extract_acc_list(all_results[key]))
        if len(groups) >= 3:
            try:
                chi2, p = stats.friedmanchisquare(*groups)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                friedman_rows.append({
                    'Database': ds, 'DB_Label': DB_LABELS[ds],
                    'Chi2': float(chi2), 'p': float(p), 'Sig': sig, 'N_Classifiers': len(groups),
                })
            except (ValueError, Exception):
                pass

    # ====================================================================
    # HOLM-SIDAK CORRECTION
    # ====================================================================
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
                'Comparison': comp, 'Raw_p': float(p), 'Adj_p': float(adj_p[orig_idx]),
                'Rejected': bool(p <= alpha_adj), 'Sig': sig,
            })

    # ====================================================================
    # PER-SUBJECT CSV
    # ====================================================================
    subject_rows = []
    for ds in DATASETS:
        for clf in CLASSIFIERS:
            key = (clf, ds)
            if key not in all_results:
                continue
            data = all_results[key]
            acc_items = data.get('per_subject_accuracy', [])
            f1_items = data.get('per_subject_macro_f1', [])
            subj_ids = _extract_subject_ids(data, len(acc_items))
            n = len(acc_items)
            for i in range(n):
                acc_val = acc_items[i] if isinstance(acc_items[i], (int, float)) else acc_items[i].get('accuracy', 0)
                f1_val = f1_items[i]['macro_f1'] if i < len(f1_items) and isinstance(f1_items[i], dict) else (f1_items[i] if i < len(f1_items) else 0)
                subject_rows.append({
                    'Classifier': clf, 'Database': DB_SHORT[ds],
                    'Subject': int(subj_ids[i]) + 1,  # 1-indexed
                    'Accuracy': float(acc_val),
                    'Macro_F1': float(f1_val),
                })

    # ====================================================================
    # SAVE ALL FILES
    # ====================================================================
    print("  Saving output files...\n")

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
    lines.append(f"  ALL FILES SAVED TO: {RESULTS_DIR}")
    lines.append("=" * 72)

    return "\n".join(lines)


if __name__ == '__main__':
    run_statistical_analysis()
