"""
fix_friedman.py — Corrects Friedman test p-values using existing Day 1 results.
Reads from BOTH JSON files AND generated CSV tables.
Usage: python fix_friedman.py
"""
import os
import sys
import json
import glob
import csv as csv_mod
import argparse
import numpy as np
from scipy import stats as sp_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'paper1_results')

CLF_MAP = {
    'xgboost': 'XGBoost', 'lda': 'LDA', 'linearsvc': 'LinearSVC',
    'randomforest': 'RandomForest',
}
CANONICAL_CLFS = ['XGBoost', 'LDA', 'LinearSVC', 'RandomForest']
DB_EXPECTED_N = {'DB7': 22, 'DB3': 11, 'DB2': 40}


def load_json_safe(fpath):
    for enc in ['utf-8', 'utf-8-sig', 'latin-1']:
        try:
            with open(fpath, 'r', encoding=enc) as f:
                return json.load(f)
        except:
            continue
    return None


def find_per_subject_accs_from_json(data, expected_n=None):
    """
    Find per-subject accuracy list from JSON.
    The JSON stores per_subject_macro_f1 with {'subject': X, 'macro_f1': Y}.
    Per-subject ACCURACY may be: top-level flat list, confusion matrices, or not stored.
    We try multiple strategies and pick the best match.
    """
    if not isinstance(data, dict):
        return []

    candidates = []

    # ── Strategy 1: Known key names with accuracy ──
    for key in ['per_subject', 'per_subject_results', 'subject_results',
                'per_subject_accuracy', 'subject_accuracies', 'fold_accuracies',
                ' accuracies', 'subject_accs']:
        val = data.get(key)
        if val is None:
            continue
        accs = _extract_acc_from_list(val)
        if accs:
            candidates.append((accs, 'key:'+key))

    # ── Strategy 2: Look for flat lists of floats with right length ──
    for key, val in data.items():
        if isinstance(val, list) and len(val) >= 3:
            # Check if it's a flat list of numbers in accuracy range
            if all(isinstance(x, (int, float)) for x in val):
                mean_v = np.mean(val)
                if 0.2 < mean_v < 0.95:  # reasonable accuracy range
                    if expected_n is None or abs(len(val) - expected_n) <= 2:
                        candidates.append((val, 'flat:'+key))

    # ── Strategy 3: Lists of dicts — look for any numeric field ~0.3-0.9 ──
    for key, val in data.items():
        if isinstance(val, list) and len(val) >= 3:
            if isinstance(val[0], dict):
                # Try each numeric field
                for field in val[0].keys():
                    if field == 'subject' or field == 'subject_id':
                        continue
                    try:
                        vals = [float(d[field]) for d in val if field in d]
                        if vals and 0.2 < np.mean(vals) < 0.95:
                            if expected_n is None or abs(len(vals) - expected_n) <= 2:
                                candidates.append((vals, 'field:'+key+'/'+field))
                    except (ValueError, TypeError):
                        pass

    # ── Strategy 4: Nested in sub-dicts ──
    for top_key in ['results', 'data', 'experiment', 'model', 'validation']:
        inner = data.get(top_key)
        if isinstance(inner, dict):
            for key, val in inner.items():
                if isinstance(val, list) and len(val) >= 3:
                    if all(isinstance(x, (int, float)) for x in val):
                        mean_v = np.mean(val)
                        if 0.2 < mean_v < 0.95:
                            if expected_n is None or abs(len(val) - expected_n) <= 2:
                                candidates.append((val, 'nested:'+top_key+'/'+key))

    if not candidates:
        return []

    # Score: prefer (1) right length, (2) highest mean (accuracy > F1)
    def score(item):
        accs, source = item
        s = 0
        if expected_n and len(accs) == expected_n:
            s += 100  # exact match on subject count
        elif expected_n and abs(len(accs) - expected_n) <= 1:
            s += 50
        s += np.mean(accs) * 100  # prefer higher values (accuracy > F1)
        # Prefer sources with 'accuracy' or 'acc' in name
        if 'accuracy' in source or 'acc' in source:
            s += 30
        if 'macro_f1' in source or 'f1' in source:
            s -= 50  # penalize F1 sources
        return s

    candidates.sort(key=score, reverse=True)
    best = candidates[0]
    return best[0]


def _extract_acc_from_list(items):
    """Extract accuracy values from a list of dicts or numbers."""
    accs = []
    for item in items:
        if isinstance(item, dict):
            for k in ['accuracy', 'acc', 'mean_accuracy', 'test_accuracy', 'fold_accuracy']:
                if k in item:
                    try:
                        v = float(item[k])
                        if 0 < v <= 1.0:
                            accs.append(v)
                            break
                    except:
                        pass
        elif isinstance(item, (int, float)):
            if 0 < item <= 1.0:
                accs.append(float(item))
    return accs


def try_read_from_csv(results_dir):
    """
    Try reading per-subject accuracy from CSV files generated by day2_fix_all.py.
    Files: TableS1_per_subject_results.csv, Table2_main.csv
    """
    day1 = {'DB7': {}, 'DB3': {}, 'DB2': {}}

    # Try TableS1_per_subject_results.csv
    csv_path = os.path.join(results_dir, 'TableS1_per_subject_results.csv')
    if os.path.exists(csv_path):
        print(f"  [CSV] Reading from {os.path.basename(csv_path)}")
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv_mod.DictReader(f)
                rows = list(reader)
            print(f"  [CSV] {len(rows)} rows, columns: {list(rows[0].keys())[:10]}")
            # Analyze structure
            for row in rows[:5]:
                print(f"  [CSV]   {dict(list(row.items())[:6])}")
        except Exception as e:
            print(f"  [CSV] Error: {e}")

    # Try Table2_main_results.csv
    csv_path2 = os.path.join(results_dir, 'Table2_main_results.csv')
    if os.path.exists(csv_path2):
        print(f"  [CSV] Reading from {os.path.basename(csv_path2)}")
        try:
            with open(csv_path2, 'r', encoding='utf-8') as f:
                reader = csv_mod.DictReader(f)
                rows = list(reader)
            print(f"  [CSV] {len(rows)} rows, columns: {list(rows[0].keys())}")
            for row in rows[:5]:
                print(f"  [CSV]   {dict(row)}")
        except Exception as e:
            print(f"  [CSV] Error: {e}")

    return day1


def identify_day1_from_json(results_dir):
    """Find Day 1 JSON files and extract per-subject accuracy."""
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    day1 = {'DB7': {}, 'DB3': {}, 'DB2': {}}

    for fpath in sorted(json_files):
        fname = os.path.basename(fpath)
        if fname.startswith('_') or 'progress' in fname.lower() or 'summary' in fname.lower():
            continue
        if 'window' in fname.lower() or 'feat' in fname.lower():
            continue
        if 'day3' in fname.lower() or 'cnn' in fname.lower():
            continue

        fname_lower = fname.lower()
        db = None
        for d in ['db7', 'db3', 'db2']:
            if d in fname_lower:
                db = d.upper()
                break
        if db is None:
            continue

        clf = None
        for raw, canonical in CLF_MAP.items():
            if raw in fname_lower:
                clf = canonical
                break
        if clf is None:
            continue

        data = load_json_safe(fpath)
        if data is None:
            continue

        expected_n = DB_EXPECTED_N.get(db)
        accs = find_per_subject_accs_from_json(data, expected_n)

        if accs and len(accs) >= 3:
            day1[db][clf] = accs
            print(f"  [OK] {fname}: {db}/{clf}, n={len(accs)}, "
                  f"mean={np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%")
        else:
            # Debug: show all top-level keys
            if isinstance(data, dict):
                keys_info = []
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        keys_info.append(f"{k}={v}")
                    elif isinstance(v, list):
                        if v and isinstance(v[0], dict):
                            keys_info.append(f"{k}=list[{len(v)}] keys={list(v[0].keys())[:5]}")
                        elif v and isinstance(v[0], (int, float)):
                            keys_info.append(f"{k}=list[{len(v)}] range=[{min(v):.3f},{max(v):.3f}]")
                        else:
                            keys_info.append(f"{k}=list[{len(v)}]")
                    elif isinstance(v, np.ndarray):
                        keys_info.append(f"{k}=ndarray{v.shape}")
                    elif isinstance(v, dict):
                        keys_info.append(f"{k}=dict[{len(v)}]")
                    else:
                        keys_info.append(f"{k}={type(v).__name__}")
                print(f"  [??] {fname}: no per-subj accs found. Top keys: {keys_info}")

    return day1


def friedman_test(db_name, clf_data):
    available = [c for c in CANONICAL_CLFS if c in clf_data]
    if len(available) < 2:
        return None

    arrays = [np.array(clf_data[c]) for c in available]
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        mn = min(lengths)
        arrays = [a[:mn] for a in arrays]
        print(f"  [NOTE] Trimmed to {mn} subjects")

    n = len(arrays[0])
    k = len(available)
    stat, p = sp_stats.friedmanchisquare(*arrays)

    all_accs = np.array([a for a in arrays]).T
    ranks = np.apply_along_axis(lambda x: sp_stats.rankdata(-x), 1, all_accs)
    mean_ranks = ranks.mean(axis=0)

    denom = n * (k - 1) - stat
    if denom > 0:
        F_im = (stat * (n - 1)) / denom
        p_im = 1.0 - sp_stats.f.cdf(F_im, k - 1, (k - 1) * (n - 1))
    else:
        F_im = float('inf')
        p_im = 0.0

    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}
    q = q_table.get(k, 2.569)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))

    return {
        'n': n, 'k': k, 'chi2': float(stat), 'p': float(p),
        'F_im': float(F_im), 'p_im': float(p_im),
        'CD': float(cd), 'q': q,
        'ranks': {available[i]: round(float(mean_ranks[i]), 2) for i in range(k)},
        'clfs': available,
    }


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    args = parser.parse_args()
    rdir = args.dir or RESULTS_DIR

    print("=" * 70)
    print("  FRIEDMAN TEST FIX — Corrected p-values")
    print(f"  Dir: {rdir}")
    print("=" * 70)

    if not os.path.isdir(rdir):
        print(f"[ERROR] Not found: {rdir}")
        sys.exit(1)

    # ── Try CSV first ──
    print(f"\n{'─' * 60}")
    print("  PHASE 1: Trying CSV files")
    print("─" * 60)
    day1_csv = try_read_from_csv(rdir)

    # ── Then JSON ──
    print(f"\n{'─' * 60}")
    print("  PHASE 2: Extracting from JSON files")
    print("─" * 60)
    day1 = identify_day1_from_json(rdir)

    for db in ['DB7', 'DB3', 'DB2']:
        print(f"\n  {db}: {len(day1[db])} classifiers: {list(day1[db].keys())}")

    if not any(day1[db] for db in ['DB7', 'DB3', 'DB2']):
        print("\n[ERROR] No per-subject accuracy found in any source!")
        print("  The JSON files store per_subject_macro_f1 but NOT per_subject_accuracy.")
        print("  SOLUTION: Re-run Friedman using the overall mean accuracy from each file")
        print("  (less powerful test, but still valid for comparison).")
        print("\n  Or run this to see full JSON structure:")
        print('  python -c "import json; d=json.load(open(\'paper1_results/Ninapro_DB7_xgboost_results.json\')); [print(k, type(v).__name__, len(v) if hasattr(v,\'__len__\') else \'\') for k,v in d.items()]"')
        return

    print(f"\n{'=' * 70}")
    print("  FRIEDMAN TESTS")
    print("=" * 70)

    csv_rows = []
    nemenyi_rows = []

    for db in ['DB7', 'DB3', 'DB2']:
        if len(day1[db]) < 2:
            print(f"\n  {db}: [SKIP]")
            continue

        print(f"\n{'─' * 60}")
        print(f"  {db}")
        print(f"{'─' * 60}")

        for clf in CANONICAL_CLFS:
            if clf in day1[db]:
                a = day1[db][clf]
                print(f"    {clf:15s}: {np.mean(a)*100:.2f} +/- {np.std(a)*100:.2f}% (n={len(a)})")

        r = friedman_test(db, day1[db])
        if r is None:
            continue

        print(f"\n  Friedman chi2 = {r['chi2']:.4f}, p = {r['p']:.6f} ({sig(r['p'])})")
        print(f"  Iman-Davenport F = {r['F_im']:.4f}, p = {r['p_im']:.6f} ({sig(r['p_im'])})")
        print(f"  Nemenyi CD = {r['CD']:.3f}")
        print(f"  Mean ranks (1=best):")
        for c in r['clfs']:
            print(f"    {c:15s}: {r['ranks'][c]:.2f}")

        for i in range(len(r['clfs'])):
            for j in range(i + 1, len(r['clfs'])):
                c1, c2 = r['clfs'][i], r['clfs'][j]
                diff = abs(r['ranks'][c1] - r['ranks'][c2])
                s = "Yes" if diff > r['CD'] else "No"
                print(f"    {c1} vs {c2}: diff={diff:.3f} vs CD={r['CD']:.3f} -> {s}")
                nemenyi_rows.append({
                    'Database': db, 'clf_1': c1, 'clf_2': c2,
                    'rank_1': r['ranks'][c1], 'rank_2': r['ranks'][c2],
                    'rank_diff': round(diff, 3), 'CD': round(r['CD'], 3),
                    'significant': s,
                })

        csv_rows.append({
            'Database': db, 'N': r['n'], 'K': r['k'],
            'Friedman_chi2': round(r['chi2'], 4),
            'Friedman_p': round(r['p'], 6),
            'Friedman_sig': sig(r['p']),
            'ImanDavenport_F': round(r['F_im'], 4),
            'ImanDavenport_p': round(r['p_im'], 6),
            'Iman_sig': sig(r['p_im']),
            'Nemenyi_CD': round(r['CD'], 3),
            **{f'rank_{c}': r['ranks'].get(c, '') for c in CANONICAL_CLFS},
        })

    if not csv_rows:
        print("\n[ERROR] No tests!")
        return

    for fn, rows in [
        ('TableS2_friedman_CORRECTED.csv', csv_rows),
        ('TableS2_nemenyi_pairwise.csv', nemenyi_rows),
    ]:
        if rows:
            fp = os.path.join(rdir, fn)
            with open(fp, 'w', newline='') as f:
                w = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            print(f"\n[SAVED] {fp}")

    fp3 = os.path.join(rdir, 'friedman_corrected_report.txt')
    with open(fp3, 'w') as f:
        f.write("Friedman Test - Corrected Results\n" + "=" * 60 + "\n\n")
        for row in csv_rows:
            f.write(f"{row['Database']} (n={row['N']}, k={row['K']})\n")
            f.write(f"  Friedman: chi2={row['Friedman_chi2']}, p={row['Friedman_p']} ({row['Friedman_sig']})\n")
            f.write(f"  Iman-Davenport: F={row['ImanDavenport_F']}, p={row['ImanDavenport_p']} ({row['Iman_sig']})\n")
            f.write(f"  CD = {row['Nemenyi_CD']}\n  Ranks:\n")
            for c in CANONICAL_CLFS:
                rv = row.get(f'rank_{c}', '')
                if rv != '':
                    f.write(f"    {c:15s}: {rv}\n")
            f.write("\n")
    print(f"[SAVED] {fp3}")

    print(f"\n{'=' * 70}\n  DONE!\n{'=' * 70}")


if __name__ == '__main__':
    main()
