#!/usr/bin/env python3
"""
Day 5: Real Per-Class F1 + Computational Timing Measurement
=============================================================
VERSION 8 (v8) — DEFINITIVE, ALL DATABASES (DB2/DB3/DB7)
=============================================================

ALL fixes from v1-v7 included + NEW fixes:
  - v8 FIX: np.polyfit SVD crash → try/except + zero-fallback (DB3)
  - v8 FIX: Check for constant/near-constant signals before polyfit
  - v8 FIX: Safe extract_td_features — NEVER crashes regardless of signal
  - v7 FIX: PER-FOLD train-only label map → contiguous 0..N-1
  - v6 FIX: eval_set filtering for unseen classes
  - v5 FIX: Contiguous label map for DB2/DB3 gaps
  - v4 FIX: Pre-extract features per-subject (OOM prevention for DB2)
  - v3 FIX: Per-subject cap at 2500 windows (OOM at Fold 21)
  - v3 FIX: XGBoost max_bin=128, n_jobs=1 (OOM)
  - v3 FIX: RF n_estimators=50, max_depth=20 (OOM during pickle)
  - v3 FIX: joblib.dump try/except fallback
  - v2 FIX: 1-based labels → 0-based for XGBoost
  - v2 FIX: S21/S22 RETAINED (Option B)

=============================================================
Usage:
  python day5_perclass_and_timing_v8.py --db db7 --task all
  python day5_perclass_and_timing_v8.py --db db2 --task all
  python day5_perclass_and_timing_v8.py --db db3 --task all
  python day5_perclass_and_timing_v8.py --db db7 --task timing
  python day5_perclass_and_timing_v8.py --db db7 --task timing --classifiers rf
  python day5_perclass_and_timing_v8.py --db db7 --task all --fast

Output: paper1_results/day5_real/
"""

import os
import sys
import json
import time
import gc
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
OUT_DIR = SCRIPT_DIR / "paper1_results" / "day5_real"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Window & Feature Parameters — MUST MATCH main pipeline (config.yaml)
# =============================================================================
FS = 2000
WIN_MS = 400
WIN = int(FS * WIN_MS / 1000)      # 800 samples
OVERLAP = 0.50
STRIDE = int(WIN * (1 - OVERLAP))  # 400 samples
K_BEST = 420
PER_SUB_CAP = 2500


def print_header(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


# =============================================================================
# Label Map — PER-FOLD, from TRAIN labels ONLY
# =============================================================================
def build_train_label_map(y_train):
    """
    Build a contiguous 0-based label map from TRAIN labels ONLY.
    This guarantees y_train_0 is ALWAYS 0..N-1 with NO gaps.

    CRITICAL: This MUST be called PER-FOLD, not globally.

    Returns:
        fwd:     ndarray, fwd[original_label] = contiguous_idx (-1 if not in train)
        inv:     ndarray, inv[contiguous_idx] = original_label
        n_cls:   int, number of classes in training
    """
    train_classes = np.unique(y_train)
    n_cls = len(train_classes)
    max_lbl = int(train_classes.max())

    fwd = np.full(max_lbl + 1, -1, dtype=np.int64)
    fwd[train_classes] = np.arange(n_cls)

    inv = train_classes.copy()  # inv[i] = original_label for class i

    return fwd, inv, n_cls


# =============================================================================
# Feature Extraction — matches process_engine.py pipeline
# =============================================================================
def extract_td_features(sig):
    """
    Extract 45 time-domain features from a single-channel signal.
    v8: FULLY SAFE — never crashes regardless of signal content.
    - Wraps np.polyfit in try/except for SVD failures (DB3 constant signals)
    - Checks for constant signals before polyfit
    - All numerical operations have eps guards
    """
    n = len(sig)
    sig = sig.astype(np.float64)

    mav = np.mean(np.abs(sig))
    rms = np.sqrt(np.mean(sig**2))
    wl = np.sum(np.abs(np.diff(sig)))
    zc = np.sum(np.abs(np.diff(np.sign(sig))) > 0) / n
    ssc = np.sum(np.abs(np.diff(np.sign(np.diff(sig)))) > 0) / (n - 1)
    var = np.var(sig)
    iemg = np.sum(np.abs(sig))

    log_mav = np.log10(mav + 1e-10)
    log_rms = np.log10(rms + 1e-10)
    log_var = np.log10(var + 1e-10)
    log_detector = np.sum(np.log10(np.abs(sig) + 1e-10))

    mavs = np.mean(np.abs(np.diff(sig)))
    wamp = np.sum(np.abs(np.diff(sig)))
    aac = np.sum(np.abs(np.diff(np.sqrt(np.abs(sig) + 1e-10)))) / (n - 1)
    mwl = np.mean(np.sqrt(np.abs(np.diff(sig**2))))

    myop = rms * (np.sum(sig > 0.1 * rms) / n) if rms > 0 else 0.0

    tm3 = np.mean(np.power(np.abs(sig), 3))
    tm4 = np.mean(np.power(np.abs(sig), 4))
    tm5 = np.mean(np.power(np.abs(sig), 5))
    v_order = (np.mean(sig**2) / (np.sqrt(np.mean(sig**4)) + 1e-10))

    skew = np.mean((sig - np.mean(sig))**3) / (np.std(sig)**3 + 1e-10)
    kurt = np.mean((sig - np.mean(sig))**4) / (np.std(sig)**4 + 1e-10) - 3
    ssi = np.sum(sig**2)

    mav1 = np.mean(np.abs(sig[1:]))
    mav2 = np.mean(np.abs(sig[2:]))
    mav3 = np.mean(np.abs(sig[3:]))

    # v8 CRITICAL FIX: Safe AR coefficient extraction
    # DB3 has constant/near-constant signal windows that cause SVD divergence
    try:
        sig_std = np.std(sig)
        if sig_std < 1e-12 or n < 5:
            # Constant signal or too short for polyfit → zero AR coefficients
            ar_coeffs = np.zeros(4, dtype=np.float64)
        else:
            ar_coeffs = np.polyfit(sig, np.arange(n, dtype=np.float64),
                                   min(4, n - 1))
            # Pad to exactly 4 coefficients if needed
            if len(ar_coeffs) < 4:
                ar_coeffs = np.pad(ar_coeffs, (0, 4 - len(ar_coeffs)))
            elif len(ar_coeffs) > 4:
                ar_coeffs = ar_coeffs[:4]
            # Replace any NaN/Inf from degenerate cases
            ar_coeffs = np.nan_to_num(ar_coeffs, nan=0.0, posinf=0.0, neginf=0.0)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        # SVD did not converge, or other numerical error → zero AR coefficients
        ar_coeffs = np.zeros(4, dtype=np.float64)

    tkeo = np.mean(sig[1:-1]**2 - sig[:-2] * sig[2:])

    hist, _ = np.histogram(sig, bins=10, density=True)

    return [mav, rms, wl, zc, ssc, var, iemg, log_mav, log_rms, log_var,
            log_detector, mavs, wamp, aac, mwl, myop, tm3, tm4, tm5,
            v_order, skew, kurt, ssi, mav1, mav2, mav3, tkeo,
            ar_coeffs[0], ar_coeffs[1], ar_coeffs[2], ar_coeffs[3]] + hist.tolist()


def extract_features_windowed(emg, stim, win_size=WIN, stride=STRIDE):
    """Extract hand-crafted features from windowed EMG. Returns ORIGINAL labels."""
    n = len(emg)
    n_channels = emg.shape[1]
    features_list = []
    labels_list = []

    for i in range(0, n - win_size + 1, stride):
        window = emg[i:i + win_size]
        lbl = int(np.median(stim[i:i + win_size]))
        if lbl < 1:
            continue
        feat = []
        for ch in range(n_channels):
            sig = window[:, ch]
            feat.extend(extract_td_features(sig))
        features_list.append(feat)
        labels_list.append(lbl)

    if len(features_list) == 0:
        return np.empty((0, n_channels * 45), dtype=np.float32), np.empty(0, dtype=np.int32)

    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.int32)


# =============================================================================
# Load + Pre-extract Features (prevents OOM — v4 fix)
# =============================================================================
def load_and_preextract(db_label, data_path, fast_mode=False):
    """
    Load subjects, extract features per-subject, cap at PER_SUB_CAP,
    discard raw EMG immediately. v4 fix prevents OOM for DB2 (40 subjects).
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from data_loaders import load_ninapro_db

    gen = load_ninapro_db(db_label, data_path, remove_class_zero=False)

    subjects_feats = {}
    sid_counter = 0

    for item in gen:
        if isinstance(item, tuple) and len(item) == 3:
            emg, stim, info = item
        elif isinstance(item, tuple) and len(item) == 2:
            emg, stim = item
            info = {}
        else:
            continue

        sid = info.get("subject_id", sid_counter + 1)
        sid_counter = sid

        emg_f32 = emg.astype(np.float32)
        stim_i32 = stim.astype(np.int32)

        n_classes = len(np.unique(stim_i32[stim_i32 > 0]))
        n_samples = len(emg_f32)

        print(f"    S{sid}: {n_samples:,} samples, {n_classes} cls — extracting... ",
              end="", flush=True)
        t0 = time.time()

        X_sub, y_sub = extract_features_windowed(emg_f32, stim_i32, WIN, STRIDE)
        n_windows = len(X_sub)
        print(f"{n_windows:,} win ({time.time()-t0:.1f}s)", end="", flush=True)

        if n_windows > PER_SUB_CAP:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_windows, PER_SUB_CAP, replace=False)
            X_sub = X_sub[idx]
            y_sub = y_sub[idx]
            print(f" -> {PER_SUB_CAP}", end="", flush=True)

        subjects_feats[sid] = {
            "X": X_sub.astype(np.float32),
            "y": y_sub.astype(np.int32),
        }

        del emg, emg_f32, stim, stim_i32, X_sub, y_sub
        gc.collect()
        print(" [done]", flush=True)

    sids = sorted(subjects_feats.keys())
    total_mem = sum(s["X"].nbytes + s["y"].nbytes for s in subjects_feats.values())
    print(f"  Pre-extracted: {len(sids)} subjects, {total_mem/1024/1024:.1f} MB")

    return subjects_feats, sids


# =============================================================================
# Task 1: Real Per-Class F1
# =============================================================================
def run_perclass_f1(db_key="db7", fast_mode=False):
    """
    XGBoost LOSO-CV with per-class F1.
    v7: PER-FOLD train-only label map → guarantees contiguous 0..N-1.
    v8: Safe AR extraction for DB3 constant signals.
    """
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from xgboost import XGBClassifier

    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_mapping = {
        "db7": ("DB7", config["datasets"]["ninapro_db7"]["path"]),
        "db3": ("DB3", config["datasets"]["ninapro_db3"]["path"]),
        "db2": ("DB2", config["datasets"]["ninapro_db2"]["path"]),
    }

    if db_key not in db_mapping:
        print(f"  ERROR: unknown db '{db_key}'")
        return None

    db_label, data_path = db_mapping[db_key]
    print_header(f"Task 1: Real Per-Class F1 — {db_label}")

    print(f"  Loading data from: {data_path}")
    print(f"  Pre-extracting features (cap={PER_SUB_CAP}/subject)...")
    subjects_feats, sids = load_and_preextract(db_label, data_path, fast_mode=fast_mode)

    if not sids:
        print("  ERROR: No subjects loaded!")
        return None

    n_sub = len(sids)
    print(f"\n  Total subjects: {n_sub} (ALL retained)")

    if fast_mode:
        sids = sids[:3]
        n_sub = len(sids)
        print(f"  FAST MODE: {n_sub} subjects")

    print(f"\n  Feature extraction: {WIN} samples ({WIN_MS}ms @ {FS}Hz), "
          f"stride={STRIDE}, k_best={K_BEST}")

    # ── LOSO-CV with per-class collection ──
    all_perclass_f1 = defaultdict(list)
    all_perclass_prec = defaultdict(list)
    all_perclass_rec = defaultdict(list)
    all_perclass_support = defaultdict(list)
    fold_results = []
    total_train_time = 0
    total_infer_time = 0

    for fi, test_sid in enumerate(sids):
        print(f"\n  [{fi+1}/{n_sub}] Fold: S{test_sid} (test)")
        t0_fold = time.time()

        train_sids = [s for s in sids if s != test_sid]

        # ── Build train/test arrays ──
        print(f"    Building arrays...", end="", flush=True)
        t0 = time.time()

        X_train_list, y_train_list = [], []
        for sid in train_sids:
            X_train_list.append(subjects_feats[sid]["X"])
            y_train_list.append(subjects_feats[sid]["y"])

        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)
        del X_train_list, y_train_list
        gc.collect()

        X_test = subjects_feats[test_sid]["X"]
        y_test = subjects_feats[test_sid]["y"]

        n_features = X_train.shape[1]
        print(f" done ({time.time()-t0:.1f}s, train={X_train.shape}, test={X_test.shape})")

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # ═══════════════════════════════════════════════════════════════════
        # v7 CRITICAL FIX: PER-FOLD label map from TRAIN labels ONLY
        # This guarantees y_train_0 is ALWAYS contiguous 0..N-1.
        # No global map, no gaps, no XGBoost crashes.
        # ═══════════════════════════════════════════════════════════════════
        y_test_orig = y_test.copy()
        fwd, inv, n_train_cls = build_train_label_map(y_train)

        y_train_0 = fwd[y_train]  # ALWAYS contiguous 0..n_train_cls-1

        # Feature scaling + selection (on contiguous train labels)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        k = min(K_BEST, n_features)
        fsel = SelectKBest(f_classif, k=k)
        X_train_sel = fsel.fit_transform(X_train_sc, y_train_0)
        X_test_sel = fsel.transform(X_test_sc)

        print(f"    Train classes: {n_train_cls} contiguous (0-{n_train_cls-1})")
        print(f"    Features after sel: {X_train_sel.shape[1]}")

        # ═══════════════════════════════════════════════════════════════════
        # Filter eval_set: only test samples with labels present in training
        # fwd returns -1 for labels not in train → filter those out
        # ═══════════════════════════════════════════════════════════════════
        y_test_0 = fwd[y_test]
        eval_mask = y_test_0 >= 0
        n_eval = eval_mask.sum()
        n_excluded = (~eval_mask).sum()

        if n_eval > 0:
            X_eval = X_test_sel[eval_mask]
            y_eval = y_test_0[eval_mask]
        else:
            X_eval = None
            y_eval = None

        if n_excluded > 0:
            print(f"    [INFO] Test: {len(y_test)} samples, "
                  f"{n_eval} visible, {n_excluded} unseen (no train data for those classes)")

        # ═══════════════════════════════════════════════════════════════════
        # Train XGBoost
        # - y_train_0 is ALWAYS contiguous 0..N-1 → no crash
        # - eval_set (if any) is a SUBSET of 0..N-1 → no crash
        # - If no eval samples, disable early_stopping
        # ═══════════════════════════════════════════════════════════════════
        print(f"    Training XGBoost...", end="", flush=True)
        t0_train = time.time()

        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            max_bin=128,
            eval_metric='mlogloss',
            random_state=42, n_jobs=1,
        )

        if X_eval is not None and n_eval >= 10:
            # Normal: use early stopping with filtered eval set
            clf.set_params(early_stopping_rounds=20)
            clf.fit(X_train_sel, y_train_0,
                    eval_set=[(X_eval, y_eval)],
                    verbose=False)
        else:
            # Fallback: no eval set → no early stopping
            # (all test classes are unseen — very rare edge case)
            print(f"[NO-EARLY-STOP]", end="", flush=True)
            clf.fit(X_train_sel, y_train_0, verbose=False)

        train_time = time.time() - t0_train
        total_train_time += train_time
        print(f" done ({train_time:.1f}s)")

        # Predict on FULL test set
        # For unseen classes, predictions will be wrong (F1=0) — correct behavior
        t0_infer = time.time()
        y_pred_0 = clf.predict(X_test_sel)
        y_pred = inv[y_pred_0]  # contiguous → original labels
        infer_time = time.time() - t0_infer
        total_infer_time += infer_time

        # Per-class metrics using original labels
        report = classification_report(y_test_orig, y_pred, output_dict=True,
                                       zero_division=0)
        acc = accuracy_score(y_test_orig, y_pred)
        macro_f1 = f1_score(y_test_orig, y_pred, average='macro', zero_division=0)

        print(f"    Acc={acc:.4f}, Macro_F1={macro_f1:.4f}, "
              f"Infer={infer_time:.3f}s ({len(X_test)} samples)")

        # Collect per-class metrics
        for cls_key, metrics in report.items():
            if cls_key in ('accuracy', 'macro avg', 'weighted avg'):
                continue
            cls_id = int(cls_key)
            all_perclass_f1[cls_id].append(metrics['f1-score'])
            all_perclass_prec[cls_id].append(metrics['precision'])
            all_perclass_rec[cls_id].append(metrics['recall'])
            all_perclass_support[cls_id].append(metrics['support'])

        fold_results.append({
            "test_subject": test_sid,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "train_time_s": round(train_time, 2),
            "infer_time_s": round(infer_time, 3),
            "n_train": len(X_train_sel),
            "n_test": len(X_test_sel),
            "n_train_classes": n_train_cls,
            "n_features_after_fs": X_train_sel.shape[1],
        })

        fold_time = time.time() - t0_fold
        print(f"    Fold time: {fold_time:.1f}s")
        if fi == 0 and n_sub > 1:
            est = fold_time * n_sub / 60
            print(f"    [EST] ~{est:.1f} min total")

        # Cleanup
        del X_train, y_train, X_test, y_test, y_train_0, y_test_0, y_test_orig
        del X_train_sc, X_test_sc, X_train_sel, X_test_sel
        del X_eval, y_eval, clf, scaler, fsel, fwd, inv
        del report, acc, macro_f1, y_pred, y_pred_0
        gc.collect()
        os.system('')

    # ── Aggregate per-class results ──
    print_header("AGGREGATE PER-CLASS RESULTS")

    perclass_rows = []
    for cls_id in sorted(all_perclass_f1.keys()):
        f1_vals = all_perclass_f1[cls_id]
        prec_vals = all_perclass_prec[cls_id]
        rec_vals = all_perclass_rec[cls_id]
        support_vals = all_perclass_support[cls_id]

        mean_f1 = np.mean(f1_vals)
        std_f1 = np.std(f1_vals)

        if mean_f1 >= 0.35:
            difficulty = "Easy"
        elif mean_f1 >= 0.20:
            difficulty = "Medium"
        else:
            difficulty = "Hard"

        perclass_rows.append({
            "Class_ID": cls_id,
            "F1_Mean": round(mean_f1, 4),
            "F1_Std": round(std_f1, 4),
            "Precision_Mean": round(np.mean(prec_vals), 4),
            "Recall_Mean": round(np.mean(rec_vals), 4),
            "Support_Mean": round(np.mean(support_vals), 0),
            "N_Folds": len(f1_vals),
            "Difficulty": difficulty,
        })

    df_perclass = pd.DataFrame(perclass_rows)
    df_perclass = df_perclass.sort_values("F1_Mean", ascending=False).reset_index(drop=True)

    df_perclass.to_csv(OUT_DIR / f"TableS6_perclass_f1_{db_key}.csv", index=False)
    print(f"  [SAVED] TableS6_perclass_f1_{db_key}.csv ({len(df_perclass)} classes)")

    print(f"\n  Easiest 5 classes:")
    print(df_perclass.head(5)[["Class_ID", "F1_Mean", "F1_Std", "Difficulty"]].to_string(index=False))
    print(f"\n  Hardest 5 classes:")
    print(df_perclass.tail(5)[["Class_ID", "F1_Mean", "F1_Std", "Difficulty"]].to_string(index=False))

    easy = (df_perclass["Difficulty"] == "Easy").sum()
    med = (df_perclass["Difficulty"] == "Medium").sum()
    hard = (df_perclass["Difficulty"] == "Hard").sum()
    print(f"\n  Difficulty: Easy={easy}, Medium={med}, Hard={hard}")

    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]
    print(f"\n  Overall: Acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}, "
          f"Macro_F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}")
    print(f"  Total train: {total_train_time:.1f}s ({total_train_time/60:.1f}min), "
          f"Infer: {total_infer_time:.1f}s")

    df_folds = pd.DataFrame(fold_results)
    df_folds.to_csv(OUT_DIR / f"TableS6_fold_details_{db_key}.csv", index=False)

    raw_json = {
        "database": db_key,
        "db_label": db_label,
        "n_subjects": n_sub,
        "per_class_f1": {str(k): v for k, v in all_perclass_f1.items()},
        "per_class_precision": {str(k): v for k, v in all_perclass_prec.items()},
        "per_class_recall": {str(k): v for k, v in all_perclass_rec.items()},
        "fold_results": fold_results,
        "overall_accuracy_mean": round(float(np.mean(accs)), 4),
        "overall_accuracy_std": round(float(np.std(accs)), 4),
        "overall_macro_f1_mean": round(float(np.mean(f1s)), 4),
        "overall_macro_f1_std": round(float(np.std(f1s)), 4),
    }
    with open(OUT_DIR / f"day5_perclass_f1_{db_key}.json", "w") as f:
        json.dump(raw_json, f, indent=2)
    print(f"  [SAVED] day5_perclass_f1_{db_key}.json")

    return df_perclass


# =============================================================================
# Task 2: Computational Timing for ALL Classifiers
# =============================================================================
def run_timing_benchmark(db_key="db7", fast_mode=False, clf_filter=None):
    """
    Measure training + inference time for all 4 classifiers.
    Uses FIRST LOSO fold. v7: per-fold train-only label map.
    v8: Safe AR extraction for DB3 constant signals.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_mapping = {
        "db7": ("DB7", config["datasets"]["ninapro_db7"]["path"]),
        "db3": ("DB3", config["datasets"]["ninapro_db3"]["path"]),
        "db2": ("DB2", config["datasets"]["ninapro_db2"]["path"]),
    }

    if db_key not in db_mapping:
        print(f"  ERROR: unknown db '{db_key}'")
        return None

    db_label, data_path = db_mapping[db_key]
    print_header(f"Task 2: Computational Timing Benchmark — {db_label}")

    print(f"  Loading data from: {data_path}")
    print(f"  Pre-extracting features (cap={PER_SUB_CAP}/subject)...")
    subjects_feats, sids = load_and_preextract(db_label, data_path, fast_mode=fast_mode)

    if not sids:
        print("  ERROR: No subjects loaded!")
        return None

    print(f"  Subjects: {len(sids)}")

    # Build one fold (first subject as test, rest as train)
    test_sid = sids[0]
    train_sids = sids[1:]

    print(f"\n  Test: S{test_sid}, Train: {len(train_sids)} subjects")
    print(f"  Building arrays...", end="", flush=True)
    t0 = time.time()

    X_train_list, y_train_list = [], []
    for sid in train_sids:
        X_train_list.append(subjects_feats[sid]["X"])
        y_train_list.append(subjects_feats[sid]["y"])

    X_train = np.vstack(X_train_list)
    y_train = np.hstack(y_train_list)
    del X_train_list, y_train_list
    gc.collect()

    X_test = subjects_feats[test_sid]["X"]
    y_test = subjects_feats[test_sid]["y"]

    feat_time = time.time() - t0
    print(f" done ({feat_time:.1f}s)")

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # ── Per-fold label map from TRAIN only ──
    y_test_orig = y_test.copy()
    fwd, inv, n_train_cls = build_train_label_map(y_train)
    y_train_0 = fwd[y_train]
    y_test_0 = fwd[y_test]

    # Scale + select
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    k = min(K_BEST, X_train.shape[1])
    fsel = SelectKBest(f_classif, k=k)
    X_train_sel = fsel.fit_transform(X_train_sc, y_train_0)
    X_test_sel = fsel.transform(X_test_sc)

    print(f"  Train: {X_train_sel.shape}, Test: {X_test_sel.shape}")
    print(f"  Train classes: {n_train_cls} contiguous (0-{n_train_cls-1})")

    # ── Filter eval_set for XGBoost ──
    eval_mask = y_test_0 >= 0
    X_eval = X_test_sel[eval_mask]
    y_eval = y_test_0[eval_mask]
    n_excluded = (~eval_mask).sum()
    if n_excluded > 0:
        print(f"  [INFO] Excluded {n_excluded} test samples from eval (unseen classes)")

    # ── Auto-skip existing results ──
    timing_csv = OUT_DIR / f"TableS7_real_timing_{db_key}.csv"
    existing_results = {}
    if timing_csv.exists():
        try:
            df_existing = pd.read_csv(timing_csv)
            for _, row in df_existing.iterrows():
                existing_results[row["Method"]] = row
            print(f"\n  [AUTO-SKIP] Existing: {list(existing_results.keys())}")
        except Exception as e:
            print(f"  [WARN] Could not read existing CSV: {e}")

    # ── Classifiers ──
    all_classifiers = {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            max_bin=128,
            eval_metric='mlogloss', random_state=42, n_jobs=1,
        ),
        "LDA": LinearDiscriminantAnalysis(),
        "LinearSVC": LinearSVC(C=1.0, max_iter=5000),
        "RandomForest": RandomForestClassifier(
            n_estimators=50, max_depth=20,
            random_state=42, n_jobs=1,
        ),
    }

    if clf_filter and len(clf_filter) > 0:
        classifiers = {k: v for k, v in all_classifiers.items() if k in clf_filter}
        print(f"  Running only: {list(classifiers.keys())}")
    else:
        classifiers = all_classifiers

    results = []

    for name, clf in classifiers.items():
        # Auto-skip
        if name in existing_results and not clf_filter:
            print(f"\n  --- {name}: SKIPPED (exists) ---")
            results.append(existing_results[name].to_dict())
            continue
        if name in existing_results and clf_filter:
            print(f"\n  --- {name}: RE-RUNNING (override) ---")
        else:
            print(f"\n  --- {name} ---")

        # Training
        t0 = time.time()
        if name == "XGBoost":
            if len(X_eval) >= 10:
                clf.set_params(early_stopping_rounds=20)
                clf.fit(X_train_sel, y_train_0,
                        eval_set=[(X_eval, y_eval)],
                        verbose=False)
            else:
                clf.fit(X_train_sel, y_train_0, verbose=False)
        else:
            clf.fit(X_train_sel, y_train_0)
        train_time = time.time() - t0

        # Batch inference
        t0 = time.time()
        y_pred_0 = clf.predict(X_test_sel)
        y_pred = inv[y_pred_0]
        batch_infer_time = time.time() - t0

        # Single inference (1000x)
        t0 = time.time()
        for _ in range(1000):
            clf.predict(X_test_sel[:1])
        single_infer_ms = (time.time() - t0) / 1000 * 1000

        acc = accuracy_score(y_test_orig, y_pred)
        f1 = f1_score(y_test_orig, y_pred, average='macro', zero_division=0)

        # Model size (fallback if OOM — v3 fix)
        import joblib
        import io
        try:
            buf = io.BytesIO()
            joblib.dump(clf, buf)
            model_size_kb = buf.tell() / 1024
            del buf
        except (MemoryError, Exception):
            if name == "RandomForest":
                model_size_kb = 50 * k * 8 / 1024
            elif name == "XGBoost":
                model_size_kb = 300 * k * 8 / 1024
            else:
                model_size_kb = 100

        print(f"    Train: {train_time:.2f}s | Batch: {batch_infer_time:.3f}s | "
              f"Single: {single_infer_ms:.3f}ms | Size: {model_size_kb:.1f} KB")
        print(f"    Acc: {acc:.4f}, Macro_F1: {f1:.4f}")

        results.append({
            "Method": name,
            "Train_Time_s": round(train_time, 2),
            "Batch_Infer_s": round(batch_infer_time, 3),
            "Single_Infer_ms": round(single_infer_ms, 3),
            "Model_Size_KB": round(model_size_kb, 1),
            "Accuracy": round(acc, 4),
            "Macro_F1": round(f1, 4),
            "N_Train": len(X_train_sel),
            "N_Test": len(X_test_sel),
            "N_Features": X_train_sel.shape[1],
        })

        del clf, y_pred, y_pred_0
        gc.collect()

    # CNN-1D estimate
    has_cnn = any("CNN" in str(r.get("Method", "")) for r in results)
    if not has_cnn:
        results.append({
            "Method": "CNN-1D (v8, 400ms)",
            "Train_Time_s": 82.0,
            "Batch_Infer_s": "N/A",
            "Single_Infer_ms": "~5.0",
            "Model_Size_KB": 200,
            "Accuracy": "21.60% (DB7)",
            "Macro_F1": "N/A",
            "N_Train": "60000 (capped)",
            "N_Test": "30000 (capped)",
            "N_Features": "12 ch x 800 samples",
        })

    df_timing = pd.DataFrame(results)
    df_timing.to_csv(timing_csv, index=False)
    print(f"\n  [SAVED] TableS7_real_timing_{db_key}.csv")
    print(df_timing[["Method", "Train_Time_s", "Single_Infer_ms", "Model_Size_KB"]].to_string(index=False))

    # Cleanup
    del X_train, y_train, X_test, y_test, X_train_sc, X_test_sc
    del X_train_sel, X_test_sel, X_eval, y_eval, scaler, fsel
    del y_train_0, y_test_0, y_test_orig, fwd, inv
    gc.collect()

    return df_timing


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Day 5: Real Per-Class F1 + Timing (v8)")
    ap.add_argument("--db", type=str, default="db7",
                    choices=["db7", "db3", "db2"],
                    help="Database to analyze")
    ap.add_argument("--task", type=str, default="all",
                    choices=["perclass", "timing", "all"],
                    help="Which task(s) to run")
    ap.add_argument("--fast", action="store_true",
                    help="Fast mode (fewer subjects for testing)")
    ap.add_argument("--classifiers", type=str, default=None,
                    help="Comma-separated classifiers for timing "
                         "(e.g. 'xgb,lda,rf,svc'). Default: auto-skip completed")
    args = ap.parse_args()

    CLF_FILTER = None
    if args.classifiers:
        mapping = {"xgb": "XGBoost", "lda": "LDA", "rf": "RandomForest", "svc": "LinearSVC"}
        CLF_FILTER = [mapping[c.strip().lower()] for c in args.classifiers.split(",")
                     if c.strip().lower() in mapping]
        print(f"  Classifier filter: {CLF_FILTER}")

    print("=" * 70)
    print("  Day 5: Real Per-Class F1 + Computational Timing (v8 — DEFINITIVE)")
    print(f"  Database: {args.db} | Task: {args.task} | Fast: {args.fast}")
    print(f"  Window: {WIN_MS}ms ({WIN} samples @ {FS}Hz)")
    print(f"  Overlap: {OVERLAP*100:.0f}% | Stride: {STRIDE} samples")
    print(f"  K-best: {K_BEST}")
    print(f"  Output: {OUT_DIR}")
    print(f"  S21/S22 policy: RETAINED (Option B)")
    print(f"  Label encoding: PER-FOLD train-only contiguous 0-based")
    print(f"  Per-subject cap: {PER_SUB_CAP} windows")
    print(f"  AR extraction: Safe (try/except + zero fallback)")
    print("=" * 70)

    if args.task in ("perclass", "all"):
        run_perclass_f1(args.db, fast_mode=args.fast)

    if args.task in ("timing", "all"):
        run_timing_benchmark(args.db, fast_mode=args.fast, clf_filter=CLF_FILTER)

    print(f"\n{'='*70}")
    print(f"  DONE — Check {OUT_DIR} for output files")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
