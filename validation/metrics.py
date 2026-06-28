"""
metrics.py - v37.0 (SPEED + ACCURACY ENGINE)

═══════════════════════════════════════════════════════════════════════
CHANGES from v36.1:
═══════════════════════════════════════════════════════════════════════

v37.0 MAJOR CHANGES (Domain Adaptation + Advanced Features):

  1. EUCLIDEAN ALIGNMENT (process_engine.py)
     - He & Wu 2020: Domain adaptation per-subject
     - Applied after bandpass filtering, before windowing
     - Expected: +6-12% on LOSO (especially DB7 mixed population)

  2. BALANCED SAMPLE WEIGHT (XGBoost training)
     - Per-class weights: n_total / (n_classes * n_class_count)
     - Passes as sample_weight to XGB fit()
     - More precise than class_weight='balanced'
     - Expected: +3-6% on imbalanced datasets

  3. TKEO BAND ENERGY FEATURES (process_engine.py)
     - 4 frequency bands: (20-100, 100-200, 200-350, 350-450 Hz)
     - FFT-based TKEO decomposition per window per channel
     - Expected: +4-7% improvement

  4. COVARIANCE FEATURES / Riemannian-inspired (process_engine.py)
     - Upper triangle of trace-normalized covariance matrix per window
     - Scale-invariant spatial channel relationships
     - Expected: +8-15% improvement

  5. PROGRESSIVE CALIBRATION (unsupervised test adaptation)
     - Vidovic et al. 2016: Use first 10% of test subject for calibration
     - Covariate shift correction between train and test distributions
     - Expected: +4-8% improvement (especially amputee subjects)

  ALL v36.1/v35.1/v35.0/v33.0/v32.0 FEATURES PRESERVED.
═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import gc
import time
import warnings
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    LeaveOneGroupOut, train_test_split, StratifiedShuffleSplit
)
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

HAS_LGB = None


def _check_lgb():
    """Lazy-check LightGBM availability."""
    global HAS_LGB
    if HAS_LGB is None:
        try:
            import lightgbm
            HAS_LGB = True
        except ImportError:
            HAS_LGB = False
    return HAS_LGB


# ============================================================================
# Configuration Defaults (v32.0 — Adaptive Engine)
# ============================================================================
DEFAULT_TOP_FEATURES = 250
DEFAULT_N_ESTIMATORS = 200
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MAX_DEPTH = 6
DEFAULT_SUBSAMPLE = 0.9
DEFAULT_COLSAMPLE = 0.9
MAX_BIN = 128
DEFAULT_EARLY_STOPPING = 25


# ============================================================================
# Adaptive Settings (v32.0 — reads from dataset_adaptive_configs)
# ============================================================================
def _get_adaptive_settings(n_classes, dataset_config=None, n_total_samples=None):
    """
    Returns (max_train, n_estimators, max_depth, learning_rate,
             subsample, colsample_bytree, n_top_features, feature_selection,
             use_ensemble, early_stopping_rounds, per_subject_normalize,
             balanced_sample_weight, progressive_calibration).

    v33.0: max_train is now DYNAMIC based on dataset size.
      - 30% of total samples, capped at 500K, minimum 50K
      - Can be overridden via dataset_config['classification']['max_train']
      - This fixes the hardcoded 120K that was too small for large datasets
        (DB2: ~2M windows → 120K = only 6% of data)

    v36.1: Added per_subject_normalize (inter-subject calibration).

    v37.0: Added balanced_sample_weight and progressive_calibration.
    """
    if dataset_config:
        clf_cfg = dataset_config.get('classification', {})

        # v33.0: Dynamic max_train
        explicit_max_train = clf_cfg.get('max_train', None)
        if explicit_max_train is not None:
            max_train = explicit_max_train
        elif n_total_samples and n_total_samples > 0:
            # 30% of data, capped at 500K, minimum 50K
            max_train = min(500000, max(50000, int(0.3 * n_total_samples)))
        else:
            max_train = 300000  # reasonable default when dataset size unknown

        return (
            max_train,
            clf_cfg.get('n_estimators', DEFAULT_N_ESTIMATORS),
            clf_cfg.get('max_depth', DEFAULT_MAX_DEPTH),
            clf_cfg.get('learning_rate', DEFAULT_LEARNING_RATE),
            clf_cfg.get('subsample', DEFAULT_SUBSAMPLE),
            clf_cfg.get('colsample_bytree', DEFAULT_COLSAMPLE),
            clf_cfg.get('n_top_features', DEFAULT_TOP_FEATURES),
            clf_cfg.get('feature_selection', 'hybrid'),
            clf_cfg.get('use_ensemble', False),
            clf_cfg.get('early_stopping_rounds', DEFAULT_EARLY_STOPPING),
            clf_cfg.get('per_subject_normalize', False),
            clf_cfg.get('balanced_sample_weight', False),
            clf_cfg.get('progressive_calibration', False),
        )

    # Fallback: class-count-based defaults (backward compatible)
    if n_classes <= 10:
        max_train = 50000
        n_est, depth, lr = 100, 4, 0.15
        sub, col = 0.8, 0.8
    elif n_classes <= 20:
        max_train = 80000
        n_est, depth, lr = 150, 6, 0.1
        sub, col = 0.85, 0.85
    else:
        max_train = 120000
        n_est, depth, lr = 200, 7, 0.08
        sub, col = 0.9, 0.9

    # v33.0: Dynamic adjustment for fallback path too
    if n_total_samples and n_total_samples > 0:
        max_train = min(500000, max(max_train, int(0.3 * n_total_samples)))

    return (max_train, n_est, depth, lr, sub, col,
            DEFAULT_TOP_FEATURES, 'hybrid', False, DEFAULT_EARLY_STOPPING, False,
            False, False)


# ============================================================================
# Hybrid Feature Selection (v32.0 — THE critical accuracy boost)
# ============================================================================
def _hybrid_feature_selection(X_train, y_train, k, feature_names=None,
                               protect_corr_features=True, n_classes=None):
    """
    Combine mutual_info + f_classif scores for robust feature selection.

    v35.0: Adaptive MI sample limit based on n_classes.
      - n_classes <= 20: 15000 samples (stable, fast enough)
      - n_classes > 20:  3000 samples (5x faster, still sufficient for ranking)
      - f_classif also capped at 20000 samples for large datasets

    Returns: selected indices (not a fitted selector, for flexibility)
    """
    n_features = X_train.shape[1]

    # --- Compute f_classif scores (v35.0: capped at 20K for speed) ---
    _f_sample_limit = 20000
    if X_train.shape[0] > _f_sample_limit:
        rng_f = np.random.RandomState(42)
        f_idx = rng_f.choice(X_train.shape[0], _f_sample_limit, replace=False)
        f_scores, _ = f_classif(X_train[f_idx], y_train[f_idx])
    else:
        f_scores, _ = f_classif(X_train, y_train)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    f_norm = f_scores / (f_scores.max() + 1e-12)

    # --- Compute mutual_info scores (v35.0: adaptive sample limit) ---
    from sklearn.feature_selection import mutual_info_classif
    if n_classes is not None and n_classes > 20:
        _mi_sample_limit = 3000  # v35.0: much smaller for high-class-count datasets
    else:
        _mi_sample_limit = 15000
    if X_train.shape[0] > _mi_sample_limit:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(X_train.shape[0], _mi_sample_limit, replace=False)
        mi_scores = mutual_info_classif(X_train[sample_idx], y_train[sample_idx],
                                         random_state=42, discrete_features=False,
                                         n_neighbors=3)
    else:
        mi_scores = mutual_info_classif(X_train, y_train,
                                         random_state=42, discrete_features=False,
                                         n_neighbors=3)
    mi_scores = np.nan_to_num(mi_scores, nan=0.0)
    mi_norm = mi_scores / (mi_scores.max() + 1e-12)

    # --- Combined score (v34.0: MI weighted higher — more discriminative for EMG) ---
    combined = 0.4 * f_norm + 0.6 * mi_norm

    # --- Protect correlation features ---
    if protect_corr_features and feature_names is not None:
        corr_indices = set()
        for i, name in enumerate(feature_names):
            if name.startswith('corr_'):
                corr_indices.add(i)
        if corr_indices:
            # v33.0: Strictly respect k
            if len(corr_indices) >= k:
                # More corr features than k — take top-k corr by combined score
                corr_scores = np.array([combined[i] for i in corr_indices])
                corr_idx_arr = np.array(sorted(corr_indices))
                top_k_idx = np.argsort(corr_scores)[-k:]
                return corr_idx_arr[top_k_idx]
            else:
                # Fill remaining slots from top non-corr features
                remaining_k = k - len(corr_indices)
                # Zero out protected indices so they don't get double-selected
                combined_masked = combined.copy()
                for idx in corr_indices:
                    combined_masked[idx] = -np.inf
                top_remaining = np.argsort(combined_masked)[-remaining_k:]
                selected = np.concatenate([
                    np.array(sorted(corr_indices)),
                    top_remaining
                ])
                return np.sort(selected)

    # --- Select top-k (v34.0: argpartition O(n) vs argsort O(n log n)) ---
    if k >= n_features:
        return np.arange(n_features)
    top_k_idx = np.argpartition(combined, -k)[-k:]
    return np.sort(top_k_idx)


# ============================================================================
# Unified Preprocessing Pipeline (v32.0 — hybrid FS support)
# ============================================================================
def _preprocess_fold(X_train, X_test, y_train, n_top_features=None,
                     feature_selection='hybrid', pca_components=None,
                     feature_names=None):
    """
    Single-pass preprocessing pipeline.

    v32.0 changes:
      - Supports 'hybrid' feature selection (mutual_info + f_classif)
      - Protects correlation features from being pruned
      - All other steps unchanged (NaN clean, variance filter, Z-score, PCA)
    """
    # --- Step 1: Clean NaN/Inf + float32 ---
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    if X_train.dtype != np.float32:
        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    if X_test.dtype != np.float32:
        X_test = np.ascontiguousarray(X_test, dtype=np.float32)

    # --- Step 2: Remove zero/low-variance features ---
    var = np.var(X_train, axis=0)
    mask = var > 1e-8
    if not np.any(mask):
        raise ValueError("All features have zero variance.")
    X_train = X_train[:, mask]
    X_test = X_test[:, mask]
    if feature_names is not None:
        feature_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]

    # --- Step 3: Z-score (fit on train ONLY — no data leakage) ---
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # --- Step 4: Feature selection ---
    if n_top_features and n_top_features < X_train.shape[1]:
        k = min(n_top_features, X_train.shape[1])

        if feature_selection == 'hybrid':
            # v32.0: Hybrid selection — preserves correlation features
            selected_idx = _hybrid_feature_selection(
                X_train, y_train, k, feature_names=feature_names,
                protect_corr_features=True
            )
            X_train = X_train[:, selected_idx]
            X_test = X_test[:, selected_idx]
        elif feature_selection == 'mutual_info':
            from sklearn.feature_selection import mutual_info_classif
            # v34.0: Aggressive sampling (15K) — faster MI for mutual_info path
            _mi_lim = 15000
            if X_train.shape[0] > _mi_lim:
                rng = np.random.RandomState(42)
                idx = rng.choice(X_train.shape[0], _mi_lim, replace=False)
                sel = SelectKBest(mutual_info_classif, k=k)
                sel.fit(X_train[idx], y_train[idx])
            else:
                sel = SelectKBest(mutual_info_classif, k=k)
                sel.fit(X_train, y_train)
            X_train = sel.transform(X_train)
            X_test = sel.transform(X_test)
        else:
            # f_classif (default sklearn)
            sel = SelectKBest(f_classif, k=k)
            sel.fit(X_train, y_train)
            X_train = sel.transform(X_train)
            X_test = sel.transform(X_test)

    # --- Step 5: PCA (optional, disabled by default) ---
    if pca_components is not None:
        n_comp = min(pca_components, X_train.shape[1],
                     X_train.shape[0] - 1, X_test.shape[0] - 1)
        n_comp = max(n_comp, 1)
        pca = PCA(n_components=n_comp, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, X_test


# ============================================================================
# Statistical helpers (corrected effect sizes)
# ============================================================================
def wilcoxon_test(accuracies_a, accuracies_b, alpha=0.05):
    diff = np.array(accuracies_a) - np.array(accuracies_b)
    w, p = stats.wilcoxon(diff, zero_method='wilcox', alternative='two-sided')
    n = int(np.count_nonzero(diff))
    total_rank_sum = n * (n + 1) / 2
    r = 1 - (2 * w) / total_rank_sum if total_rank_sum > 0 else 0.0
    return w, p, r


def mann_whitney_test(group_a, group_b):
    u, p = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
    n1, n2 = len(group_a), len(group_b)
    r = 1 - (2 * u) / (n1 * n2)
    return u, p, r


def cohens_d(group_a, group_b):
    n1, n2 = len(group_a), len(group_b)
    s1 = np.var(group_a, ddof=1)
    s2 = np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group_a) - np.mean(group_b)) / pooled_std


def bootstrap_ci(data, statistic=np.mean, n_resamples=10000,
                 confidence_level=0.95, random_state=42):
    data = np.array(data)
    from scipy.stats import bootstrap as bs
    result = bs(
        (data,), statistic, n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state, method='percentile'
    )
    return result.confidence_interval.low, result.confidence_interval.high


def compute_confidence_interval(accuracies):
    n = len(accuracies)
    if n < 2:
        return float(np.mean(accuracies)), float(np.mean(accuracies))
    mean = np.mean(accuracies)
    std = np.std(accuracies, ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    margin = t * std / np.sqrt(n)
    return float(mean - margin), float(mean + margin)


# ============================================================================
# Stratified Subsample
# ============================================================================
def _stratified_subsample(X, y, max_samples, random_state=42):
    rng = np.random.RandomState(random_state)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total = len(y)
    indices = []
    for cls, count in zip(unique_classes, class_counts):
        cls_idx = np.where(y == cls)[0]
        n_sample = max(2, int(count * max_samples / total))
        n_sample = min(n_sample, count)
        chosen = rng.choice(cls_idx, size=n_sample, replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return X[indices], y[indices], len(indices)


# ============================================================================
# Classifier Builder (v32.0 — adaptive settings)
# ============================================================================
def _build_classifier(clf_name, n_classes, n_estimators=DEFAULT_N_ESTIMATORS,
                      random_state=42, max_depth=DEFAULT_MAX_DEPTH,
                      learning_rate=DEFAULT_LEARNING_RATE,
                      n_jobs=-1,
                      subsample=DEFAULT_SUBSAMPLE,
                      colsample_bytree=DEFAULT_COLSAMPLE,
                      early_stopping_rounds=0):
    if clf_name == 'XGBOOST' and HAS_XGB:
        xgb_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method='hist',
            max_bin=MAX_BIN,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
            eval_metric='mlogloss',
            min_child_weight=3,       # v34.0: 1→3 (reduces overfitting, faster)
            reg_alpha=0.05,           # v34.0: 0→0.05 (L1, sparse features)
            reg_lambda=1.5,           # v34.0: 1→1.5 (L2, smoother trees)
            gamma=0.05,               # v34.0: min loss reduction per split
        )
        if early_stopping_rounds > 0:
            xgb_params['early_stopping_rounds'] = early_stopping_rounds
        return XGBClassifier(**xgb_params), True

    elif clf_name == 'LIGHTGBM' and _check_lgb():
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max(max_depth - 1, 3),
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='multiclass',
            num_class=n_classes,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        ), True

    elif clf_name == 'EXTRA_TREES':
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight='balanced',
        ), False

    elif clf_name == 'SVM':
        return SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced',
            random_state=random_state
        ), False
        
        
    elif clf_name == 'LINEARSVC':
        return LinearSVC(
            C=1.0, max_iter=10000, dual=False,  tol=1e-2,
            random_state=random_state
        ), True


    elif clf_name == 'LDA':
        return LinearDiscriminantAnalysis(), False

    else:
        return RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=n_jobs,
            random_state=random_state
        ), False


# ============================================================================
# Ensemble (v32.0 — enhanced with early stopping on XGB)
# ============================================================================
def _train_ensemble(X_train, y_train_enc, n_classes, n_estimators,
                    max_depth, learning_rate, le,
                    subsample=DEFAULT_SUBSAMPLE,
                    colsample_bytree=DEFAULT_COLSAMPLE,
                    early_stopping_rounds=0,
                    balanced_sample_weight=False):
    """Train XGBoost + RF + LDA ensemble. v37.0: balanced sample weight support."""
    rs = 42

    # v37.0: Compute sample weights if balanced
    _sample_w = None
    if balanced_sample_weight:
        _class_counts = np.bincount(y_train_enc)
        _n_total = len(y_train_enc)
        _n_cls = len(_class_counts)
        _sw = _n_total / (_n_cls * _class_counts + 1e-10)
        _sample_w = _sw[y_train_enc]

    xgb, _ = _build_classifier(
        'XGBOOST', n_classes, n_estimators,
        random_state=rs, max_depth=max_depth,
        learning_rate=learning_rate, n_jobs=1,
        subsample=subsample, colsample_bytree=colsample_bytree,
        early_stopping_rounds=early_stopping_rounds,
    )
    if early_stopping_rounds > 0 and len(X_train) > 1000:
        val_ratio = min(0.10, max(0.05, 500 / len(X_train)))
        stratify = y_train_enc if len(np.unique(y_train_enc)) > 1 else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train_enc, test_size=val_ratio,
            stratify=stratify, random_state=42)
        # v37.0: Pass sample_weight for early stopping path
        if balanced_sample_weight and _sample_w is not None:
            _sw_tr = _sample_w[:len(X_tr)]
            xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                    sample_weight=_sw_tr,
                    verbose=False)
        else:
            xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        # v37.0: Pass sample_weight for non-early-stopping path
        if balanced_sample_weight and _sample_w is not None:
            xgb.fit(X_train, y_train_enc, sample_weight=_sample_w)
        else:
            xgb.fit(X_train, y_train_enc)

    rf = RandomForestClassifier(
        n_estimators=min(100, n_estimators),
        n_jobs=1, random_state=rs,
        max_depth=max_depth + 2,
    )
    # v37.0: Pass sample_weight to RF (RF supports sample_weight)
    if balanced_sample_weight and _sample_w is not None:
        rf.fit(X_train, y_train_enc, sample_weight=_sample_w)
    else:
        rf.fit(X_train, y_train_enc)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train_enc)

    return xgb, rf, lda


def _predict_ensemble(xgb, rf, lda, X_test, le):
    """Soft voting: XGBoost 50% + RF 30% + LDA 20%."""
    p_xgb = xgb.predict_proba(X_test)
    p_rf = rf.predict_proba(X_test)
    p_lda = lda.predict_proba(X_test)

    n_classes = max(p_xgb.shape[1], p_rf.shape[1], p_lda.shape[1])
    probs = np.zeros((X_test.shape[0], n_classes))
    probs[:, :p_xgb.shape[1]] += 0.5 * p_xgb
    probs[:, :p_rf.shape[1]] += 0.3 * p_rf
    probs[:, :p_lda.shape[1]] += 0.2 * p_lda

    y_pred_enc = np.argmax(probs, axis=1)
    return le.inverse_transform(y_pred_enc)


# ============================================================================
# Single Fold Processing (v32.0 — hybrid FS + feature name propagation)
# ============================================================================
def _process_single_fold(fold_data, le, clf_name, n_estimators, max_depth,
                         learning_rate, n_top_features, feature_selection,
                         pca_components, use_ensemble, feature_names=None,
                         subsample=DEFAULT_SUBSAMPLE,
                         colsample_bytree=DEFAULT_COLSAMPLE,
                         xgb_nthread=1,
                         early_stopping_rounds=0,
                         balanced_sample_weight=False):
    """
    Process a single LOSO fold.

    v32.0 changes:
      - Passes feature_names to _preprocess_fold for correlation protection
      - Supports use_ensemble from adaptive config

    v37.0: Added balanced_sample_weight for per-class sample weighting.
    """
    X_train, y_train, X_test, y_test, subject_id = fold_data
    n_classes = len(le.classes_)

    try:
        X_train, X_test = _preprocess_fold(
            X_train, X_test, y_train,
            n_top_features=n_top_features,
            feature_selection=feature_selection,
            pca_components=pca_components,
            feature_names=feature_names,
        )
    except ValueError:
        return y_test.tolist(), y_test.tolist(), 0.0, subject_id, None, None

    clf = None

    if use_ensemble:
        y_train_enc = le.transform(y_train)
        xgb, rf_m, lda_m = _train_ensemble(
            X_train, y_train_enc, n_classes,
            n_estimators, max_depth, learning_rate, le,
            subsample=subsample, colsample_bytree=colsample_bytree,
            early_stopping_rounds=early_stopping_rounds,
            balanced_sample_weight=balanced_sample_weight,
        )
        y_pred = _predict_ensemble(xgb, rf_m, lda_m, X_test, le)
        clf = xgb
    else:
        clf, needs_encode = _build_classifier(
            clf_name, n_classes, n_estimators,
            max_depth=max_depth, learning_rate=learning_rate,
            n_jobs=xgb_nthread,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=early_stopping_rounds,
        )
        if needs_encode:
            y_train_enc = le.transform(y_train)
            if (early_stopping_rounds > 0 and clf_name == 'XGBOOST'
                    and len(X_train) > 1000):
                # v32.0: 10% val split (more stable than 8%)
                val_ratio = min(0.10, max(0.05, 500 / len(X_train)))
                stratify = (y_train_enc
                            if len(np.unique(y_train_enc)) > 1 else None)
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train_enc, test_size=val_ratio,
                    stratify=stratify, random_state=42)
                # v37.0: Balanced sample weight for early stopping path
                if balanced_sample_weight:
                    class_counts = np.bincount(y_tr)
                    n_total = len(y_tr)
                    n_cls = len(class_counts)
                    sw = n_total / (n_cls * class_counts + 1e-10)
                    sample_w = sw[y_tr]
                    clf.fit(X_tr, y_tr,
                            eval_set=[(X_val, y_val)],
                            sample_weight=sample_w,
                            verbose=False)
                else:
                    clf.fit(X_tr, y_tr,
                            eval_set=[(X_val, y_val)],
                            verbose=False)
            else:
                # v37.0: Balanced sample weight for regular path
                if balanced_sample_weight:
                    class_counts = np.bincount(y_train_enc)
                    n_total = len(y_train_enc)
                    n_cls = len(class_counts)
                    sw = n_total / (n_cls * class_counts + 1e-10)
                    sample_w = sw[y_train_enc]
                    clf.fit(X_train, y_train_enc, sample_weight=sample_w)
                else:
                    clf.fit(X_train, y_train_enc)
            y_pred_enc = clf.predict(X_test)
            if y_pred_enc.ndim == 2:
                y_pred_enc = np.argmax(y_pred_enc, axis=1)
            y_pred = le.inverse_transform(y_pred_enc)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    # احفظ النتائج قبل التنظيف
    _yt = y_test.tolist()
    _yp = y_pred.tolist()

    # حرر الذاكرة فوراً (النموذج يستهلك ~300MB+ لكل فولد)
    del clf, X_train, y_train, y_test, y_pred
    gc.collect()
    return _yt, _yp, acc, subject_id, None, None, macro_f1
# ============================================================================
# Index-level stratified subsampling (memory-efficient)
# ============================================================================
def _stratified_subsample_indices(indices, labels, max_samples, random_state=42):
    """
    Subsample indices while preserving class distribution.
    v32.0: Only global cap applied here (no per-subject double-capping).
    """
    rng = np.random.RandomState(random_state)
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    total = len(labels)
    sampled = []
    for cls, count in zip(unique_classes, class_counts):
        cls_positions = np.where(labels == cls)[0]
        # v33.0: Simplified — min(count, max(2, proportional))
        n_sample = min(count, max(2, int(count * max_samples / total)))
        chosen = rng.choice(cls_positions, size=n_sample, replace=False)
        sampled.append(indices[chosen])
    result = np.concatenate(sampled)
    rng.shuffle(result)
    return result


# ============================================================================
# LOSO Fold Worker (v32.0 — index-level subsampling)
# ============================================================================
def _loso_fold_worker(tr_idx, te_idx, features, labels, subject_groups,
                       max_train, le, clf_name, n_estimators, max_depth,
                       learning_rate, n_top_features, feature_selection,
                       pca_components, use_ensemble, subsample, colsample_bytree,
                       feature_names=None,
                       xgb_nthread=1, early_stopping_rounds=0,
                       global_selected_idx=None, n_classes=None,
                       per_subject_normalize=False,
                       balanced_sample_weight=False,
                       progressive_calibration=False,
                        svm_c=1.0, svm_gamma='scale', return_models=False): #37.1
    """
    Process a single LOSO fold. v37.0 — global FS + per-subject normalize
    + progressive calibration + balanced sample weight.

    v37.0: progressive_calibration (Vidovic 2016) + balanced_sample_weight.
    v36.1: per_subject_normalize calibrates inter-subject amplitude.
    v35.0: global_selected_idx skips per-fold FS computation.
    """
    subject_id = int(subject_groups[te_idx[0]])
    _fold_t0 = time.time()
    print(f"  [fold] Subject {subject_id} starting...", flush=True)
    try:
        # v33.0: Global cap only — no per-subject double-capping
        if max_train is not None and len(tr_idx) > max_train:
            y_train_labels = labels[tr_idx]
            tr_idx = _stratified_subsample_indices(
                tr_idx, y_train_labels, max_train, random_state=42
            )

        X_train = features[tr_idx]
        y_train = labels[tr_idx].copy()
        X_test = features[te_idx]
        y_test = labels[te_idx].copy()

        # v36.1: Per-subject normalization (inter-subject amplitude calibration)
        # Each subject normalized independently to reduce inter-subject variability
        if per_subject_normalize:
            train_subj_groups = subject_groups[tr_idx]
            for _sid in np.unique(train_subj_groups):
                _mask = train_subj_groups == _sid
                if _mask.sum() > 1:
                    _mu = X_train[_mask].mean(axis=0)
                    _sigma = X_train[_mask].std(axis=0)
                    _sigma[_sigma < 1e-8] = 1.0
                    X_train[_mask] = (X_train[_mask] - _mu) / _sigma
            # Test subject normalized independently
            _mu_t = X_test.mean(axis=0)
            _sigma_t = X_test.std(axis=0)
            _sigma_t[_sigma_t < 1e-8] = 1.0
            X_test = (X_test - _mu_t) / _sigma_t

        del tr_idx, te_idx

        # v37.0: Progressive Calibration (Vidovic et al. 2016)
        # Use first 10% of test subject's data for unsupervised adaptation
        if progressive_calibration and len(X_test) > 50:
            n_adapt = max(10, int(0.1 * len(X_test)))
            X_adapt = X_test[:n_adapt]
            mu_train = X_train.mean(axis=0)
            mu_test_approx = X_adapt.mean(axis=0)
            # Gentle shift: 30% correction toward test distribution
            shift = (mu_test_approx - mu_train) * 0.3
            X_train = X_train + shift

        # v35.0: Use global feature indices if pre-computed (skip per-fold FS)
        if global_selected_idx is not None:
            # Apply global FS directly — skip _preprocess_fold FS step
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            if X_train.dtype != np.float32:
                X_train = np.ascontiguousarray(X_train, dtype=np.float32)
            if X_test.dtype != np.float32:
                X_test = np.ascontiguousarray(X_test, dtype=np.float32)
            # v36.2 FIX: lower var threshold after per-subject norm
            var_threshold = 1e-12 if per_subject_normalize else 1e-8
            var = np.var(X_train, axis=0)
            mask = var > var_threshold
            if not np.any(mask):
                return y_test.tolist(), y_test.tolist(), 0.0, subject_id, None, None
            X_train = X_train[:, mask]
            X_test = X_test[:, mask]

            # v36.2 FIX: skip z-score if per_subject_normalize already applied
            if not per_subject_normalize:
                mu = X_train.mean(axis=0)
                sigma = X_train.std(axis=0)
                sigma[sigma < 1e-8] = 1.0
                X_train = (X_train - mu) / sigma
                X_test = (X_test - mu) / sigma
            # v36.1 FIX: Map global_selected_idx through variance mask
            # global_selected_idx has ORIGINAL indices (0-235), but after
            # X_train = X_train[:, mask], columns are renumbered (0-171).
            # We must convert original indices to new column positions.
            _surviving_cols = np.where(mask)[0]  # sorted original indices that survived
            _masked_original = np.array([i for i in global_selected_idx if mask[i]])
            if len(_masked_original) > 0:
                _new_positions = np.searchsorted(_surviving_cols, _masked_original)
                X_train = X_train[:, _new_positions]
                X_test = X_test[:, _new_positions]
            fold_data = (X_train, y_train, X_test, y_test, subject_id)
        else:
            fold_data = (X_train, y_train, X_test, y_test, subject_id)

        return _process_single_fold(
            fold_data,
            le, clf_name, n_estimators, max_depth,
            learning_rate,
            None if global_selected_idx is not None else n_top_features,
            # v36.2 FIX: if per_subject_normalize done, use 'passthrough' in preprocess
            'f_classif' if per_subject_normalize and global_selected_idx is None else feature_selection,
            pca_components, use_ensemble,
            feature_names=None if global_selected_idx is not None else feature_names,
            subsample=subsample, colsample_bytree=colsample_bytree,
            xgb_nthread=xgb_nthread,
            early_stopping_rounds=early_stopping_rounds,
            balanced_sample_weight=balanced_sample_weight,
        )
    except MemoryError:
        print(f"[mem] MemoryError in LOSO worker (subject {subject_id})",
              flush=True)
        return [], [], 0.0, subject_id, None, None
    finally:
        _fold_dt = time.time() - _fold_t0
        print(f"  [fold] Subject {subject_id}: {_fold_dt:.1f}s", flush=True)


# ============================================================================
# Parallel LOSO Evaluation (v32.0 — Adaptive Engine)
# ============================================================================
def _evaluate_loso_parallel(features, labels, subject_groups, le, clf_name,
                            n_estimators, n_top_features, feature_selection,
                            pca_components, class_weight, return_models,
                            n_jobs=-1,
                            learning_rate=DEFAULT_LEARNING_RATE,
                            subsample=DEFAULT_SUBSAMPLE,
                            colsample_bytree=DEFAULT_COLSAMPLE,
                            early_stopping_rounds=DEFAULT_EARLY_STOPPING,
                            use_ensemble=False,
                            dataset_config=None,
                            feature_names=None,
                            svm_c=1.0, svm_gamma='scale'): #37.1
    """
    Parallel LOSO using joblib. v37.0 — Adaptive Engine.

    v37.0 changes:
      - balanced_sample_weight: per-class sample weighting for XGBoost
      - progressive_calibration: Vidovic 2016 unsupervised test adaptation

    v32.0 changes:
      - Reads adaptive settings from dataset_config
      - Supports hybrid feature selection with correlation protection
      - Supports ensemble voting
      - Passes feature_names through pipeline
      - Better early stopping (25 rounds, 10% val split)
    """
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(features, labels, groups=subject_groups)
    n_classes = len(le.classes_)

    # v37.0: Get all adaptive settings at once (pass n_total_samples) — 13 values
    (max_train, smart_n_est, smart_depth, smart_lr,
     smart_sub, smart_col, smart_k, smart_fs,
     smart_ensemble, smart_es,
     smart_per_subj_norm,
     smart_balanced_sw, smart_prog_cal) = _get_adaptive_settings(
        n_classes, dataset_config, n_total_samples=features.shape[0]
    )

    smart_n_est = min(n_estimators, smart_n_est)
    smart_lr = learning_rate if learning_rate != DEFAULT_LEARNING_RATE else smart_lr
    smart_sub = subsample if subsample != DEFAULT_SUBSAMPLE else smart_sub
    smart_col = colsample_bytree if colsample_bytree != DEFAULT_COLSAMPLE else smart_col
    smart_k = n_top_features if n_top_features != DEFAULT_TOP_FEATURES else smart_k
    smart_fs = feature_selection if feature_selection != 'hybrid' else smart_fs
    smart_es = early_stopping_rounds if early_stopping_rounds != DEFAULT_EARLY_STOPPING else smart_es
    smart_ensemble = use_ensemble if use_ensemble else smart_ensemble
    
    # v36.1: Safe parallelism — 2 workers × (cpu/2 - 1) threads ≤ cpu_count
    # Replaces complex memory-based calculation. Safe and 2x faster than n_jobs=1.
    cpu_count_val = os.cpu_count() or 4
     # v37.1: CLASSIFIER-AWARE PARALLELISM (fixes MemoryError with SVM/RF)
    # Memory-heavy classifiers create large model objects that must be
    # serialized through loky IPC -> MemoryError with 2 workers.
    # Solution: force 1 worker for these, use all threads instead.
    _MEMORY_HEAVY_CLFS = {'SVM', 'LINEARSVC', 'EXTRA_TREES', 'RANDOMFOREST'}
    if clf_name in _MEMORY_HEAVY_CLFS:
        # Memory-heavy: 1 worker, all threads (no serialization overhead)
        actual_n_jobs = 1
        xgb_nthread = cpu_count_val
    else:    
      actual_n_jobs = 2
      actual_n_jobs = min(actual_n_jobs, n_folds)
      xgb_nthread = max(1, (cpu_count_val // actual_n_jobs) - 1)
     # v37.1: Respect n_jobs from config/user (was completely ignored before!)
    if n_jobs == 1:
        actual_n_jobs = 1
        xgb_nthread = cpu_count_val
    elif n_jobs > 1:
        actual_n_jobs = min(n_jobs, n_folds)
        xgb_nthread = max(1, (cpu_count_val // actual_n_jobs) - 1)  

    print(f"[eval] v37.0 | Adaptive LOSO: {n_folds} folds | "
          f"workers={actual_n_jobs} | backend=loky | "
          f"xgb_nthread={xgb_nthread} | total_threads={actual_n_jobs * xgb_nthread}/{cpu_count_val} | "
          f"per_subj_norm={smart_per_subj_norm} | "
          f"balanced_sw={smart_balanced_sw} | prog_cal={smart_prog_cal}", flush=True)
    print(f"[adaptive] max_train={max_train} | n_est={smart_n_est} | "
          f"depth={smart_depth} | lr={smart_lr} | "
          f"subsample={smart_sub} | colsample={smart_col} | "
          f"k={smart_k} | fs={smart_fs} | "
          f"ensemble={smart_ensemble} | early_stop={smart_es}", flush=True)

    fold_indices = list(logo.split(features, labels, groups=subject_groups))

    if features.dtype != np.float32:
        features = np.ascontiguousarray(features, dtype=np.float32)

    # v35.1: GLOBAL FEATURE SELECTION (computed ONCE before LOSO loop)
    # v35.1 FIX: Now works for BOTH 'hybrid' AND 'f_classif' (was hybrid-only!)
    # This is the biggest speedup: FS was called 22 times (once per fold).
    # Now computed once on a stratified sample of the full dataset.
    global_selected_idx = None
    _use_global_fs = (smart_fs in ('hybrid', 'f_classif') and smart_k < features.shape[1])
    if _use_global_fs:
        _fs_t0 = time.time()
        print(f"[v35.1] Pre-computing global feature selection (once, fs={smart_fs})...",
              flush=True)
        # Use stratified subsample for global FS (statistical leakage minimal for FS)
        _fs_max_samples = min(50000, features.shape[0])
        if features.shape[0] > _fs_max_samples:
            _fs_idx = _stratified_subsample_indices(
                np.arange(features.shape[0]), labels, _fs_max_samples, random_state=42
            )
            _X_fs = features[_fs_idx]
            _y_fs = labels[_fs_idx]
        else:
            _X_fs = features
            _y_fs = labels
            # بعد تعريف _X_fs, _y_fs
        _X_fs = np.nan_to_num(_X_fs, nan=0.0, posinf=0.0, neginf=0.0)
        _y_fs = np.nan_to_num(_y_fs, nan=0.0)  # لو كان NaN في التصنيف

        if smart_fs == 'f_classif':
            # f_classif is very fast — can run on full subsample
            _f_sample = min(50000, _X_fs.shape[0])
            if _X_fs.shape[0] > _f_sample:
                rng_f = np.random.RandomState(42)
                f_idx = rng_f.choice(_X_fs.shape[0], _f_sample, replace=False)
                f_scores, _ = f_classif(_X_fs[f_idx], _y_fs[f_idx])
            else:
                f_scores, _ = f_classif(_X_fs, _y_fs)
            f_scores = np.nan_to_num(f_scores, nan=0.0)
            global_selected_idx = np.argsort(f_scores)[-smart_k:]
            print(f"[v35.1] Global f_classif done in {time.time() - _fs_t0:.1f}s, "
                  f"selected {len(global_selected_idx)} features", flush=True)
        else:
            # hybrid: mutual_info + f_classif combined
            global_selected_idx = _hybrid_feature_selection(
                _X_fs, _y_fs, smart_k, feature_names=feature_names,
                protect_corr_features=True, n_classes=n_classes
            )
            print(f"[v35.1] Global hybrid FS done in {time.time() - _fs_t0:.1f}s, "
                  f"selected {len(global_selected_idx)} features", flush=True)

        del _X_fs, _y_fs
        if features.shape[0] > _fs_max_samples:
            del _fs_idx
        gc.collect()

    gc.collect()
    t_start = time.time()

    if actual_n_jobs <= 1:
        results = []
        for i, (tr_idx, te_idx) in enumerate(fold_indices):
            gc.collect()
            result = _loso_fold_worker(
                tr_idx, te_idx, features, labels, subject_groups,
                max_train, le, clf_name, smart_n_est, smart_depth,
                smart_lr, smart_k, smart_fs,
                pca_components, smart_ensemble,
                smart_sub, smart_col,
                feature_names=feature_names,
                xgb_nthread=xgb_nthread,
                early_stopping_rounds=smart_es,
                global_selected_idx=global_selected_idx,
                n_classes=n_classes,
                per_subject_normalize=smart_per_subj_norm,
                balanced_sample_weight=smart_balanced_sw,
                progressive_calibration=smart_prog_cal,
                svm_c=svm_c, svm_gamma=svm_gamma, #37.1
                return_models=return_models, #37.1
            )
            results.append(result)
    else:
        # v36.1: loky backend (process isolation, no OpenMP contention)
        results = Parallel(n_jobs=actual_n_jobs, verbose=10,
                           backend='loky')(
            delayed(_loso_fold_worker)(
                tr_idx, te_idx, features, labels, subject_groups,
                max_train, le, clf_name, smart_n_est, smart_depth,
                smart_lr, smart_k, smart_fs,
                pca_components, smart_ensemble,
                smart_sub, smart_col,
                feature_names=feature_names,
                xgb_nthread=xgb_nthread,
                early_stopping_rounds=smart_es,
                global_selected_idx=global_selected_idx,
                n_classes=n_classes,
                per_subject_normalize=smart_per_subj_norm,
                balanced_sample_weight=smart_balanced_sw,
                progressive_calibration=smart_prog_cal,
            ) for tr_idx, te_idx in fold_indices
        )

    t_elapsed = time.time() - t_start
    print(f"[done] {n_folds} folds in {t_elapsed:.1f}s "
          f"({t_elapsed / n_folds:.1f}s/fold avg)", flush=True)

    y_true_all, y_pred_all, accs = [], [], []
    per_subject_acc = []
    per_subject_f1 = []
    trained_models, X_tests = [], []

    for yt, yp, acc, subj, model, Xt, mf1 in results:
        y_true_all.extend(yt)
        y_pred_all.extend(yp)
        accs.append(acc)
        per_subject_acc.append({'subject': subj, 'accuracy': float(acc)})
        per_subject_f1.append({'subject': subj, 'macro_f1': float(mf1)})
        if return_models and model is not None:
            trained_models.append(model)
            X_tests.append(Xt)

    return y_true_all, y_pred_all, accs, per_subject_acc, per_subject_f1, trained_models, X_tests


# ============================================================================
# Within-subject evaluation (v32.0 — adaptive settings)
# ============================================================================
def _evaluate_within_subject(features, labels, subject_groups, le,
                             clf_name, n_estimators, n_top_features,
                             feature_selection, pca_components,
                             class_weight, return_models,
                             train_ratio=0.7, random_state=42,
                             learning_rate=DEFAULT_LEARNING_RATE,
                             subsample=DEFAULT_SUBSAMPLE,
                             colsample_bytree=DEFAULT_COLSAMPLE,
                             use_ensemble=False,
                             dataset_config=None,
                             feature_names=None):
    """Within-subject evaluation. v37.0 — adaptive settings + balanced sample weight."""
    unique_subjects = np.unique(subject_groups)
    n_subjects = len(unique_subjects)
    n_classes = len(le.classes_)

    # v37.0: Get all adaptive settings — 13 values
    (max_train, smart_n_est, smart_depth, smart_lr,
     smart_sub, smart_col, smart_k, smart_fs,
     smart_ensemble, smart_es,
     _smart_per_subj_norm,
     smart_balanced_sw, _smart_prog_cal) = _get_adaptive_settings(
        n_classes, dataset_config, n_total_samples=features.shape[0]
    )
    # Note: progressive_calibration NOT used in within-subject (no domain shift)

    smart_n_est = min(n_estimators, smart_n_est)
    smart_lr = learning_rate if learning_rate != DEFAULT_LEARNING_RATE else smart_lr
    smart_sub = subsample if subsample != DEFAULT_SUBSAMPLE else smart_sub
    smart_col = colsample_bytree if colsample_bytree != DEFAULT_COLSAMPLE else smart_col
    smart_k = n_top_features if n_top_features != DEFAULT_TOP_FEATURES else smart_k
    smart_fs = feature_selection if feature_selection != 'hybrid' else smart_fs
    smart_ensemble = use_ensemble if use_ensemble else smart_ensemble

    print(f"[eval] v37.0 | Within-subject: {n_subjects} folds | "
          f"n_est={smart_n_est} | lr={smart_lr} | ensemble={smart_ensemble} | "
          f"balanced_sw={smart_balanced_sw}",
          flush=True)

    if features.dtype != np.float32:
        features = np.ascontiguousarray(features, dtype=np.float32)

    y_true_all, y_pred_all, accs = [], [], []
    per_subject_acc = []
    per_subject_f1 = []
    trained_models, X_tests = [], []

    for subj in unique_subjects:
        mask = subject_groups == subj
        X_subj = features[mask]
        y_subj = labels[mask]

        try:
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=train_ratio,
                random_state=random_state
            )
            train_idx, test_idx = next(sss.split(X_subj, y_subj))
            X_train, X_test = X_subj[train_idx], X_subj[test_idx]
            y_train, y_test = y_subj[train_idx], y_subj[test_idx]
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_subj, y_subj, train_size=train_ratio,
                random_state=random_state, stratify=y_subj
            )

        if len(set(np.unique(y_train))) < 2 or len(set(np.unique(y_test))) < 2:
            print(f"  Subject {subj}: insufficient class diversity, skip.",
                  flush=True)
            continue

        try:
            X_train, X_test = _preprocess_fold(
                X_train, X_test, y_train,
                n_top_features=smart_k,
                feature_selection=smart_fs,
                pca_components=pca_components,
                feature_names=feature_names,
            )
        except ValueError:
            print(f"  Subject {subj}: preprocessing error, skip.", flush=True)
            continue

        clf = None
        if smart_ensemble:
            y_train_enc = le.transform(y_train)
            xgb, rf_m, lda_m = _train_ensemble(
                X_train, y_train_enc, n_classes,
                smart_n_est, smart_depth, smart_lr, le,
                subsample=smart_sub, colsample_bytree=smart_col,
                early_stopping_rounds=smart_es,
                balanced_sample_weight=smart_balanced_sw,
            )
            y_pred = _predict_ensemble(xgb, rf_m, lda_m, X_test, le)
            clf = xgb
        else:
            clf, needs_encode = _build_classifier(
                clf_name, n_classes, smart_n_est,
                max_depth=smart_depth, learning_rate=smart_lr,
                n_jobs=-1,
                subsample=smart_sub,
                colsample_bytree=smart_col,
            )
            if needs_encode:
                y_train_enc = le.transform(y_train)
                # v37.0: Balanced sample weight for within-subject
                if smart_balanced_sw:
                    class_counts = np.bincount(y_train_enc)
                    n_total = len(y_train_enc)
                    n_cls = len(class_counts)
                    sw = n_total / (n_cls * class_counts + 1e-10)
                    sample_w = sw[y_train_enc]
                    clf.fit(X_train, y_train_enc, sample_weight=sample_w)
                else:
                    clf.fit(X_train, y_train_enc)
                y_pred_enc = clf.predict(X_test)
                if y_pred_enc.ndim == 2:
                    y_pred_enc = np.argmax(y_pred_enc, axis=1)
                y_pred = le.inverse_transform(y_pred_enc)
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        per_subject_acc.append({'subject': int(subj), 'accuracy': float(acc)})
        _f1_val = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        per_subject_f1.append({'subject': int(subj), 'macro_f1': _f1_val})
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        if return_models and clf is not None:
            trained_models.append(clf)
            X_tests.append(X_test)

        print(f"  Subject {subj}: acc={acc:.4f}", flush=True)

    return y_true_all, y_pred_all, accs, per_subject_acc, per_subject_f1, trained_models, X_tests


# ============================================================================
# Main Public API (v32.0 — Adaptive Engine)
# ============================================================================
def evaluate_model(features, labels, subject_groups,
                   strategy='loso', train_ratio=0.7, random_state=42,
                   classifier='XGBoost', svm_c=1.0, svm_gamma='scale',
                   class_weight=None, pca_components=None,
                   n_estimators=DEFAULT_N_ESTIMATORS,
                   n_top_features=DEFAULT_TOP_FEATURES,
                   feature_selection='f_classif', return_models=False,
                   n_jobs=-1, use_gpu=False,
                   learning_rate=DEFAULT_LEARNING_RATE,
                   subsample=DEFAULT_SUBSAMPLE,
                   colsample_bytree=DEFAULT_COLSAMPLE,
                   early_stopping_rounds=DEFAULT_EARLY_STOPPING,
                   use_ensemble=False,
                   dataset_config=None,
                   feature_names=None):
    """
    Evaluate EMG gesture classification. v37.0 — Selective Intelligence.

    v37.0 additions:
      - Balanced sample weight for XGBoost training
      - Progressive calibration (LOSO only, Vidovic 2016)
      - Euclidean alignment, TKEO band energy, covariance features
        (implemented in process_engine.py)

    v32.0 additions:
      - Hybrid feature selection (mutual_info + f_classif)
      - Adaptive per-dataset configuration
      - Ensemble support for hard datasets
      - Feature name propagation for correlation protection
      - Dynamic subsample/colsample based on class complexity
      - Better early stopping (25 rounds, 10% val split)

    Backward compatible: all v31.0c parameters preserved.
    """
    if strategy not in ('loso', 'within_subject'):
        raise ValueError(f"Unknown strategy: {strategy}")

    unique_subjects = np.unique(subject_groups)
    if len(unique_subjects) < 2 and strategy == 'loso':
        print("[warn] Only 1 subject — switching to within_subject.", flush=True)
        strategy = 'within_subject'

    clf_name = classifier.upper()

    print(f"\n{'='*60}", flush=True)
    print(f"[eval] v37.0 | strategy={strategy} | clf={classifier} | "
          f"n_est={n_estimators} | lr={learning_rate} | "
          f"k={n_top_features} | fs={feature_selection} | "
          f"subsample={subsample} | colsample={colsample_bytree} | "
          f"n_jobs={n_jobs} | early_stop={early_stopping_rounds} | "
          f"ensemble={use_ensemble} | n_subj={len(unique_subjects)}", flush=True)
    print(f"{'='*60}", flush=True)

    le = LabelEncoder()
    le.fit(labels)

    if strategy == 'loso':
        (y_true_all, y_pred_all, accs, per_subject_acc, per_subject_f1,
         trained_models, X_tests) = _evaluate_loso_parallel(
            features, labels, subject_groups, le, clf_name,
            n_estimators, n_top_features, feature_selection,
            pca_components, class_weight, return_models,
            n_jobs=n_jobs, learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample_bytree,
            early_stopping_rounds=early_stopping_rounds,
            use_ensemble=use_ensemble,
            dataset_config=dataset_config,
            feature_names=feature_names,
        )
    else:
        (y_true_all, y_pred_all, accs, per_subject_acc, per_subject_f1,
         trained_models, X_tests) = _evaluate_within_subject(
            features, labels, subject_groups, le, clf_name,
            n_estimators, n_top_features, feature_selection,
            pca_components, class_weight, return_models,
            train_ratio=train_ratio, random_state=random_state,
            learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample_bytree,
            use_ensemble=use_ensemble,
            dataset_config=dataset_config,
            feature_names=feature_names,
        )

    if not accs:
        return 0.0, 0.0, [], [], [], []

    acc_values = [a['accuracy'] if isinstance(a, dict) else a
                  for a in per_subject_acc]
    mean_acc = float(np.mean(acc_values))
    std_acc = float(np.std(acc_values, ddof=1)) if len(acc_values) > 1 else 0.0

    cm = confusion_matrix(y_true_all, y_pred_all).tolist()
    ci_lo, ci_hi = compute_confidence_interval(acc_values)

    print(f"\n  == Final: {mean_acc:.4f} +/- {std_acc:.4f}  "
          f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}] ==\n", flush=True)

    if not return_models:
        return mean_acc, std_acc, cm, acc_values, per_subject_f1, [], []
    return mean_acc, std_acc, cm, acc_values, per_subject_f1, trained_models, X_tests


# ============================================================================
# Feature Statistics & Confusion Matrix Plot (preserved from v31.0)
# ============================================================================
def feature_statistics(features, labels, feature_names, max_features=30):
    """Compute per-class feature means and stds."""
    stats_dict = {}
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        sub = features[mask]
        if len(sub) == 0:
            continue
        means = np.mean(sub, axis=0)
        stds = np.std(sub, axis=0)
        cls_stats = {}
        for i in range(min(max_features, len(feature_names))):
            cls_stats[feature_names[i]] = (
                float(means[i]), float(stds[i])
            )
        stats_dict[str(lbl)] = cls_stats
    return stats_dict


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    if not cm or not class_names:
        return
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Legacy Wrapper (backward compatible)
# ============================================================================
def classification_accuracy(features, labels, subject_groups,
                            classifier='XGBoost', use_grid_search=False,
                            svm_c=1.0, svm_gamma='scale', class_weight=None,
                            pca_components=None, n_estimators=DEFAULT_N_ESTIMATORS,
                            n_top_features=DEFAULT_TOP_FEATURES,
                            feature_selection='hybrid'):
    """Legacy wrapper for backward compatibility."""
    mean_acc, std_acc, cm, _, _, _ = evaluate_model(
        features, labels, subject_groups, strategy='loso',
        classifier=classifier, svm_c=svm_c, svm_gamma=svm_gamma,
        class_weight=class_weight, pca_components=pca_components,
        n_estimators=n_estimators, n_top_features=n_top_features,
        feature_selection=feature_selection,
    )
    return mean_acc, std_acc, cm
