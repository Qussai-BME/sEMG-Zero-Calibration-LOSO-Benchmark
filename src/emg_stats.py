#!/usr/bin/env python3
"""
emg_stats.py - Statistical analysis utilities for EMG data.
Provides descriptive stats, correlation, PCA, t-tests, and fatigue index.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from typing import Dict, List, Optional


def descriptive_stats(features_matrix: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Compute descriptive statistics (mean, std, min, max) for each channel.
    features_matrix: shape (n_windows, n_channels, n_features)
    feature_names: list of feature names (e.g., ['MAV','RMS','ZCR','WL','SSC'])
    Returns a DataFrame with columns: Channel, Feature, Mean, Std, Min, Max.
    """
    if features_matrix.ndim != 3:
        # If 2D, add feature dimension
        if features_matrix.ndim == 2:
            features_matrix = features_matrix[:, :, np.newaxis]
        else:
            raise ValueError(f"Expected 3D array, got {features_matrix.ndim}D")
    n_windows, n_channels, n_features = features_matrix.shape
    data = []
    for ch in range(n_channels):
        for f_idx, fname in enumerate(feature_names):
            vals = features_matrix[:, ch, f_idx]
            data.append({
                'Channel': f'Ch{ch}',
                'Feature': fname,
                'Mean': np.mean(vals),
                'Std': np.std(vals),
                'Min': np.min(vals),
                'Max': np.max(vals)
            })
    return pd.DataFrame(data)


def compute_correlation_matrix(features_matrix: np.ndarray) -> pd.DataFrame:
    """
    Compute correlation matrix between channels.
    features_matrix: shape (n_windows, n_channels) – e.g., RMS values per window per channel.
    """
    corr = np.corrcoef(features_matrix.T)
    return pd.DataFrame(corr, columns=[f"Ch{i}" for i in range(corr.shape[1])],
                        index=[f"Ch{i}" for i in range(corr.shape[0])])


def pca_analysis(features_matrix: np.ndarray, n_components: int = 2) -> Dict:
    """
    Perform PCA on the feature matrix.
    features_matrix: shape (n_windows, n_channels)
    Returns a dict with explained variance ratio and the transformed components.
    """
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(features_matrix)
    return {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'components': components.tolist()
    }


def t_test(group1: np.ndarray, group2: np.ndarray) -> Dict:
    """
    Independent two-sample t-test.
    """
    tstat, pvalue = stats.ttest_ind(group1, group2)
    return {
        't_statistic': float(tstat),
        'p_value': float(pvalue),
        'group1_mean': float(np.mean(group1)),
        'group1_std': float(np.std(group1)),
        'group2_mean': float(np.mean(group2)),
        'group2_std': float(np.std(group2))
    }


def fatigue_index(rms_over_time: np.ndarray, fs_features: float) -> float:
    """
    Estimate fatigue index as the negative slope of RMS over time.
    rms_over_time: array of RMS values (one per window)
    fs_features: number of windows per second (inverse of window step).
    Returns a fatigue index (positive = increasing activity, negative = fatigue).
    """
    x = np.arange(len(rms_over_time)) / fs_features  # time in seconds
    if len(x) < 2:
        return 0.0
    slope, _ = np.polyfit(x, rms_over_time, 1)
    return float(-slope)  # negative slope indicates fatigue