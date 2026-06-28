"""
shap_analysis.py — v3.0 (CLEAN REWRITE)
SHAP feature importance analysis with robust error handling.

IMPROVEMENTS:
  - Proper fallback chain: Explainer → TreeExplainer → skip
  - Correct handling of multi-class SHAP values
  - Feature name alignment validation
  - Memory-efficient: processes one fold at a time
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def compute_fold_shap(model, X, feature_names=None, n_background=200,
                      random_state=42):
    """
    Compute SHAP values for a single fold model.

    Returns (mean_abs_shap, raw_shap_values) or (None, None) on failure.
    """
    if not HAS_SHAP:
        warnings.warn("SHAP not installed. Skipping.")
        return None, None

    if len(X) == 0:
        return None, None

    try:
        # Subsample background data
        if len(X) > n_background:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(X), n_background, replace=False)
            X_bg = X[idx]
        else:
            X_bg = X

        # Validate feature dimension
        expected_features = len(feature_names) if feature_names else X.shape[1]
        if X.shape[1] != expected_features:
            warnings.warn(
                f"Feature mismatch: model expects {expected_features}, "
                f"got {X.shape[1]}. Skipping SHAP for this fold."
            )
            return None, None

        # Try modern Explainer first
        try:
            explainer = shap.Explainer(
                model, X_bg, feature_names=feature_names
            )
            shap_values = explainer(X)
            if hasattr(shap_values, 'values'):
                vals = shap_values.values
            else:
                vals = shap_values
        except Exception:
            # Fallback to TreeExplainer
            try:
                if hasattr(model, 'get_booster'):
                    booster = model.get_booster()
                    explainer = shap.TreeExplainer(
                        booster, X_bg, model_output='raw'
                    )
                else:
                    explainer = shap.TreeExplainer(
                        model, X_bg, model_output='raw'
                    )
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    vals = np.stack(shap_values, axis=2)
                else:
                    vals = shap_values
            except Exception as e:
                warnings.warn(f"TreeExplainer failed: {e}")
                return None, None

        # Compute mean absolute SHAP per feature
        if vals.ndim == 3:
            # Multi-class: average across classes and samples
            mean_abs_shap = np.abs(vals).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(vals).mean(axis=0)

        return mean_abs_shap, vals

    except Exception as e:
        warnings.warn(f"SHAP computation failed: {e}. Skipping this fold.")
        return None, None


def aggregate_loso_shap(models, X_tests, feature_names, n_background=200):
    """
    Aggregate SHAP values across all LOSO folds.

    Returns (DataFrame with features sorted by importance, list of raw values).
    """
    fold_shap_means = []
    all_shap_values = []
    n_features = len(feature_names)

    for i, (model, X_test) in enumerate(zip(models, X_tests)):
        mean_shap, shap_vals = compute_fold_shap(
            model, X_test, feature_names, n_background=n_background
        )
        if mean_shap is not None:
            # Ensure dimension alignment
            if len(mean_shap) == n_features:
                fold_shap_means.append(mean_shap)
                all_shap_values.append(shap_vals)
            else:
                warnings.warn(
                    f"Fold {i}: SHAP returned {len(mean_shap)} features, "
                    f"expected {n_features}. Skipping."
                )

    if not fold_shap_means:
        return None, None

    fold_shap_means = np.array(fold_shap_means)

    df = pd.DataFrame({
        'feature': feature_names[:len(fold_shap_means[0])],
        'mean_abs_shap': fold_shap_means.mean(axis=0),
        'std_abs_shap': fold_shap_means.std(axis=0)
    })
    df = df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    return df, all_shap_values


def compute_group_importance(df_shap, feature_groups):
    """
    Compute aggregated importance for feature groups.

    Parameters
    ----------
    df_shap : DataFrame with 'feature' and 'mean_abs_shap' columns.
    feature_groups : dict
        {group_name: pattern_or_list}
        If list: exact feature names.
        If str: regex pattern for feature names.

    Returns
    -------
    dict: {group_name: proportion_of_total_shap}
    """
    total_shap = df_shap['mean_abs_shap'].sum()
    if total_shap == 0:
        return {}

    group_importance = {}
    for group, pattern in feature_groups.items():
        if isinstance(pattern, list):
            mask = df_shap['feature'].isin(pattern)
        elif isinstance(pattern, str):
            mask = df_shap['feature'].str.contains(pattern, regex=True)
        else:
            continue
        group_sum = df_shap.loc[mask, 'mean_abs_shap'].sum()
        group_importance[group] = group_sum / total_shap

    return group_importance


def plot_shap_summary(shap_values, X, feature_names, save_path,
                     max_display=20):
    """Generate SHAP summary plot (beeswarm)."""
    plt.figure(figsize=(10, max_display * 0.35 + 2))
    try:
        shap.summary_plot(
            shap_values, X, feature_names=feature_names,
            max_display=max_display, show=False
        )
    except Exception as e:
        warnings.warn(f"SHAP summary plot failed: {e}")
        plt.close()
        return

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_group_importance_pie(group_importance, save_path):
    """Generate pie chart of feature group importance."""
    labels = list(group_importance.keys())
    sizes = [group_importance[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=90, pctdistance=0.85
    )
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)

    ax.set_title('SHAP Feature Group Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_missing_channel_impact(df_shap, channel_mask,
                                   channel_feature_prefix='ch'):
    """
    Analyze SHAP contribution lost due to missing/inactive channels.

    Parameters
    ----------
    df_shap : DataFrame with 'feature' and 'mean_abs_shap'
    channel_mask : list of int (1=active, 0=inactive)
    channel_feature_prefix : str

    Returns
    -------
    (total_missing_shap, proportion_of_total)
    """
    total_shap = df_shap['mean_abs_shap'].sum()
    if total_shap == 0:
        return 0.0, 0.0

    missing_shap = 0.0
    for ch_idx, active in enumerate(channel_mask):
        if active == 0:
            # Direct channel features
            pattern = f'{channel_feature_prefix}{ch_idx}_'
            mask = df_shap['feature'].str.startswith(pattern)
            missing_shap += df_shap.loc[mask, 'mean_abs_shap'].sum()

            # Correlation features involving this channel
            pattern_i = f'corr_{ch_idx}_'
            pattern_j = f'corr_\\d+_{ch_idx}'
            mask_corr = (
                df_shap['feature'].str.contains(pattern_i)
                | df_shap['feature'].str.contains(pattern_j, regex=True)
            )
            missing_shap += df_shap.loc[mask_corr, 'mean_abs_shap'].sum()

    return missing_shap, missing_shap / total_shap


def run_shap_analysis(models, X_tests, feature_names, output_dir,
                      dataset_name, feature_groups=None, channel_mask=None):
    """
    Run complete SHAP analysis pipeline.

    Generates:
      - CSV: top 50 features by SHAP importance
      - CSV: feature group importance
      - PNG: SHAP summary beeswarm plot
      - PNG: feature group pie chart
      - TXT: missing channel impact analysis
    """
    if not HAS_SHAP:
        print("SHAP not installed. Skipping SHAP analysis.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"[SHAP] Analyzing {len(models)} folds for {dataset_name}...", flush=True)

    df_shap, all_shap_vals = aggregate_loso_shap(
        models, X_tests, feature_names
    )

    if df_shap is None:
        print(f"[SHAP] Analysis for {dataset_name} failed (computation error).")
        return

    # Save top 50 features
    csv_path = os.path.join(
        output_dir, f'shap_top50_{dataset_name}.csv'
    )
    df_shap.head(50).to_csv(csv_path, index=False)
    print(f"[SHAP] Top 50 features → {csv_path}")

    # Feature group analysis
    if feature_groups is not None:
        group_imp = compute_group_importance(df_shap, feature_groups)
        if group_imp:
            group_csv = os.path.join(
                output_dir, f'shap_groups_{dataset_name}.csv'
            )
            pd.DataFrame(
                list(group_imp.items()),
                columns=['group', 'importance']
            ).to_csv(group_csv, index=False)

            pie_path = os.path.join(
                output_dir, f'shap_groups_{dataset_name}.png'
            )
            plot_group_importance_pie(group_imp, pie_path)
            print(f"[SHAP] Group analysis → {group_csv}, {pie_path}")

    # Summary plot
    if all_shap_vals:
        shap_vals_0 = all_shap_vals[0]
        X_0 = X_tests[0]
        if shap_vals_0.ndim == 3:
            shap_vals_plot = shap_vals_0[:, :, 0]
        else:
            shap_vals_plot = shap_vals_0

        summary_path = os.path.join(
            output_dir, f'shap_summary_{dataset_name}.png'
        )
        plot_shap_summary(
            shap_vals_plot, X_0, feature_names, summary_path
        )
        print(f"[SHAP] Summary plot → {summary_path}")

    # Missing channel analysis
    if channel_mask is not None:
        missing_sum, missing_pct = analyze_missing_channel_impact(
            df_shap, channel_mask
        )
        impact_path = os.path.join(
            output_dir, f'missing_channel_shap_{dataset_name}.txt'
        )
        with open(impact_path, 'w') as f:
            f.write(f"Total SHAP sum: {df_shap['mean_abs_shap'].sum():.4f}\n")
            f.write(f"Missing channel SHAP: {missing_sum:.4f}\n")
            f.write(f"Percentage of total: {missing_pct:.2%}\n")
            f.write(f"Channel mask: {channel_mask}\n")
            f.write(f"Active channels: {sum(channel_mask)}/{len(channel_mask)}\n")
        print(f"[SHAP] Channel impact → {impact_path}")

    print(f"[SHAP] Analysis for {dataset_name} completed.")
