"""
statistical_reporter.py — v2.0 (CLEAN REWRITE)
D1-quality statistical reporting for EMG Analysis Engine.

Provides:
  - Wilcoxon signed-rank with effect size + power analysis
  - Mann-Whitney U (intact vs amputee)
  - Bootstrap CI + t-CI with disagreement detection
  - Cohen's d (paired and independent)
  - Holm-Sidak correction for multiple comparisons
  - LaTeX table generation (IEEE format)
  - Publication-ready console summaries

No changes to core logic — this module was already clean.
Minor improvements: better error handling, cleaner output formatting.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, bootstrap

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    from statsmodels.stats.power import TTestPower
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# =====================================================================
# Core statistical primitives
# =====================================================================

def wilcoxon_signed_rank(a, b, alpha=0.05):
    """
    Two-sided Wilcoxon signed-rank test on paired per-subject accuracies.

    Returns dict with: W, p, significant, r_effect, ci_r, interpretation
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError(f"Arrays must be same length: {len(a)} vs {len(b)}")
    if len(a) < 4:
        warnings.warn("n<4: Wilcoxon test is unreliable at this sample size.")

    diff = a - b
    if np.all(diff == 0):
        return {
            "W": 0, "p": 1.0, "significant": False, "r_effect": 0.0,
            "interpretation": "No difference (all ties)"
        }

    W, p = wilcoxon(diff, zero_method='wilcox', alternative='two-sided')
    n = len(diff)
    r = 1.0 - (2.0 * W) / (n * (n + 1))

    # Conservative SE approximation for CI on r
    ci_lo = r - 1.96 * 0.6 / np.sqrt(n)
    ci_hi = r + 1.96 * 0.6 / np.sqrt(n)
    ci_lo = max(ci_lo, -1.0)
    ci_hi = min(ci_hi, 1.0)

    r_abs = abs(r)
    magnitude = "negligible"
    if r_abs >= 0.1:
        magnitude = "small"
    if r_abs >= 0.3:
        magnitude = "medium"
    if r_abs >= 0.5:
        magnitude = "large"

    return {
        "W": float(W),
        "p": float(p),
        "significant": bool(p < alpha),
        "r_effect": float(r),
        "ci_r": [float(ci_lo), float(ci_hi)],
        "n": n,
        "magnitude": magnitude,
        "mean_diff": float(np.mean(a) - np.mean(b)),
        "interpretation": (
            f"W={W:.0f}, p={p:.4f} ({'*' if p < alpha else 'ns'}), "
            f"r={r:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}] ({magnitude})"
        )
    }


def mann_whitney_u(group_a, group_b, label_a="A", label_b="B", alpha=0.05):
    """
    Two-sided Mann-Whitney U test for independent groups.

    Returns dict with: U, p, significant, r_effect, cohens_d, interpretation
    """
    a, b = np.asarray(group_a, float), np.asarray(group_b, float)
    if len(a) < 2 or len(b) < 2:
        warnings.warn(
            f"Mann-Whitney requires n>=2 per group. Got {len(a)}, {len(b)}."
        )

    U, p = mannwhitneyu(a, b, alternative='two-sided')
    n1, n2 = len(a), len(b)
    r = 1.0 - (2.0 * U) / (n1 * n2)

    pooled_var = (
        ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1))
        / (n1 + n2 - 2)
    ) if (n1 + n2 - 2) > 0 else 1.0
    d = (np.mean(a) - np.mean(b)) / (np.sqrt(pooled_var) + 1e-12)

    d_abs = abs(d)
    magnitude = "negligible"
    if d_abs >= 0.2:
        magnitude = "small"
    if d_abs >= 0.5:
        magnitude = "medium"
    if d_abs >= 0.8:
        magnitude = "large"

    return {
        "U": float(U),
        "p": float(p),
        "significant": bool(p < alpha),
        "r_effect": float(r),
        "cohens_d": float(d),
        "n_a": n1, "n_b": n2,
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "magnitude": magnitude,
        "label_a": label_a, "label_b": label_b,
        "interpretation": (
            f"U={U:.0f}, p={p:.4f} ({'*' if p < alpha else 'ns'}), "
            f"r={r:+.3f}, d={d:+.3f} ({magnitude}) | "
            f"{label_a}: {np.mean(a):.3f}+/-{np.std(a, ddof=1):.3f} "
            f"vs {label_b}: {np.mean(b):.3f}+/-{np.std(b, ddof=1):.3f}"
        )
    }


def bootstrap_ci(data, statistic=np.mean, n_resamples=10000,
                 confidence_level=0.95, random_state=42):
    """
    Bootstrap confidence interval. Cross-check against t-CI.
    Large disagreement -> non-normal distribution -> prefer bootstrap in paper.
    """
    data = np.asarray(data, float)
    result = bootstrap(
        (data,), statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state,
        method='percentile'
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def t_ci(data, confidence=0.95):
    """Student-t confidence interval."""
    data = np.asarray(data, float)
    n = len(data)
    if n < 2:
        m = float(np.mean(data))
        return m, m
    m = float(np.mean(data))
    s = float(np.std(data, ddof=1))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * s / np.sqrt(n)
    return m - margin, m + margin


def cohens_d_paired(a, b):
    """Cohen's d for paired samples."""
    diff = np.asarray(a) - np.asarray(b)
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def power_analysis_wilcoxon(effect_r, n_observed, alpha=0.05):
    """
    Approximate power for observed Wilcoxon effect size.

    Returns (observed_power, n_required_for_80pct_power).
    """
    if not HAS_STATSMODELS:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        d_approx = 2 * abs(effect_r) / np.sqrt(1 - effect_r ** 2 + 1e-12)
        observed_power = stats.norm.cdf(
            abs(d_approx) * np.sqrt(n_observed / 2) - z_alpha
        )
        n_for_80 = int(np.ceil(
            ((z_alpha + stats.norm.ppf(0.80)) / (abs(d_approx) + 1e-12)) ** 2 * 2
        ))
    else:
        d_approx = 2 * abs(effect_r) / np.sqrt(1 - effect_r ** 2 + 1e-12)
        tp = TTestPower()
        observed_power = tp.solve_power(
            effect_size=d_approx, nobs=n_observed,
            alpha=alpha, alternative='two-sided'
        )
        n_for_80 = int(np.ceil(tp.solve_power(
            effect_size=d_approx, power=0.80,
            alpha=alpha, alternative='two-sided'
        )))

    return float(observed_power), int(n_for_80)


def holm_sidak_correction(p_values, alpha=0.05):
    """
    Holm-Sidak correction for multiple comparisons.
    More powerful than Bonferroni, controls FWER.

    Returns dict: adjusted_p, rejected, method, alpha
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    rejected = [False] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        alpha_adj = 1 - (1 - alpha) ** (1 / (n - rank))
        rejected[orig_idx] = bool(p <= alpha_adj)
        adjusted[orig_idx] = float(min(1.0, p * (n - rank)))

    return {
        "adjusted_p": adjusted,
        "rejected": rejected,
        "method": "Holm-Sidak",
        "alpha": alpha
    }


# =====================================================================
# High-level reporter class
# =====================================================================

class StatisticalReporter:
    """
    Orchestrates all statistical reporting for EMG Analysis Engine.

    All methods:
      - Print IEEE-style summary to console
      - Save JSON results to output_dir
      - Optionally save LaTeX tables and figures
    """

    def __init__(self, output_dir: str = './validation_reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save_json(self, name, data):
        path = os.path.join(self.output_dir, f"{name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def _save_latex(self, name, latex_str):
        path = os.path.join(self.output_dir, f"{name}.tex")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        return path

    def _header(self, title):
        line = "=" * 72
        print(f"\n{line}")
        print(f"  {title}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(line)

    # ── ablation study ──────────────────────────────────────────────────

    def ablation_report(self, baseline_accs, config_results,
                        dataset_name="UCI_Gesture", alpha=0.05):
        """Full ablation statistical analysis with multiple-comparison correction."""
        self._header(f"ABLATION STUDY -- {dataset_name}")

        baseline = np.asarray(baseline_accs, float)
        n = len(baseline)
        print(
            f"\n  Baseline: {np.mean(baseline) * 100:.2f}% "
            f"+/- {np.std(baseline, ddof=1) * 100:.2f}%  (n={n})\n"
        )

        all_p_values = []
        config_names = list(config_results.keys())
        raw_results = {}

        for name, accs in config_results.items():
            accs = np.asarray(accs, float)
            res = wilcoxon_signed_rank(baseline, accs, alpha=alpha)
            raw_results[name] = res
            all_p_values.append(res['p'])
            print(
                f"  {name:<35} D={res['mean_diff'] * 100:+.2f}%  "
                f"{res['interpretation']}"
            )

        # Multiple-comparison correction
        correction = holm_sidak_correction(all_p_values, alpha=alpha)
        print(
            f"\n  Multiple comparison correction "
            f"(Holm-Sidak, k={len(all_p_values)}):"
        )
        for i, name in enumerate(config_names):
            raw_p = all_p_values[i]
            adj_p = correction['adjusted_p'][i]
            star = "REJECT H0" if correction['rejected'][i] else "accept H0"
            print(
                f"    {name:<35} raw p={raw_p:.4f}  "
                f"adj p={adj_p:.4f}  {star}"
            )

        # Power analysis
        print(f"\n  Power analysis (alpha={alpha}, target 80%):")
        power_results = {}
        for name in config_names:
            r = raw_results[name]['r_effect']
            obs_pwr, n80 = power_analysis_wilcoxon(r, n, alpha)
            power_results[name] = {
                "observed_power": obs_pwr, "n_for_80pct": n80
            }
            print(
                f"    {name:<35} power={obs_pwr:.0%}  n_needed={n80}"
            )

        # LaTeX table
        latex = self._ablation_latex_table(
            baseline, config_results, raw_results,
            correction, dataset_name
        )
        tex_path = self._save_latex(f"ablation_{dataset_name}", latex)
        print(f"\n  LaTeX table -> {tex_path}")

        full_results = {
            "dataset": dataset_name,
            "baseline_mean": float(np.mean(baseline)),
            "baseline_std": float(np.std(baseline, ddof=1)),
            "n_subjects": n,
            "tests": {k: v for k, v in raw_results.items()},
            "holm_sidak": correction,
            "power_analysis": power_results,
            "latex_path": tex_path
        }
        json_path = self._save_json(
            f"ablation_stats_{dataset_name}", full_results
        )
        print(f"  JSON results  -> {json_path}\n")
        return full_results

    def _ablation_latex_table(self, baseline, config_results, raw_results,
                              correction, dataset_name):
        config_names = list(config_results.keys())
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Feature Ablation Study --- "
            + dataset_name.replace('_', r'\_')
            + r" (LOSO, n=" + str(len(baseline)) + r"). "
            r"Wilcoxon signed-rank test, two-sided. "
            r"$p$ values Holm-\v{S}\'{i}d\'{a}k corrected for "
            + str(len(config_names)) + r" comparisons.}",
            r"\label{tab:ablation_" + dataset_name.lower() + r"}",
            r"\begin{tabular}{lrrrrr}",
            r"\hline",
            r"\textbf{Configuration} & \textbf{Acc (\%)} & $\Delta$ & "
            r"$W$ & $p$ (adj.) & $r$ \\",
            r"\hline",
        ]
        base_mean = np.mean(baseline) * 100
        base_std = np.std(baseline, ddof=1) * 100
        lines.append(
            rf"\textbf{{Full baseline}} & {base_mean:.2f} $\pm$ {base_std:.2f} & "
            rf"--- & --- & --- & --- \\"
        )
        for i, name in enumerate(config_names):
            accs = np.asarray(config_results[name], float)
            m = np.mean(accs) * 100
            s = np.std(accs, ddof=1) * 100
            res = raw_results[name]
            adj_p = correction['adjusted_p'][i]
            rej = correction['rejected'][i]
            star = r"$\star$" if rej else ""
            delta = res['mean_diff'] * 100
            r_val = res['r_effect']
            W_val = res['W']
            p_str = f"{adj_p:.4f}{star}"
            name_tex = name.replace('_', r'\_')
            if 'Recommended' in name:
                name_tex = r"\textbf{" + name_tex + r"}"
            lines.append(
                rf"{name_tex} & {m:.2f} $\pm$ {s:.2f} & {delta:+.2f} & "
                rf"{W_val:.0f} & {p_str} & {r_val:+.3f} \\"
            )
        lines += [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    # ── group comparison (intact vs amputee) ────────────────────────────

    def group_comparison(self, intact_accs, amputee_accs,
                         dataset_name="NinaPro_DB7"):
        """Mann-Whitney U test: intact vs amputee per-subject LOSO accuracies."""
        self._header(f"GROUP COMPARISON -- {dataset_name}")

        res = mann_whitney_u(
            intact_accs, amputee_accs,
            label_a="Intact", label_b="Amputee"
        )
        print(f"\n  Intact  (n={res['n_a']}): {res['mean_a'] * 100:.2f}%")
        print(f"  Amputee (n={res['n_b']}): {res['mean_b'] * 100:.2f}%")
        print(
            f"  Gap     : {(res['mean_a'] - res['mean_b']) * 100:+.2f}%"
        )
        print(f"\n  {res['interpretation']}")

        if res['n_b'] < 4:
            print(
                f"\n  WARNING: n_amputee={res['n_b']} -- "
                f"insufficient for statistical inference."
            )

        full_results = {**res, "dataset": dataset_name}
        json_path = self._save_json(
            f"group_comparison_{dataset_name}", full_results
        )
        print(f"\n  JSON -> {json_path}\n")
        return full_results

    # ── classifier comparison (XGBoost vs CNN) ─────────────────────────

    def classifier_comparison(self, xgb_accs, cnn_accs,
                              xgb_times_ms=None, cnn_times_ms=None,
                              dataset_name="UCI_Gesture"):
        """XGBoost vs CNN: Wilcoxon + Cohen's d + bootstrap CI + inference timing."""
        self._header(
            f"CLASSIFIER COMPARISON XGBoost vs CNN -- {dataset_name}"
        )

        xgb = np.asarray(xgb_accs, float)
        cnn = np.asarray(cnn_accs, float)

        assert len(xgb) == len(cnn), "Must use same LOSO folds for paired test."

        wil = wilcoxon_signed_rank(xgb, cnn)
        d = cohens_d_paired(xgb, cnn)
        xgb_boot_lo, xgb_boot_hi = bootstrap_ci(xgb)
        cnn_boot_lo, cnn_boot_hi = bootstrap_ci(cnn)
        xgb_t_lo, xgb_t_hi = t_ci(xgb)
        cnn_t_lo, cnn_t_hi = t_ci(cnn)

        print(
            f"\n  XGBoost: {np.mean(xgb) * 100:.2f}% "
            f"+/- {np.std(xgb, ddof=1) * 100:.2f}%"
        )
        print(
            f"           t-CI: [{xgb_t_lo * 100:.2f}%, {xgb_t_hi * 100:.2f}%]"
        )
        print(
            f"           Bootstrap CI: [{xgb_boot_lo * 100:.2f}%, "
            f"{xgb_boot_hi * 100:.2f}%]"
        )
        print(
            f"\n  CNN:     {np.mean(cnn) * 100:.2f}% "
            f"+/- {np.std(cnn, ddof=1) * 100:.2f}%"
        )
        print(
            f"           t-CI: [{cnn_t_lo * 100:.2f}%, {cnn_t_hi * 100:.2f}%]"
        )
        print(
            f"           Bootstrap CI: [{cnn_boot_lo * 100:.2f}%, "
            f"{cnn_boot_hi * 100:.2f}%]"
        )

        # CI consistency check
        xgb_tdiff = xgb_t_hi - xgb_t_lo
        xgb_bdiff = xgb_boot_hi - xgb_boot_lo
        if xgb_tdiff > 0:
            ratio = abs(xgb_tdiff - xgb_bdiff) / xgb_tdiff
            if ratio > 0.15:
                print(
                    f"\n  WARNING: CI disagreement detected "
                    f"(t vs bootstrap width ratio {xgb_tdiff / xgb_bdiff:.2f}x). "
                    f"Use bootstrap CI (non-normal distribution)."
                )
            else:
                print(
                    "\n  OK: t-CI and bootstrap CI agree -- "
                    "distribution is approximately normal."
                )

        print(f"\n  Paired comparison: {wil['interpretation']}")
        print(f"  Cohen's d (paired): {d:+.3f}")

        timing_result = {}
        if xgb_times_ms and cnn_times_ms:
            xgb_t_mean = float(np.mean(xgb_times_ms))
            cnn_t_mean = float(np.mean(cnn_times_ms))
            budget_ms = 150.0
            print(f"\n  Inference timing:")
            print(
                f"    XGBoost: {xgb_t_mean:.3f} ms  "
                f"(headroom: {budget_ms - xgb_t_mean:.1f} ms)"
            )
            print(
                f"    CNN:     {cnn_t_mean:.3f} ms  "
                f"(headroom: {budget_ms - cnn_t_mean:.1f} ms)"
            )
            if xgb_t_mean > 0:
                print(
                    f"    Speed ratio: CNN is "
                    f"{cnn_t_mean / xgb_t_mean:.1f}x slower"
                )
            timing_result = {
                "xgb_ms": xgb_t_mean,
                "cnn_ms": cnn_t_mean,
                "speed_ratio": (
                    cnn_t_mean / xgb_t_mean if xgb_t_mean > 0 else 0
                ),
                "prosthetic_budget_ms": budget_ms
            }

        latex = self._classifier_comparison_latex(
            xgb, cnn, wil, d, xgb_t_lo, xgb_t_hi, cnn_t_lo, cnn_t_hi,
            dataset_name
        )
        tex_path = self._save_latex(
            f"classifier_comparison_{dataset_name}", latex
        )

        full_results = {
            "dataset": dataset_name,
            "xgb_mean": float(np.mean(xgb)),
            "xgb_std": float(np.std(xgb, ddof=1)),
            "xgb_ci_t": [xgb_t_lo, xgb_t_hi],
            "xgb_ci_bootstrap": [xgb_boot_lo, xgb_boot_hi],
            "cnn_mean": float(np.mean(cnn)),
            "cnn_std": float(np.std(cnn, ddof=1)),
            "cnn_ci_t": [cnn_t_lo, cnn_t_hi],
            "cnn_ci_bootstrap": [cnn_boot_lo, cnn_boot_hi],
            "wilcoxon": wil,
            "cohens_d": float(d),
            "timing": timing_result,
        }
        json_path = self._save_json(
            f"classifier_comparison_{dataset_name}", full_results
        )
        print(f"\n  LaTeX -> {tex_path}")
        print(f"  JSON -> {json_path}\n")
        return full_results

    def _classifier_comparison_latex(self, xgb, cnn, wil, d,
                                     xgb_lo, xgb_hi, cnn_lo, cnn_hi, name):
        n = len(xgb)
        d_abs = abs(d)
        d_interp = "negligible"
        if d_abs >= 0.2:
            d_interp = "small"
        if d_abs >= 0.5:
            d_interp = "medium"
        if d_abs >= 0.8:
            d_interp = "large"

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{XGBoost vs 1D-CNN baseline on "
            + name.replace('_', r'\_')
            + r" (n=" + str(n)
            + r" LOSO folds, paired Wilcoxon).}",
            r"\label{tab:clf_comparison_" + name.lower() + r"}",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"\textbf{Classifier} & \textbf{LOSO Acc.} & \textbf{95\% CI} & "
            r"\textbf{Wilcoxon} & \textbf{Cohen's $d$}\\",
            r"\hline",
            rf"\textbf{{XGBoost}} & {np.mean(xgb) * 100:.2f}\% $\pm$ "
            rf"{np.std(xgb, ddof=1) * 100:.2f}\% & "
            rf"[{xgb_lo * 100:.2f}\%, {xgb_hi * 100:.2f}\%] & "
            rf"W={wil['W']:.0f} & ---\\",
            rf"1D-CNN baseline & {np.mean(cnn) * 100:.2f}\% $\pm$ "
            rf"{np.std(cnn, ddof=1) * 100:.2f}\% & "
            rf"[{cnn_lo * 100:.2f}\%, {cnn_hi * 100:.2f}\%] & "
            rf"$p$={wil['p']:.4f} & {d:+.3f} ({d_interp})\\",
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    # ── cross-dataset summary ──────────────────────────────────────────

    def cross_dataset_summary(self, results):
        """Generate cross-dataset performance summary table."""
        self._header("CROSS-DATASET PERFORMANCE SUMMARY")

        rows = []
        for ds_name, info in results.items():
            accs = np.asarray(info['accs'], float)
            n = len(accs)
            m = np.mean(accs)
            s = np.std(accs, ddof=1) if n > 1 else 0.0
            ci_lo, ci_hi = t_ci(accs) if n > 1 else (m, m)
            chance = 1.0 / info.get('n_classes', 1)
            rows.append({
                'Dataset': ds_name.replace('_', ' '),
                'n': n,
                'Classes': info.get('n_classes', '?'),
                'Channels': info.get('n_ch', '?'),
                'Fs (Hz)': info.get('fs', '?'),
                'LOSO Acc': f"{m * 100:.2f}% +/- {s * 100:.2f}%",
                '95% CI (t)': f"[{ci_lo * 100:.2f}%, {ci_hi * 100:.2f}%]",
                'Chance': f"{chance * 100:.1f}%",
            })
            print(
                f"\n  {ds_name}: {m * 100:.2f}% +/- {s * 100:.2f}% "
                f"(n={n}, CI:[{ci_lo * 100:.2f}%, {ci_hi * 100:.2f}%])"
            )

        latex = self._cross_dataset_latex(rows)
        tex_path = self._save_latex("cross_dataset_summary", latex)
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, "cross_dataset_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n  LaTeX -> {tex_path}")
        print(f"  CSV   -> {csv_path}\n")
        return tex_path

    def _cross_dataset_latex(self, rows):
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Cross-dataset performance summary under strict "
            r"LOSO evaluation.}",
            r"\label{tab:cross_dataset}",
            r"\begin{tabular}{lcccrll}",
            r"\hline",
            r"\textbf{Dataset} & \textbf{$n$} & \textbf{Classes} & "
            r"\textbf{Channels} & \textbf{$F_s$} & "
            r"\textbf{LOSO Acc.} & \textbf{Chance}\\",
            r"\hline",
        ]
        for r in rows:
            ds = r['Dataset'].replace(' ', r'\ ')
            lines.append(
                rf"{ds} & {r['n']} & {r['Classes']} & {r['Channels']} & "
                rf"{r['Fs (Hz)']} & {r['LOSO Acc']} & {r['Chance']}\\"
            )
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    # ── manuscript-ready summary ───────────────────────────────────────

    def generate_paper_stats_summary(self, all_results):
        """Generate IEEE/APA formatted statistics summary."""
        self._header("MANUSCRIPT-READY STATISTICS SUMMARY")
        lines = [
            "EMG Analysis Engine -- Statistical Summary",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 72, ""
        ]

        for ds_name, info in all_results.items():
            accs = np.asarray(
                info.get('per_subject_accuracy', []), float
            )
            if len(accs) == 0:
                continue
            n = len(accs)
            m = np.mean(accs)
            s = np.std(accs, ddof=1) if n > 1 else 0.0
            ci_lo, ci_hi = t_ci(accs) if n > 1 else (m, m)

            lines += [
                f"Dataset: {ds_name}",
                f"  n = {n} subjects",
                f"  LOSO accuracy: {m * 100:.2f}% +/- {s * 100:.2f}%",
                f"  95% CI (t-dist, df={n - 1}): "
                f"[{ci_lo * 100:.2f}%, {ci_hi * 100:.2f}%]",
                f"  Median: {np.median(accs) * 100:.2f}%",
                f"  Range: [{np.min(accs) * 100:.2f}%, "
                f"{np.max(accs) * 100:.2f}%]",
                f"  IEEE: The framework achieves {m * 100:.2f}% "
                f"+/- {s * 100:.2f}% LOSO accuracy "
                f"(95% CI: [{ci_lo * 100:.2f}%, {ci_hi * 100:.2f}%]) "
                f"on {ds_name} (n={n} subjects).",
                ""
            ]

        text = "\n".join(lines)
        path = os.path.join(
            self.output_dir, "paper_stats_summary.txt"
        )
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(text)
        print(f"  Saved -> {path}\n")
        return path


# =====================================================================
# Convenience function
# =====================================================================

def run_full_statistical_pipeline(results_json_paths,
                                   output_dir='./validation_reports',
                                   ablation_accs=None):
    """One-shot: load validate_engine JSON outputs and produce all stats."""
    reporter = StatisticalReporter(output_dir=output_dir)
    all_results = {}

    for ds_name, json_path in results_json_paths.items():
        with open(json_path, 'r') as f:
            data = json.load(f)
        all_results[ds_name] = data

    reporter.generate_paper_stats_summary(all_results)

    if ablation_accs and 'baseline' in ablation_accs:
        baseline = ablation_accs.pop('baseline')
        reporter.ablation_report(baseline, ablation_accs)

    return reporter


if __name__ == '__main__':
    np.random.seed(42)
    print("Running statistical_reporter.py smoke test...")

    reporter = StatisticalReporter(output_dir='/tmp/stats_test')

    xgb_accs = np.clip(
        np.random.normal(0.8716, 0.1445, 36), 0.4, 1.0
    ).tolist()
    minus_icc = np.clip(
        np.array(xgb_accs) - 0.0073 + np.random.normal(0, 0.01, 36),
        0.4, 1.0
    ).tolist()
    rec_cfg = np.clip(
        np.array(xgb_accs) + 0.0024 + np.random.normal(0, 0.01, 36),
        0.4, 1.0
    ).tolist()
    cnn_accs = np.clip(
        np.random.normal(0.78, 0.12, 36), 0.4, 1.0
    ).tolist()

    reporter.ablation_report(
        xgb_accs,
        {'minus_ICC': minus_icc, 'RecommendedConfig': rec_cfg},
        dataset_name='UCI_Gesture_TEST'
    )
    reporter.classifier_comparison(
        rec_cfg, cnn_accs, dataset_name='UCI_Gesture_TEST'
    )
    reporter.group_comparison(
        xgb_accs[:3], xgb_accs[3:5], dataset_name='NinaPro_DB7_TEST'
    )
    print("\nSmoke test passed -- all outputs in /tmp/stats_test/")
