"""
report_generator.py — v6.0 (CLEAN REWRITE)
Enhanced report generation with proper error handling and formatting.

FIXES:
  - Proper handling of None classification results
  - Cleaner Markdown and HTML output
  - No external dependencies required for basic output
"""

import os
import json
import sys
import numpy as np
import traceback
from datetime import datetime

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

try:
    from metrics import plot_confusion_matrix
except ImportError:
    try:
        from validation.metrics import plot_confusion_matrix
    except ImportError:
        plot_confusion_matrix = None


def generate_report(dataset_name, config, results, output_dir):
    """
    Generate validation report in JSON, Markdown, and HTML formats.

    Parameters
    ----------
    dataset_name : str
    config : dict - processing configuration
    results : dict - validation results from process_dataset
    output_dir : str - directory to save reports
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Save JSON (primary, machine-readable) ──────────────────────────
    json_path = os.path.join(
        output_dir, f"{dataset_name}_results.json"
    )
    print(f"Saving JSON -> {json_path}", flush=True)

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(_convert(results), f, indent=2, ensure_ascii=False)
        print("JSON saved.", flush=True)
    except Exception as e:
        print(f"ERROR saving JSON: {e}", flush=True)
        traceback.print_exc()
        return

    # ── Generate Markdown report ────────────────────────────────────────
    try:
        md = _build_markdown_report(dataset_name, config, results)
        md_path = os.path.join(
            output_dir, f"{dataset_name}_report.md"
        )
        print(f"Writing Markdown -> {md_path}", flush=True)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md)

        # ── Generate HTML report ────────────────────────────────────────
        if HAS_MARKDOWN:
            html_body = markdown.markdown(
                md, extensions=['tables']
            )
        else:
            html_body = f"<pre>{md}</pre>"

        html_path = os.path.join(
            output_dir, f"{dataset_name}_report.html"
        )
        print(f"Writing HTML -> {html_path}", flush=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(
                f"<!DOCTYPE html><html><head>"
                f"<meta charset='utf-8'>"
                f"<title>{dataset_name} Report</title>"
                f"<style>"
                f"body{{font-family:sans-serif;max-width:1200px;"
                f"margin:auto;padding:2em}}"
                f"table{{border-collapse:collapse;width:100%}}"
                f"th,td{{border:1px solid #ccc;padding:4px 8px;"
                f"font-size:0.8em}}"
                f"th{{background:#f5f5f5}}"
                f"</style>"
                f"</head><body>{html_body}</body></html>"
            )

        print(
            f"Report complete: {md_path}, {html_path}, {json_path}",
            flush=True
        )

    except Exception as e:
        print(f"ERROR generating report: {e}", flush=True)
        traceback.print_exc()


def _build_markdown_report(dataset_name, config, results):
    """Build Markdown report content."""
    md = []

    # Header
    md.append(f"# Validation Report: {dataset_name}")
    md.append(
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
    )
    md.append("")

    # Dataset overview
    md.append("## Dataset Overview")
    md.append(f"- **Subjects**: {results.get('n_subjects', 'N/A')}")
    md.append(f"- **Channels**: {results.get('n_channels', 'N/A')}")
    md.append(
        f"- **Sampling Rate**: {results.get('sampling_rate', 'N/A')} Hz"
    )
    md.append(f"- **Movements**: {results.get('n_movements', 'N/A')}")
    md.append("")

    # Processing parameters
    md.append("## Processing Parameters")
    for k, v in config.items():
        md.append(f"- **{k}**: {v}")
    md.append("")

    # Feature statistics
    stats = results.get('feature_stats', {})
    if stats:
        md.append("## Feature Statistics (first 20 features, Mean +/- Std)")
        classes = sorted(
            stats.keys(),
            key=lambda x: int(x) if str(x).lstrip('-').isdigit() else x
        )
        if classes:
            all_feats = list(stats[classes[0]].keys())
            feat_cols = all_feats[:20]
            header = "| Movement | " + " | ".join(feat_cols) + " |"
            sep = "|---|" + "|".join(["---"] * len(feat_cols)) + "|"
            md.append(header)
            md.append(sep)
            for cls in classes[:30]:  # Limit to 30 classes for readability
                row = f"| {cls} |"
                for fn in feat_cols:
                    if fn in stats[cls]:
                        mean, std = stats[cls][fn]
                        row += f" {mean:.4f} +/- {std:.4f} |"
                    else:
                        row += " N/A |"
                md.append(row)
            if len(classes) > 30:
                md.append(f"| ... | ({len(classes) - 30} more classes) |")
        md.append("")

    # Classification results
    clf_res = results.get('classification')
    if clf_res is not None:
        acc, std_acc, cm = clf_res
        md.append("## Classification Results")
        md.append(f"- **Strategy**: Leave-One-Subject-Out (LOSO)")
        md.append(f"- **Accuracy**: {acc:.2%} +/- {std_acc:.2%}")

        # Per-subject accuracy
        per_subj = results.get('per_subject_accuracy', [])
        if per_subj:
            md.append("")
            md.append("### Per-Subject Accuracy")
            md.append("| Subject | Accuracy |")
            md.append("|---|---|")
            for a in per_subj:
                if isinstance(a, dict):
                    subj = a.get('subject', '?')
                    acc_val = a.get('accuracy', 0.0)
                else:
                    subj = '?'
                    acc_val = float(a)
                md.append(f"| Subject {subj} | {acc_val:.2%} |")

        # Confusion matrix
        if plot_confusion_matrix is not None and cm:
            cm_path = os.path.join(
                results.get('_output_dir', '.'),
                f"{dataset_name}_cm.png"
            )
            # Try to plot confusion matrix
            class_names = results.get('class_names', [])
            try:
                plot_confusion_matrix(cm, class_names, cm_path)
                md.append(f"")
                md.append("### Confusion Matrix")
                md.append(
                    f"![Confusion Matrix]({os.path.basename(cm_path)})"
                )
            except Exception:
                md.append("*(Confusion matrix plot failed)*")
        md.append("")
    else:
        md.append("*No classification performed.*")
        md.append("")

    # CNN results
    cnn_res = results.get('cnn_results')
    if cnn_res:
        md.append("## CNN Baseline Results")
        md.append(
            f"- **Accuracy**: "
            f"{cnn_res['mean_accuracy']:.2%} +/- "
            f"{cnn_res['std_accuracy']:.2%}"
        )
        md.append(
            f"- **Inference Time**: "
            f"{cnn_res.get('inference_time_ms_mean', 0):.2f} +/- "
            f"{cnn_res.get('inference_time_ms_std', 0):.2f} ms"
        )
        md.append("")

    # SHAP results
    shap_files = results.get('shap_files', [])
    if shap_files:
        md.append("## SHAP Analysis")
        for f in shap_files:
            md.append(f"- [{os.path.basename(f)}]({os.path.basename(f)})")
        md.append("")

    # Issues
    issues = results.get('issues', [])
    md.append("## Issues")
    if issues:
        for issue in issues:
            md.append(f"- {issue}")
    else:
        md.append("None.")

    return "\n".join(md)
