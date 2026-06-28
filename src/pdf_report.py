#!/usr/bin/env python3
"""
pdf_report.py - Generate comprehensive PDF reports from EMG analysis results.
Includes all features, statistics, and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import pandas as pd
from datetime import datetime
import plotly.io as pio
from emg_stats import descriptive_stats, compute_correlation_matrix, pca_analysis


def generate_pdf_report(results, raw_signal=None, detailed=True):
    """
    Generate a comprehensive PDF report.
    If detailed=True, includes all features, correlation, PCA, etc.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        alignment=1
    )
    story.append(Paragraph("EMG Analysis Report", title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Metadata
    meta = results.get('metadata', {})
    timestamp = meta.get('timestamp', datetime.now().isoformat())
    story.append(Paragraph(f"<b>Date:</b> {timestamp}", styles['Normal']))
    story.append(Paragraph(f"<b>Sampling Rate:</b> {meta.get('sampling_rate', 'N/A')} Hz", styles['Normal']))
    story.append(Paragraph(f"<b>Channels:</b> {meta.get('n_channels', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Selected Channel:</b> {meta.get('selected_channel', 'N/A')}", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    # Signal quality
    quality = results.get('signal_quality', {})
    snr = quality.get('mean_snr', 0)
    noise = quality.get('estimated_noise_floor', 0)
    story.append(Paragraph("<b>Signal Quality</b>", styles['Heading2']))
    story.append(Paragraph(f"SNR: {snr:.1f} dB", styles['Normal']))
    story.append(Paragraph(f"Noise floor: {noise:.6f}", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    # Summary statistics per channel (from summary_statistics)
    stats = results.get('summary_statistics', {})
    if stats:
        story.append(Paragraph("<b>Summary per Channel</b>", styles['Heading2']))
        data = [['Channel', 'Mean Act.', 'Peak Act.', 'Fatigue Idx']]
        for ch, vals in stats.items():
            if ch.startswith('channel_'):
                idx = ch.replace('channel_', '')
                data.append([
                    idx,
                    f"{vals.get('mean_activation', 0):.3f}",
                    f"{vals.get('peak_activation', 0):.3f}",
                    f"{vals.get('fatigue_index', 0):.3f}"
                ])
        if len(data) > 1:
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 10),
                ('BOTTOMPADDING', (0,0), (-1,0), 8),
                ('GRID', (0,0), (-1,-1), 0.5, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 0.1 * inch))

    # Detailed features
    if detailed:
        time_series = results.get('time_series', {})
        features = time_series.get('features', [])
        if features and len(features) > 0 and len(features[0]) > 0:
            story.append(Paragraph("<b>Feature Statistics (per channel)</b>", styles['Heading2']))

            n_channels = len(features)
            n_windows = len(features[0])
            if n_windows > 0:
                # Get feature names from first window
                feature_names = list(features[0][0].keys())
                # Build array of shape (n_windows, n_channels, n_features)
                feat_array = np.zeros((n_windows, n_channels, len(feature_names)))
                for ch in range(n_channels):
                    for w in range(n_windows):
                        for f_idx, fname in enumerate(feature_names):
                            feat_array[w, ch, f_idx] = features[ch][w][fname]

                # Descriptive stats table
                desc_df = descriptive_stats(feat_array, feature_names)
                # Convert to reportlab table
                table_data = [['Channel', 'Feature', 'Mean', 'Std', 'Min', 'Max']]
                for _, row in desc_df.iterrows():
                    table_data.append([
                        row['Channel'],
                        row['Feature'],
                        f"{row['Mean']:.4f}",
                        f"{row['Std']:.4f}",
                        f"{row['Min']:.4f}",
                        f"{row['Max']:.4f}"
                    ])
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (1,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 8),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 0.1 * inch))

                # RMS over time plot
                timestamps = time_series.get('timestamps', [])
                if len(timestamps) > 0:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    for ch in range(min(n_channels, 8)):  # limit to 8 channels
                        rms_vals = [features[ch][w]['RMS'] for w in range(len(features[ch]))]
                        ax.plot(timestamps[:len(rms_vals)], rms_vals, label=f'Ch{ch}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('RMS')
                    ax.set_title('RMS over time')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    img_buf = BytesIO()
                    fig.savefig(img_buf, format='png', dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    img_buf.seek(0)
                    story.append(Image(img_buf, width=6*inch, height=4*inch))
                    story.append(Spacer(1, 0.1 * inch))

                # Correlation matrix if multi-channel
                if n_channels > 1:
                    # Build matrix of RMS across channels and windows
                    rms_matrix = np.array([[features[ch][w]['RMS'] for w in range(n_windows)] for ch in range(n_channels)]).T
                    corr = compute_correlation_matrix(rms_matrix)
                    story.append(Paragraph("<b>Correlation Matrix (RMS)</b>", styles['Heading2']))
                    corr_data = [[''] + list(corr.columns)] + [[corr.index[i]] + [f"{corr.iloc[i,j]:.3f}" for j in range(len(corr.columns))] for i in range(len(corr.index))]
                    corr_table = Table(corr_data)
                    corr_table.setStyle(TableStyle([
                        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.black)
                    ]))
                    story.append(corr_table)
                    story.append(Spacer(1, 0.1 * inch))

                # PCA if enough data
                if n_channels >= 3 and n_windows >= 5:
                    rms_matrix = np.array([[features[ch][w]['RMS'] for w in range(n_windows)] for ch in range(n_channels)]).T
                    pca_res = pca_analysis(rms_matrix, n_components=2)
                    story.append(Paragraph("<b>PCA (first 2 components)</b>", styles['Heading2']))
                    story.append(Paragraph(f"Explained variance: {pca_res['explained_variance_ratio'][0]:.3f}, {pca_res['explained_variance_ratio'][1]:.3f}", styles['Normal']))
                    # Plot PCA
                    comp = np.array(pca_res['components'])
                    fig2, ax2 = plt.subplots(figsize=(5,5))
                    ax2.scatter(comp[:,0], comp[:,1])
                    ax2.set_xlabel('PC1')
                    ax2.set_ylabel('PC2')
                    ax2.set_title('PCA Projection')
                    ax2.grid(True, alpha=0.3)
                    img_buf2 = BytesIO()
                    fig2.savefig(img_buf2, format='png', dpi=100, bbox_inches='tight')
                    plt.close(fig2)
                    img_buf2.seek(0)
                    story.append(Image(img_buf2, width=5*inch, height=5*inch))
                    story.append(Spacer(1, 0.1 * inch))

    # Raw signal plot (always included)
    if raw_signal is not None and raw_signal.size > 0:
        story.append(PageBreak())
        story.append(Paragraph("<b>Raw EMG Signal (Channel 0)</b>", styles['Heading2']))
        fig3, ax3 = plt.subplots(figsize=(6, 2))
        if raw_signal.ndim == 2:
            sig = raw_signal[:, 0]
        else:
            sig = raw_signal
        time = np.arange(len(sig)) / meta.get('sampling_rate', 2000)
        ax3.plot(time, sig, 'b-', linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.grid(True, alpha=0.3)
        img_buf3 = BytesIO()
        fig3.savefig(img_buf3, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig3)
        img_buf3.seek(0)
        story.append(Image(img_buf3, width=6*inch, height=2*inch))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes