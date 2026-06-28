#!/usr/bin/env python3
"""
app.py - EMG Analysis Dashboard
Multi‑channel, adaptive SNR, EDF support, memory warnings, channel selection,
simplified mode, PDF reports, database integration, statistics tab.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import tempfile
import os
import gc
import base64
from io import BytesIO

# Optional: EDF support
try:
    import pyedflib
    HAS_EDFLIB = True
except ImportError:
    HAS_EDFLIB = False

# Optional: memory profiling
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from core_engine import EMGFeatureExtractor, EMGConfig, EMGSignalSimulator
from emg_stats import descriptive_stats, compute_correlation_matrix, pca_analysis, t_test, fatigue_index
from database import init_db, save_session, load_sessions, load_session_by_id, delete_session
from pdf_report import generate_pdf_report

st.set_page_config(
    page_title="EMG Analysis Engine | Multi‑channel Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: 700; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem; }
    .info-box { background: #F3F4F6; padding: 1rem; border-radius: 10px; border-left: 4px solid #3B82F6; }
    .warning-box { background: #FEF3C7; padding: 1rem; border-radius: 10px; border-left: 4px solid #F59E0B; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'simulator' not in st.session_state:
    st.session_state.simulator = EMGSignalSimulator()
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_raw_signal' not in st.session_state:
    st.session_state.last_raw_signal = None
if 'n_channels' not in st.session_state:
    st.session_state.n_channels = 1
if 'simplified_mode' not in st.session_state:
    st.session_state.simplified_mode = False
if 'db_initialized' not in st.session_state:
    init_db()
    st.session_state.db_initialized = True
if 'select_all_channels' not in st.session_state:
    st.session_state.select_all_channels = True

st.markdown('<p class="main-header">🧬 Multi‑channel EMG Analysis Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Module A – IEEE/ISEK compliant | Adaptive SNR | Multi‑channel | EDF support</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/biotech.png", width=80)
    st.title("⚙️ Configuration")

    # Simplified mode toggle
    st.session_state.simplified_mode = st.checkbox("👩‍⚕️ Simplified Mode (for clinicians)", value=st.session_state.simplified_mode)

    st.markdown("### Signal Source")
    source_option = st.radio("Choose input source:", ["Simulation", "Upload File"], index=0)

    uploaded_file = None
    MAX_FILE_SIZE_MB = 50

    if source_option == "Upload File":
        st.markdown("#### Upload EMG data")
        uploaded_file = st.file_uploader(
            "Supported: CSV, TXT, NPY, EDF (multi‑channel)",
            type=['csv', 'txt', 'npy', 'edf'],
            help="Multi‑channel: columns = channels, rows = samples. EDF files supported via pyedflib."
        )
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.warning(f"⚠️ Large file ({file_size_mb:.2f} MB). May exceed memory limits.")

            if HAS_PSUTIL:
                mem = psutil.virtual_memory()
                st.caption(f"System memory available: {mem.available / 1024**3:.1f} GB")

            # Preview and channel selection
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_pre = pd.read_csv(uploaded_file, nrows=5)
                    numeric_cols = df_pre.select_dtypes(include=[np.number]).columns.tolist()
                    # Detect and remove time column
                    time_keywords = ['time', 'timestamp', 't', 'second', 'Time']
                    time_col = None
                    for col in df_pre.columns:
                        if col.lower() in [kw.lower() for kw in time_keywords]:
                            time_col = col
                            break
                    if time_col and time_col in numeric_cols:
                        numeric_cols.remove(time_col)
                        st.info(f"Detected time column '{time_col}' and removed it from channel list.")
                    elif time_col:
                        st.info(f"Detected non‑numeric time column '{time_col}' (ignored).")

                    st.session_state.n_channels = len(numeric_cols)
                    st.success(f"✅ CSV loaded. Found {st.session_state.n_channels} data channels.")

                    if st.session_state.n_channels > 1:
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Select All Channels"):
                                st.session_state.select_all_channels = True
                        with col2:
                            if st.button("Clear All"):
                                st.session_state.select_all_channels = False

                        channel_options = []
                        for i, col in enumerate(numeric_cols):
                            preview_vals = df_pre[col].values[:5]
                            preview_str = ", ".join([f"{v:.4f}" for v in preview_vals if not pd.isna(v)])
                            channel_options.append(f"Ch{i}: {col}  |  preview: {preview_str}...")

                        default_selection = channel_options if st.session_state.select_all_channels else []
                        selected_channels_str = st.multiselect(
                            "Select channels to analyze",
                            options=channel_options,
                            default=default_selection
                        )
                        st.session_state.selected_channels = [channel_options.index(s) for s in selected_channels_str]
                    else:
                        st.session_state.selected_channels = [0]

                elif uploaded_file.name.endswith('.edf'):
                    if not HAS_EDFLIB:
                        st.error("EDF support requires pyedflib. Install with: pip install pyedflib")
                        st.stop()
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    try:
                        f = pyedflib.EdfReader(tmp_path)
                        n_channels = f.signals_in_file
                        st.session_state.n_channels = n_channels
                        st.success(f"✅ EDF file loaded. {n_channels} signals found.")
                        if n_channels > 1:
                            channel_options = [f"Signal {i}: {f.getLabel(i)}" for i in range(n_channels)]
                            selected = st.multiselect("Select signals", channel_options, default=channel_options[:min(8, n_channels)])
                            st.session_state.selected_channels = [channel_options.index(s) for s in selected]
                        else:
                            st.session_state.selected_channels = [0]
                        f.close()
                    except Exception as e:
                        st.error(f"Error reading EDF: {e}")
                    finally:
                        os.unlink(tmp_path)

                elif uploaded_file.name.endswith('.txt'):
                    st.session_state.n_channels = 1
                    st.session_state.selected_channels = [0]

                elif uploaded_file.name.endswith('.npy'):
                    st.session_state.n_channels = "unknown"
                    st.session_state.selected_channels = [0]

            except Exception as e:
                st.warning(f"Could not parse file preview: {e}")
                st.session_state.n_channels = 1
                st.session_state.selected_channels = [0]

    # Hide advanced settings in simplified mode
    if not st.session_state.simplified_mode:
        st.markdown("### Performance Settings")
        use_downsampling = st.checkbox("Use downsampling (recommended for large files)", True)
        if use_downsampling:
            max_samples = st.number_input("Max samples per channel", 1000, 200000, 30000, 1000)

        benchmark_mode = st.checkbox("Benchmark mode (show processing time)", False)
        freq_features = st.checkbox("Compute frequency features (MDF, MNF)", False)

        st.markdown("### Advanced Filter Options")
        filter_type = st.selectbox("Filter type", ['butterworth', 'chebyshev', 'bessel', 'elliptic'], index=0)
        noise_method = st.selectbox("Noise estimation method", ['percentile', 'median', 'manual'], index=0)
        if noise_method == 'manual':
            manual_noise = st.number_input("Manual noise floor (μV)", value=10.0) / 1e6
        else:
            manual_noise = None
        psd_method = st.selectbox("PSD method", ['welch', 'fft'], index=0)
        chunk_duration = st.number_input("Chunk duration (seconds, 0=off)", value=0.0, min_value=0.0, step=1.0)
        if chunk_duration == 0.0:
            chunk_duration = None

    st.markdown("### Signal Parameters")
    if st.session_state.simplified_mode:
        sampling_rate = st.number_input("Sampling Rate (Hz)", value=2000, min_value=100, max_value=10000, step=100)
        intensity = st.slider("Intensity", 0.1, 2.0, 1.0, 0.1)
        cutoff_low = 20.0
        cutoff_high = 450.0
        filter_order = 4
        window_size_ms = 100
        overlap = 0.5
    else:
        sampling_rate = st.slider("Sampling Rate (Hz)", 500, 4000, 2000, 100)
        intensity = st.slider("Intensity", 0.1, 2.0, 1.0, 0.1)
        st.markdown("### Filter Settings")
        cutoff_low = st.slider("High‑pass (Hz)", 5.0, 50.0, 20.0, 5.0)
        cutoff_high = st.slider("Low‑pass (Hz)", 200.0, 500.0, 450.0, 10.0)
        filter_order = st.slider("Filter Order", 2, 8, 4, 1)

        st.markdown("### Feature Extraction")
        window_size_ms = st.slider("Window (ms)", 50, 300, 100, 10)
        overlap = st.slider("Overlap", 0.0, 0.9, 0.5, 0.1)

    if st.button("🚀 Initialize Engine", type="primary", use_container_width=True):
        config = EMGConfig(
            sampling_rate=sampling_rate,
            cutoff_low=cutoff_low,
            cutoff_high=cutoff_high,
            filter_order=filter_order,
            window_size=int(window_size_ms * sampling_rate / 1000),
            overlap=overlap,
            filter_type=filter_type if not st.session_state.simplified_mode else 'butterworth',
            noise_estimation_method=noise_method if not st.session_state.simplified_mode else 'percentile',
            manual_noise_floor=manual_noise if not st.session_state.simplified_mode else None,
            psd_method=psd_method if not st.session_state.simplified_mode else 'welch',
            chunk_duration=chunk_duration if not st.session_state.simplified_mode else None
        )
        st.session_state.engine = EMGFeatureExtractor(config)
        st.success("✅ Engine initialized!")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Export JSON", use_container_width=True) and st.session_state.processing_history:
            latest = st.session_state.processing_history[-1]
            st.download_button(
                label="Download JSON",
                data=json.dumps(latest, indent=2),
                file_name=f"emg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    with col2:
        if st.button("📄 Export PDF", use_container_width=True) and st.session_state.last_result is not None:
            pdf_bytes = generate_pdf_report(st.session_state.last_result, st.session_state.last_raw_signal)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

    # Session management
    st.markdown("---")
    st.markdown("### 💾 Saved Sessions")
    sessions = load_sessions(limit=5)
    for sess in sessions:
        sess_id, ts, fname, _, notes = sess
        if st.button(f"{ts} - {fname}", key=f"load_{sess_id}"):
            loaded = load_session_by_id(sess_id)
            if loaded:
                st.session_state.last_result = json.loads(loaded[3])
                st.success(f"Loaded session {sess_id}")

# Main metrics
if not st.session_state.simplified_mode:
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Sampling Rate", f"{sampling_rate} Hz")
    with col2: st.metric("Bandwidth", f"{cutoff_low:.0f}-{cutoff_high:.0f} Hz")
    with col3: st.metric("Window", f"{window_size_ms} ms", f"{overlap*100:.0f}% overlap")
    with col4: st.metric("Filter Order", f"{filter_order}")

# Process button
if st.button("🎯 Generate & Analyze", type="primary", use_container_width=True):
    if st.session_state.engine is None:
        st.warning("⚠️ Please initialize the engine first.")
    else:
        with st.spinner("Processing... (this may take a few seconds)"):
            # --- Read data ---
            if source_option == "Upload File" and uploaded_file is not None:
                try:
                    uploaded_file.seek(0)

                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        time_keywords = ['time', 'timestamp', 't', 'second', 'Time']
                        time_col = None
                        for col in df.columns:
                            if col.lower() in [kw.lower() for kw in time_keywords]:
                                time_col = col
                                break
                        if time_col and time_col in numeric_cols:
                            numeric_cols.remove(time_col)
                        if not numeric_cols:
                            st.error("No numeric data columns found.")
                            st.stop()

                        if hasattr(st.session_state, 'selected_channels') and st.session_state.selected_channels:
                            selected_indices = st.session_state.selected_channels
                            selected_indices = [i for i in selected_indices if i < len(numeric_cols)]
                            if not selected_indices:
                                st.error("Selected channels are out of range.")
                                st.stop()
                            selected_cols = [numeric_cols[i] for i in selected_indices]
                            raw_signal = df[selected_cols].values.astype(float)
                        else:
                            raw_signal = df[numeric_cols].values.astype(float)

                    elif uploaded_file.name.endswith('.txt'):
                        content = uploaded_file.read().decode('utf-8')
                        numbers = []
                        for line in content.split():
                            line = line.strip()
                            if line:
                                try:
                                    numbers.append(float(line))
                                except ValueError:
                                    continue
                        if not numbers:
                            st.error("No numeric data found in text file.")
                            st.stop()
                        raw_signal = np.array(numbers).reshape(-1, 1)

                    elif uploaded_file.name.endswith('.npy'):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        raw_signal = np.load(tmp_path)
                        os.unlink(tmp_path)
                        if raw_signal.ndim == 1:
                            raw_signal = raw_signal.reshape(-1, 1)
                        if hasattr(st.session_state, 'selected_channels') and st.session_state.selected_channels:
                            if raw_signal.shape[1] >= max(st.session_state.selected_channels)+1:
                                raw_signal = raw_signal[:, st.session_state.selected_channels]
                            else:
                                st.warning("Selected channels exceed available channels; using all.")

                    elif uploaded_file.name.endswith('.edf') and HAS_EDFLIB:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        try:
                            f = pyedflib.EdfReader(tmp_path)
                            n_signals = f.signals_in_file
                            signals = [f.readSignal(i) for i in range(n_signals)]
                            min_len = min(len(s) for s in signals)
                            signals = [s[:min_len] for s in signals]
                            raw_signal = np.column_stack(signals)
                            f.close()
                            if hasattr(st.session_state, 'selected_channels') and st.session_state.selected_channels:
                                raw_signal = raw_signal[:, st.session_state.selected_channels]
                        except Exception as e:
                            st.error(f"EDF read error: {e}")
                            st.stop()
                        finally:
                            os.unlink(tmp_path)
                    else:
                        st.error("Unsupported file format")
                        st.stop()

                    # Force 2D
                    if raw_signal.ndim == 1:
                        raw_signal = raw_signal.reshape(-1, 1)
                    elif raw_signal.ndim != 2:
                        st.error(f"Invalid data dimensions: {raw_signal.ndim}D. Expected 1D or 2D.")
                        st.stop()

                    n_channels = raw_signal.shape[1]
                    st.session_state.n_channels = n_channels

                    # Downsampling if enabled
                    if not st.session_state.simplified_mode and use_downsampling and raw_signal.shape[0] > max_samples:
                        step = raw_signal.shape[0] // max_samples
                        if step > 1:
                            raw_signal = raw_signal[::step, :]
                            st.info(f"📉 Downsampled to {raw_signal.shape[0]} samples per channel for visualization.")

                    if intensity != 1.0:
                        raw_signal = raw_signal * intensity

                    actual_duration = raw_signal.shape[0] / sampling_rate
                    selected_ch = 0

                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.stop()
            else:
                # Simulation
                if source_option == "Upload File" and uploaded_file is None:
                    st.warning("Please upload a file or switch to Simulation.")
                    st.stop()
                n_sim_channels = 1
                raw_signal = st.session_state.simulator.generate_contraction(
                    duration=3.0, sampling_rate=sampling_rate, intensity=intensity, n_channels=n_sim_channels
                )
                if raw_signal.ndim == 1:
                    raw_signal = raw_signal.reshape(-1, 1)
                actual_duration = 3.0
                selected_ch = 0
                st.session_state.n_channels = n_sim_channels

            # Store raw signal for PDF
            st.session_state.last_raw_signal = raw_signal

            # --- Process ---
            try:
                results = st.session_state.engine.process_stream(
                    raw_signal,
                    selected_channel=selected_ch,
                    measure_time=(not st.session_state.simplified_mode and benchmark_mode),
                    compute_freq_features=(not st.session_state.simplified_mode and freq_features)
                )
                st.session_state.processing_history.append(results)
                st.session_state.last_result = results

                if not st.session_state.simplified_mode and benchmark_mode and 'benchmark' in results:
                    st.info(f"⚡ Processing time: {results['benchmark']['processing_time_ms']:.2f} ms")

                notes = st.text_input("Add notes for this session (optional)")
                if st.button("💾 Save this session"):
                    filename = uploaded_file.name if uploaded_file and hasattr(uploaded_file, 'name') else "simulation"
                    save_session(filename, json.dumps(results), notes)
                    st.success("Session saved.")

                gc.collect()

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

            # --- Display results ---
            n_ch = st.session_state.n_channels
            sel_ch = selected_ch

            if st.session_state.simplified_mode:
                st.subheader("📋 Clinical Summary")
                colA, colB, colC = st.columns(3)
                with colA:
                    snr = results['signal_quality']['mean_snr']
                    if snr > 25:
                        snr_qual = "Excellent"
                    elif snr > 20:
                        snr_qual = "Good"
                    elif snr > 15:
                        snr_qual = "Acceptable"
                    else:
                        snr_qual = "Poor"
                    st.metric("Signal Quality", snr_qual, f"{snr:.1f} dB")
                with colB:
                    mean_act = results['summary_statistics'].get('channel_0', {}).get('mean_activation', 0)
                    if mean_act < 0.1:
                        act_desc = "Resting"
                    elif mean_act < 0.3:
                        act_desc = "Light"
                    elif mean_act < 0.6:
                        act_desc = "Moderate"
                    else:
                        act_desc = "Strong"
                    st.metric("Muscle Activity", act_desc)
                with colC:
                    peak = results['summary_statistics'].get('channel_0', {}).get('peak_activation', 0)
                    st.metric("Peak Activation", f"{peak:.3f}")

                fig = go.Figure()
                if raw_signal.ndim == 2:
                    plot_sig = raw_signal[:, 0]
                else:
                    plot_sig = raw_signal
                t = np.linspace(0, actual_duration, len(plot_sig))
                fig.add_trace(go.Scatter(x=t, y=plot_sig, mode='lines', name='EMG'))
                fig.update_layout(title="EMG Signal", xaxis_title="Time (s)", yaxis_title="Amplitude", height=400)
                st.plotly_chart(fig, use_container_width=True)

            else:
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📈 Signal Analysis",
                    "📊 Feature Extraction",
                    "📉 Spectral Analysis",
                    "📋 Technical Report",
                    "📐 Statistics"
                ])

                with tab1:
                    fig = make_subplots(rows=3, cols=1,
                                        subplot_titles=(f'Raw EMG (Ch0)', f'Filtered (Ch0)', 'Activation Envelope'),
                                        vertical_spacing=0.1)
                    sig_raw = raw_signal[:, 0]
                    t_raw = np.linspace(0, actual_duration, len(sig_raw))
                    fig.add_trace(go.Scatter(x=t_raw, y=sig_raw, mode='lines', name='Raw', line=dict(color='lightgray')), row=1, col=1)

                    filtered_full = st.session_state.engine.preprocess(raw_signal)
                    if filtered_full.ndim == 2:
                        sig_filt = filtered_full[:, 0]
                    else:
                        sig_filt = filtered_full
                    t_filt = np.linspace(0, actual_duration, len(sig_filt))
                    fig.add_trace(go.Scatter(x=t_filt, y=sig_filt, mode='lines', name='Filtered', line=dict(color='#3B82F6')), row=2, col=1)

                    window_samples = st.session_state.engine.config.window_size
                    rms_vals = []
                    t_rms = []
                    for i in range(0, len(sig_filt) - window_samples, window_samples//2):
                        seg = sig_filt[i:i+window_samples]
                        rms_vals.append(np.sqrt(np.mean(seg**2)))
                        t_rms.append(i / sampling_rate)
                    fig.add_trace(go.Scatter(x=t_rms, y=rms_vals, mode='lines', name='RMS Envelope', line=dict(color='#10B981', width=3)), row=3, col=1)

                    fig.update_layout(height=600, showlegend=True)
                    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    st.subheader("Features per channel (latest window)")
                    if 'time_series' in results and 'features' in results['time_series']:
                        feat_list = results['time_series']['features']
                        rows = []
                        for ch in range(len(feat_list)):
                            if len(feat_list[ch]) > 0:
                                latest = feat_list[ch][-1]
                                row = {'Channel': ch,
                                       'MAV': f"{latest['MAV']:.4f}",
                                       'RMS': f"{latest['RMS']:.4f}",
                                       'ZCR': f"{latest['ZCR']:.4f}",
                                       'WL': f"{latest['WL']:.4f}",
                                       'SSC': f"{latest['SSC']:.4f}"}
                                if freq_features and 'freq_features' in results['time_series']:
                                    f_list = results['time_series']['freq_features']
                                    if len(f_list) > ch and len(f_list[ch]) > 0:
                                        latest_f = f_list[ch][-1]
                                        row['MDF (Hz)'] = f"{latest_f['MDF']:.1f}"
                                        row['MNF (Hz)'] = f"{latest_f['MNF']:.1f}"
                                rows.append(row)
                        if rows:
                            df_feat = pd.DataFrame(rows)
                            st.dataframe(df_feat, use_container_width=True)

                    if len(feat_list) > 0 and len(feat_list[0]) > 0:
                        timestamps = results['time_series']['timestamps']
                        df_ch = pd.DataFrame(feat_list[0])
                        fig2 = make_subplots(rows=2, cols=2,
                                             subplot_titles=('MAV', 'RMS', 'ZCR', 'WL'))
                        fig2.add_trace(go.Scatter(x=timestamps, y=df_ch['MAV'], mode='lines', name='MAV', line=dict(color='#EF4444')), row=1, col=1)
                        fig2.add_trace(go.Scatter(x=timestamps, y=df_ch['RMS'], mode='lines', name='RMS', line=dict(color='#3B82F6')), row=1, col=2)
                        fig2.add_trace(go.Scatter(x=timestamps, y=df_ch['ZCR'], mode='lines', name='ZCR', line=dict(color='#10B981')), row=2, col=1)
                        fig2.add_trace(go.Scatter(x=timestamps, y=df_ch['WL'], mode='lines', name='WL', line=dict(color='#F59E0B')), row=2, col=2)
                        fig2.update_layout(height=500, showlegend=False)
                        st.plotly_chart(fig2, use_container_width=True)

                    if 'summary_statistics' in results:
                        stats = results['summary_statistics']
                        colA, colB, colC = st.columns(3)
                        with colA:
                            st.markdown('<div class="info-box"><b>📊 Summary (Ch0)</b><br>'
                                        f'Mean Activation: {stats.get("channel_0", {}).get("mean_activation", 0):.3f}<br>'
                                        f'Peak Activation: {stats.get("channel_0", {}).get("peak_activation", 0):.3f}<br>'
                                        f'Fatigue Index: {stats.get("channel_0", {}).get("fatigue_index", 0):.3f}</div>',
                                        unsafe_allow_html=True)
                        with colB:
                            noise = results['signal_quality'].get('estimated_noise_floor', 0.01)
                            snr = results['signal_quality'].get('mean_snr', 0)
                            st.markdown(f'<div class="info-box"><b>🔧 Quality</b><br>'
                                        f'Noise floor: {noise:.4f}<br>'
                                        f'SNR: {snr:.1f} dB</div>',
                                        unsafe_allow_html=True)

                with tab3:
                    from scipy.fft import fft, fftfreq
                    sig_fft = raw_signal[:, 0]
                    N = len(sig_fft)
                    yf = fft(sig_fft)
                    xf = fftfreq(N, 1/sampling_rate)[:N//2]
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(x=xf, y=2.0/N * np.abs(yf[:N//2]), mode='lines', fill='tozeroy', name='Power Spectrum', line=dict(color='#8B5CF6')))
                    fig3.add_vrect(x0=20, x1=150, fillcolor="green", opacity=0.2, line_width=0, annotation_text="Low")
                    fig3.add_vrect(x0=150, x1=450, fillcolor="blue", opacity=0.2, line_width=0, annotation_text="EMG band")
                    fig3.update_layout(title="Frequency Domain Analysis (Channel 0)", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", height=400)
                    st.plotly_chart(fig3, use_container_width=True)

                with tab4:
                    st.markdown("### 🔬 Technical Analysis Report")
                    mean_mav = results.get('summary_statistics', {}).get('channel_0', {}).get('mean_activation', 0)
                    if mean_mav < 0.1:
                        act_desc = "Very low (resting)"
                    elif mean_mav < 0.3:
                        act_desc = "Low (light contraction)"
                    elif mean_mav < 0.6:
                        act_desc = "Moderate (normal)"
                    else:
                        act_desc = "High (strong)"

                    snr = results['signal_quality'].get('mean_snr', 0)
                    if snr > 25:
                        snr_desc = "Excellent"
                    elif snr > 20:
                        snr_desc = "Good"
                    elif snr > 15:
                        snr_desc = "Acceptable"
                    else:
                        snr_desc = "Poor"

                    st.markdown(f"""
**Clinical Interpretation:**
- **Muscle Activity:** {act_desc}
- **Signal Quality:** {snr_desc} (SNR: {snr:.1f} dB)

**Signal Characteristics:**
- Duration: {actual_duration:.2f} seconds
- Samples: {raw_signal.shape[0]}
- Channels: {raw_signal.shape[1] if raw_signal.ndim==2 else 1}
""")
                    with st.expander("Raw JSON Output"):
                        st.json(results)

                with tab5:
                    st.subheader("📐 Statistical Analysis")
                    if 'time_series' in results and 'features' in results['time_series']:
                        feat_list = results['time_series']['features']
                        if len(feat_list) > 0 and len(feat_list[0]) > 0:
                            n_channels = len(feat_list)
                            n_windows = len(feat_list[0])
                            if n_windows > 0:
                                feature_names = list(feat_list[0][0].keys())
                                # Build 3D array
                                feat_array = np.zeros((n_windows, n_channels, len(feature_names)))
                                for ch in range(n_channels):
                                    for w in range(n_windows):
                                        for f_idx, fname in enumerate(feature_names):
                                            feat_array[w, ch, f_idx] = feat_list[ch][w][fname]

                                # Descriptive stats
                                desc_df = descriptive_stats(feat_array, feature_names)
                                st.write("**Descriptive Statistics per Channel**")
                                st.dataframe(desc_df, use_container_width=True)

                                # Fatigue index for channel 0
                                rms_vals = [f['RMS'] for f in feat_list[0]]
                                step_sec = (window_size_ms * (1 - overlap)) / 1000.0
                                fs_feat = 1.0 / step_sec if step_sec > 0 else 1.0
                                fi = fatigue_index(np.array(rms_vals), fs_feat)
                                st.metric("Fatigue Index (Ch0)", f"{fi:.3f}")
                                if fi < 0:
                                    st.caption("⚠️ Negative fatigue index indicates decreasing RMS over time – possible muscle fatigue.")
                                else:
                                    st.caption("✅ Positive fatigue index indicates increasing activity.")

                                # Correlation matrix if multiple channels
                                if n_channels > 1:
                                    st.write("**Correlation Matrix (RMS across channels)**")
                                    rms_matrix = np.array([[feat_list[ch][w]['RMS'] for w in range(n_windows)] for ch in range(n_channels)]).T
                                    corr = compute_correlation_matrix(rms_matrix)
                                    st.dataframe(corr, use_container_width=True)

                                # PCA if enough data
                                if n_channels >= 3 and n_windows >= 5:
                                    st.write("**PCA Analysis (RMS)**")
                                    rms_matrix = np.array([[feat_list[ch][w]['RMS'] for w in range(n_windows)] for ch in range(n_channels)]).T
                                    pca_res = pca_analysis(rms_matrix, n_components=2)
                                    st.write(f"Explained variance ratio: {pca_res['explained_variance_ratio']}")
                                    # Plot PCA
                                    comp = np.array(pca_res['components'])
                                    fig_pca = go.Figure()
                                    fig_pca.add_trace(go.Scatter(x=comp[:,0], y=comp[:,1],
                                                                  mode='markers', marker=dict(color='blue'),
                                                                  text=[f"Window {i}" for i in range(comp.shape[0])]))
                                    fig_pca.update_layout(title="PCA Projection (first 2 components)")
                                    st.plotly_chart(fig_pca, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280;'>
    Module A – Multi‑channel EMG Analysis Engine | © 2026 Qussai Adlbi<br>
    <small>Adaptive SNR • Multi‑channel • EDF support • Frequency features • Statistics • Database • PDF reports</small>
</div>
""", unsafe_allow_html=True)