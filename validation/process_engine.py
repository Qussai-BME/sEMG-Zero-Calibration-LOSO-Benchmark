"""
process_engine.py - v16.0 (SELECTIVE INTELLIGENCE — Adaptive Engine)

═══════════════════════════════════════════════════════════════════════
CHANGES from v15.0:
═══════════════════════════════════════════════════════════════════════

  v16.0 NEW FEATURES:

  1. Euclidean Alignment (domain adaptation — He & Wu 2020)
     - Reduces inter-subject variability by spatially aligning each
       subject's signal distribution to a reference
     - Applied per-subject AFTER bandpass filtering, BEFORE windowing
     - Config key: euclidean_alignment (bool, default False)

  2. TKEO Band Energy Features (Phinyomark et al. 2012)
     - Band-specific TKEO energy in 4 frequency bands per channel
     - Bands: (20-100, 100-200, 200-350, 350-450 Hz)
     - Improves discrimination between similar gestures
     - Config key: compute_tkeo_bands (bool, default False)

  3. Covariance Features (Riemannian-inspired — Barachant et al. 2013)
     - Upper triangle of per-window trace-normalized covariance matrix
     - Captures spatial relationships invariant to amplitude scaling
     - Config key: compute_covariance_features (bool, default False)

  ALL v15.0 FEATURES PRESERVED:
  - window_size_ms → window_size conversion
  - Active Signal Detection respects remove_class_zero
  - Streaming fallback uses correct W from window_size_ms
  - Active Signal Detection
  - Smart Downsampling
  - Full Inter-Channel Correlation
  - Adaptive Overlap
  - float32-first filtering, channel-by-channel fallback
  - Stride-trick windowing, chunked extraction
  - Streaming fallback for large signals
═══════════════════════════════════════════════════════════════════════
"""

import sys
import gc
import numpy as np
from scipy.signal import butter, sosfiltfilt

try:
    from src.core_engine import EMGConfig, EMGFeatureExtractor
except ImportError:
    EMGConfig = None
    EMGFeatureExtractor = None

try:
    import pywt
    HAS_WAVELETS = True
except ImportError:
    HAS_WAVELETS = False


def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# =====================================================================
# Active Signal Detection (v14.0 — removes rest/noise)
# =====================================================================

def active_signal_detection(emg, threshold=0.02):
    """
    Remove low-energy (rest/noise) segments from EMG signal.

    For each sample, compute mean absolute energy across channels.
    Keep only samples where at least one channel exceeds the threshold.

    Parameters
    ----------
    emg : np.ndarray, shape (N, C), dtype float32
    threshold : float, energy threshold (default: 0.02)

    Returns
    -------
    emg_active : np.ndarray, shape (N_active, C)
    active_mask : np.ndarray of bool, shape (N,)
    """
    if emg.shape[0] == 0:
        return emg, np.ones(emg.shape[0], dtype=bool)

    # Compute per-sample energy (mean absolute across channels)
    energy = np.mean(np.abs(emg), axis=1)

    # Keep samples where energy exceeds threshold
    mask = energy > threshold
    n_active = int(mask.sum())
    n_total = len(mask)

    if n_active < n_total * 0.1:
        # Safety: don't remove more than 90% of data
        debug_print(
            f"[active_signal] Would remove {100*(1-n_active/n_total):.1f}% "
            f"(>{90:.0f}%). Keeping all data."
        )
        return emg, np.ones(n_total, dtype=bool)

    if n_active == 0:
        return emg[:1], np.ones(1, dtype=bool)

    debug_print(
        f"[active_signal] Kept {n_active}/{n_total} samples "
        f"({100*n_active/n_total:.1f}%)"
    )
    return emg[mask], mask


# =====================================================================
# Euclidean Alignment (v16.0 — domain adaptation)
# =====================================================================

def euclidean_alignment(emg):
    """
    Euclidean Alignment (He & Wu 2020) — domain adaptation for EMG.

    Reduces inter-subject variability by spatially aligning each subject's
    signal distribution to a reference. Applied per-subject AFTER bandpass
    filtering, BEFORE windowing.

    Parameters
    ----------
    emg : np.ndarray, shape (N, C), dtype float32

    Returns
    -------
    emg_aligned : np.ndarray, shape (N, C), dtype float32
    """
    N, C = emg.shape
    if N < 10 or C < 2:
        return emg

    # Compute reference spatial covariance: R = (1/N) * X.T @ X
    R = (emg.T @ emg) / N  # (C, C)

    # Regularize for numerical stability
    R = R + 1e-6 * np.eye(C, dtype=np.float32)

    try:
        # Cholesky decomposition: R = L @ L.T
        L = np.linalg.cholesky(R.astype(np.float64))
        # Inverse square root: R^{-1/2} = L^{-T}
        R_inv_sqrt = np.linalg.inv(L).T
        # Apply alignment to all samples: x_aligned = R^{-1/2} @ x
        emg_aligned = (R_inv_sqrt @ emg.T.astype(np.float64)).T
        return np.ascontiguousarray(emg_aligned, dtype=np.float32)
    except np.linalg.LinAlgError:
        debug_print("[EA] Cholesky failed, returning original signal")
        return emg


# =====================================================================
# Feature helper functions (vectorized, memory-efficient)
# =====================================================================

def _ar_autocorr(wd, order):
    """Autoregressive coefficients via normalized autocorrelation."""
    nw, nc, T = wd.shape
    w = wd - wd.mean(axis=2, keepdims=True)
    energy = (w ** 2).sum(axis=2, keepdims=True) + 1e-12
    out = np.empty((nw, nc, order), dtype=np.float32)
    for k in range(1, order + 1):
        out[:, :, k - 1] = (w[:, :, k:] * w[:, :, :T - k]).sum(2) / energy[:, :, 0]
    return out


def _hjorth(wd):
    """Hjorth parameters: Activity, Mobility, Complexity."""
    d1 = np.diff(wd, axis=2)
    d2 = np.diff(d1, axis=2)
    v0 = np.var(wd, axis=2, ddof=1) + 1e-12
    v1 = np.var(d1, axis=2, ddof=1) + 1e-12
    v2 = np.var(d2, axis=2, ddof=1) + 1e-12
    mob = np.sqrt(v1 / v0)
    return v0, mob, np.sqrt(v2 / v1) / mob


def _tkeo(wd):
    """Teager-Kaiser Energy Operator (mean absolute)."""
    return np.mean(np.abs(
        wd[:, :, 1:-1] ** 2 - wd[:, :, :-2] * wd[:, :, 2:]
    ), axis=2)


def _inter_ch_corr(wd, max_channels=0, corr_stride=1):
    """
    Vectorized Pearson correlation across channel pairs.
    v34.0: Fused normalization (avoids intermediate division).
    """
    nw, nc, T = wd.shape

    if corr_stride > 1:
        wd = wd[:, :, ::corr_stride]
        T = wd.shape[2]

    if max_channels > 0:
        nc = min(max_channels, nc)
        wd = wd[:, :nc, :]

    w = wd - wd.mean(axis=2, keepdims=True)
    w_sq_sum = (w ** 2).sum(axis=2, keepdims=True)
    denom = np.sqrt(w_sq_sum * w_sq_sum.transpose(0, 2, 1)) + 1e-12
    corr = np.einsum('wct,wdt->wcd', w, w) / denom
    idx = np.triu_indices(nc, k=1)
    return corr[:, idx[0], idx[1]]


def _tkeo_band_features(wd, fs):
    """TKEO energy in 4 frequency bands per window per channel.

    Phinyomark et al. 2012: Band-specific TKEO improves discrimination
    between similar gestures.

    Parameters
    ----------
    wd : np.ndarray, shape (n_windows, n_channels, W)
    fs : int, sampling rate

    Returns
    -------
    out : np.ndarray, shape (n_windows, n_channels * 4)
        Log TKEO energy for each of 4 bands: (20-100, 100-200, 200-350, 350-450 Hz)
    """
    nw, nc, W = wd.shape
    bands = [(20, 100), (100, 200), (200, 350), (350, 450)]
    out = np.zeros((nw, nc, len(bands)), dtype=np.float32)

    # Compute TKEO on full window
    tkeo = wd[:, :, 1:-1] ** 2 - wd[:, :, :-2] * wd[:, :, 2:]

    # Use FFT to extract band energy (faster than IIR filtering)
    n_fft = tkeo.shape[2]
    if n_fft < 4:
        return out.reshape(nw, nc * len(bands))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    tkeo_fft = np.abs(np.fft.rfft(tkeo, axis=2))  # (nw, nc, n_fft//2+1)

    for bi, (flo, fhi) in enumerate(bands):
        if fhi > fs / 2:
            continue
        mask = (freqs >= flo) & (freqs < fhi)
        if mask.sum() == 0:
            continue
        band_energy = tkeo_fft[:, :, mask].mean(axis=2)
        out[:, :, bi] = np.log(band_energy + 1e-10)

    return out.reshape(nw, nc * len(bands))


def _covariance_features(wd):
    """Upper triangle of per-window covariance matrix (trace-normalized).

    Simplified Riemannian-inspired features. Captures spatial relationships
    between channels invariant to amplitude scaling.

    Barachant et al. 2013: Covariance-based features for BCI/EMG.
    Trace normalization provides scale invariance (amplitude-independent).

    Parameters
    ----------
    wd : np.ndarray, shape (n_windows, n_channels, W)

    Returns
    -------
    out : np.ndarray, shape (n_windows, C*(C+1)/2)
    """
    nw, nc, W = wd.shape
    if nc < 2:
        return np.zeros((nw, 1), dtype=np.float32)

    # Center the data
    centered = wd - wd.mean(axis=2, keepdims=True)  # (nw, nc, W)

    # Covariance: (1/W) * centered @ centered.T for each window
    covs = np.einsum('wct,wet->wce', centered, centered) / W  # (nw, C, C)

    # Normalize by trace for scale invariance
    traces = np.einsum('wcc->w', covs)[:, None, None] + 1e-10
    covs_norm = covs / traces

    # Extract upper triangle
    idx = np.triu_indices(nc)
    out = covs_norm[:, idx[0], idx[1]]  # (nw, C*(C+1)/2)
    return out.astype(np.float32)


def _wavelet_features_batch(wd, wavelet='db4', level=4):
    """Wavelet energy and entropy features (batched)."""
    if not HAS_WAVELETS:
        return None, None
    nw, nc, W = wd.shape
    n_bands = level + 1
    wd_flat = wd.reshape(nw * nc, W)
    coeffs = pywt.wavedec(
        wd_flat, wavelet, level=level,
        mode='periodization', axis=-1
    )
    total_e = sum(np.sum(c ** 2, axis=1) for c in coeffs) + 1e-12
    energy = np.zeros((nw * nc, n_bands), dtype=np.float32)
    entropy = np.zeros((nw * nc, n_bands), dtype=np.float32)
    for bi, c in enumerate(coeffs):
        c_sq = c ** 2
        band_e = np.sum(c_sq, axis=1)
        energy[:, bi] = band_e / total_e
        p = np.clip(c_sq / (c_sq.sum(1, keepdims=True) + 1e-12), 1e-12, 1.0)
        entropy[:, bi] = -(p * np.log(p)).sum(axis=1)
    return energy.reshape(nw, nc, n_bands), entropy.reshape(nw, nc, n_bands)


# =====================================================================
# Core feature extraction per chunk
# =====================================================================

def _extract_chunk(wd, cfg_dict, do_freq, do_ar, ar_order,
                   do_hjorth, do_icc, do_wavelet, wav_name, wav_level,
                   ssc_thr, fft_pad2, fs, W, corr_channels=0,
                   corr_stride=1, do_tkeo_bands=False,
                   do_cov_features=False):
    """
    Extract all features from a chunk of windows.
    wd : np.ndarray, shape (n_windows, n_channels, window_samples)
    Returns (features_flat, feature_names)
    """
    eps = 1e-12
    nw, C = wd.shape[0], wd.shape[1]
    wd = wd.astype(np.float32)
    abs_wd = np.abs(wd)

    # ── Time-domain features (16 per channel) ─────────────────────────
    iemg = abs_wd.sum(2)
    mav = abs_wd.mean(2)
    log_mav = np.log(mav + eps)
    half = W // 2
    mavs = abs_wd[:, :, half:].mean(2) - abs_wd[:, :, :half].mean(2)
    ssi = (wd ** 2).sum(2)
    rms = np.sqrt(ssi / W)
    log_rms = np.log(rms + eps)
    vo = (abs_wd ** 3).mean(2) ** (1.0 / 3.0)
    log_det = np.exp(np.log(abs_wd + eps).mean(2))
    wl = np.abs(np.diff(wd, axis=2)).sum(2)
    zcr = (np.diff(np.sign(wd), axis=2) != 0).sum(2) / (W - 1)
    d1 = np.diff(wd, axis=2)
    sd = np.sign(d1)
    if ssc_thr > 0:
        sd = sd * (np.abs(d1) > ssc_thr)
    ssc = (sd[:, :, 1:] != sd[:, :, :-1]).sum(2)
    log_var = np.log(np.var(wd, axis=2, ddof=1) + eps)

    _mu = wd.mean(axis=2, keepdims=True)
    _ctr = wd - _mu
    _std = np.sqrt((_ctr ** 2).mean(axis=2)) + eps
    with np.errstate(divide='ignore', invalid='ignore'):
        skw = (_ctr ** 3).mean(axis=2) / (_std ** 3)
        krt = (_ctr ** 4).mean(axis=2) / (_std ** 4) - 3.0
        skw = np.nan_to_num(skw, nan=0.0, posinf=0.0, neginf=0.0)
        krt = np.nan_to_num(krt, nan=0.0, posinf=0.0, neginf=0.0)
    del _mu, _ctr, _std

    tkeo = _tkeo(wd)

    pch = [iemg, mav, log_mav, mavs, ssi, rms, log_rms,
           vo, log_det, wl, zcr, ssc, log_var, skw, krt, tkeo]
    pnm = ['IEMG', 'MAV', 'logMAV', 'MAVS', 'SSI', 'RMS', 'logRMS',
           'VO3', 'LogDet', 'WL', 'ZCR', 'SSC', 'logVAR', 'Skew', 'Kurt', 'TKEO']

    if do_hjorth:
        act, mob, cmp = _hjorth(wd)
        pch += [act, mob, cmp]
        pnm += ['HjAct', 'HjMob', 'HjCmp']

    if do_ar:
        ar = _ar_autocorr(wd, ar_order)
        for k in range(ar_order):
            pch.append(ar[:, :, k])
            pnm.append(f'AR{k + 1}')

    if do_freq:
        fft_sz = W
        if fft_pad2:
            fft_sz = 1 << (W - 1).bit_length()
            if fft_sz == W:
                fft_sz *= 2
            wdp = np.pad(wd, ((0, 0), (0, 0), (0, fft_sz - W)))
        else:
            wdp = wd

        mag = np.abs(np.fft.rfft(wdp, axis=2))
        freqs = np.fft.rfftfreq(fft_sz, d=1.0 / fs)
        pw = mag ** 2
        tp = pw.sum(2, keepdims=True) + eps
        fr = freqs.reshape(1, 1, -1)

        with np.errstate(divide='ignore', invalid='ignore'):
            mnf = (fr * pw).sum(2) / tp[:, :, 0]
            mnf = np.nan_to_num(mnf, nan=0.0, posinf=0.0, neginf=0.0)
        cp = np.cumsum(pw, axis=2)
        mdf = freqs[np.argmax(cp >= cp[:, :, -1:] / 2, axis=2)]
        pfq = freqs[np.argmax(pw, axis=2)]
        pn = np.clip(pw / tp, 1e-12, 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            se = -(pn * np.log(pn)).sum(2) / np.log(pn.shape[2])
            se = np.nan_to_num(se, nan=0.0, posinf=0.0, neginf=0.0)

        pch += [mnf, mdf, pfq, se]
        pnm += ['MNF', 'MDF', 'PeakF', 'SpEntropy']

        for flo, fhi in [(20, 150), (150, 350), (350, 450)]:
            mask = (freqs >= flo) & (freqs < fhi)
            pch.append(pw[:, :, mask].sum(2) / tp[:, :, 0])
            pnm.append(f'BP{flo}_{fhi}')

    if do_wavelet and HAS_WAVELETS:
        w_e, w_ent = _wavelet_features_batch(wd, wav_name, wav_level)
        if w_e is not None:
            n_bands = wav_level + 1
            for b in range(n_bands):
                tag = 'cA' if b == 0 else f'cD{n_bands - b}'
                pch += [w_e[:, :, b], w_ent[:, :, b]]
                pnm += [f'WavE_{tag}', f'WavEnt_{tag}']

    pc_stack = np.stack(pch, axis=2)
    flat = pc_stack.reshape(nw, C * pc_stack.shape[2])
    names = [f'ch{c}_{n}' for c in range(C) for n in pnm]

    if do_icc and C > 1:
        icc = _inter_ch_corr(wd, max_channels=corr_channels,
                              corr_stride=corr_stride)
        eff_C = min(corr_channels, C) if corr_channels > 0 else C
        flat = np.hstack([flat, icc])
        names += [
            f'corr_{i}_{j}'
            for i in range(eff_C) for j in range(i + 1, eff_C)
        ]

    # ── v16.0: TKEO band energy features ──────────────────────────
    if do_tkeo_bands:
        tkeo_band = _tkeo_band_features(wd, fs)
        flat = np.hstack([flat, tkeo_band])
        n_tb_bands = 4
        names += [f'ch{c}_TKEO_B{b}' for c in range(C) for b in range(n_tb_bands)]

    # ── v16.0: Covariance features (Riemannian-inspired) ──────────
    if do_cov_features and C > 1:
        cov_feat = _covariance_features(wd)
        flat = np.hstack([flat, cov_feat])
        n_cov = C * (C + 1) // 2
        idx = np.triu_indices(C)
        names += [f'cov_{i}_{j}' for i, j in zip(idx[0], idx[1])]

    return flat, names


# =====================================================================
# Config flag extraction (v14.0 — new adaptive keys)
# =====================================================================

def _pop_processing_flags(p):
    do_freq = p.pop('compute_freq_features', True)
    p.pop('compute_extra_stats', None)
    p.pop('compute_spectral_centroid', None)
    p.pop('compute_spectral_rolloff', False)
    p.pop('use_sliding_window', None)
    ssc_thr = p.pop('ssc_threshold', 0.0)
    fft_pad2 = p.pop('fft_pad_to_power_of_two', True)
    do_ar = p.pop('compute_ar', False)
    ar_order = p.pop('ar_order', 6)
    do_hjorth = p.pop('compute_hjorth', True)
    do_icc = p.pop('compute_inter_channel_corr', True)
    sub_n = p.pop('subsample_every_n', 1)
    p.pop('normalize_signal', None)
    do_wavelet = p.pop('compute_wavelet', False)
    wav_name = p.pop('wavelet_name', 'db4')
    wav_level = p.pop('wavelet_level', 4)
    chunk_size = p.pop('windowing_chunk_size', 2048)
    # pop unknown classification keys
    for key in ['window_label_mode', 'majority_vote_threshold', 'use_numba',
                'filter_cache', 'colsample_bylevel', 'ensemble_enabled',
                'domain_adaptation', 'optuna_enabled',
                'lightgbm_boosting_type', 'lightgbm_num_leaves',
                'lightgbm_min_data_in_leaf', 'ensemble_classifiers',
                'ensemble_voting', 'ensemble_weights',
                'domain_adaptation_method', 'optuna_trials', 'optuna_timeout',
                'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda']:
        p.pop(key, None)

    # v14.0: adaptive processing keys
    corr_channels = p.pop('corr_channels', 0)        # 0 = all channels
    corr_stride = p.pop('corr_stride', 1)             # 1 = full, 2 = half speed
    downsample_large = p.pop('downsample_large', False)  # OFF by default
    downsample_threshold = p.pop('downsample_threshold', 2000000)
    active_signal_detection = p.pop('active_signal_detection', False)
    active_signal_threshold = p.pop('active_signal_threshold', 0.02)

    # v16.0: new feature flags
    euclidean_alignment = p.pop('euclidean_alignment', False)
    do_tkeo_bands = p.pop('compute_tkeo_bands', False)
    do_cov_features = p.pop('compute_covariance_features', False)

    return (do_freq, ssc_thr, fft_pad2, do_ar, ar_order, do_hjorth,
            do_icc, sub_n, do_wavelet, wav_name, wav_level, chunk_size,
            corr_channels, corr_stride,
            downsample_large, downsample_threshold,
            active_signal_detection, active_signal_threshold,
            euclidean_alignment, do_tkeo_bands, do_cov_features)


def _compute_window_samples(config_dict, fs):
    """Compute window size in samples from config.

    v15.0: Priority order:
      1. window_size_ms (time-based, fs-aware) — PREFERRED
      2. window_size (sample-based, legacy fallback)
      3. 400 (absolute default)

    Returns window size in samples (int).
    """
    if 'window_size_ms' in config_dict and config_dict['window_size_ms'] is not None:
        return max(10, int(config_dict['window_size_ms'] * fs / 1000))
    if 'window_size' in config_dict and config_dict['window_size'] is not None:
        return int(config_dict['window_size'])
    return 400


def _build_filter_config(p, fs):
    if 'bandpass' in p:
        lo, hi = p.pop('bandpass')
        p['cutoff_low'], p['cutoff_high'] = lo, hi
    if 'notch' in p:
        p['notch_freq'] = p.pop('notch')
    p.setdefault('sampling_rate', fs)

    # v15.0: window_size_ms (time-based) takes priority over window_size (samples)
    if 'window_size_ms' in p and p['window_size_ms'] is not None:
        window_ms = p.pop('window_size_ms')
        p['window_size'] = max(10, int(window_ms * fs / 1000))
    else:
        # Legacy fallback: window_size in samples
        p.setdefault('window_size', 400)

    p.setdefault('overlap', 0.5)
    p.setdefault('noise_estimation_method', 'percentile')
    p.setdefault('noise_percentile', 5.0)
    p.setdefault('r_threshold', 0.0)
    p.pop('remove_class_zero', None)
    return p


# =====================================================================
# Channel-by-channel filtering (memory-safe)
# =====================================================================

def _filter_emg_channels(emg, sos):
    N, C = emg.shape
    emg_filt = np.empty_like(emg)
    for ch in range(C):
        emg_filt[:, ch] = sosfiltfilt(sos, emg[:, ch])
    return emg_filt


# =====================================================================
# Streaming extraction (for very large datasets)
# =====================================================================

def extract_features_streaming(emg_generator, config_dict, fs, n_channels):
    """Stream-based feature extraction for memory-constrained environments."""
    p = config_dict.copy()
    (do_freq, ssc_thr, fft_pad2, do_ar, ar_order, do_hjorth,
     do_icc, sub_n, do_wavelet, wav_name, wav_level,
     chunk_size, corr_channels, corr_stride,
     _ds_large, _ds_thresh,
     _asd, _asd_thr,
     _ea, do_tkeo_bands, do_cov_features) = _pop_processing_flags(p)

    # v15.0: Compute W from window_size_ms or window_size
    if 'window_size_ms' in p and p['window_size_ms'] is not None:
        W = int(p['window_size_ms'] * fs / 1000)
    elif 'window_size' in p and p['window_size'] is not None:
        W = int(p['window_size'])
    else:
        raise ValueError("window_size or window_size_ms must be in config")

    p2 = _build_filter_config(p, fs)
    cfg = EMGConfig(**p2)
    extractor = EMGFeatureExtractor(cfg)

    all_feat_chunks = []
    feat_names = None
    wins = []
    snr_per_channel = None
    chunk_windows = []
    window_indices = []

    for win_idx, (start, end, raw_window) in enumerate(emg_generator):
        if sub_n > 1 and win_idx % sub_n != 0:
            continue

        window_filt = extractor.preprocess(raw_window)

        # v16.0: Euclidean Alignment applied AFTER filtering, BEFORE windowing
        if _ea and window_filt.shape[0] >= 10 and window_filt.shape[1] >= 2:
            window_filt = euclidean_alignment(window_filt)

        chunk_windows.append(window_filt.T)
        window_indices.append((win_idx, start, end))

        if snr_per_channel is None:
            snr_per_channel = []
            for c in range(n_channels):
                ch = window_filt[:, c]
                rms_val = np.sqrt(np.mean(ch ** 2))
                noise = np.percentile(np.abs(ch), 5) + 1e-12
                snr_per_channel.append(float(20 * np.log10(rms_val / noise)))

        if len(chunk_windows) >= chunk_size:
            wd = np.stack(chunk_windows, axis=0)
            chunk_flat, names = _extract_chunk(
                wd, p2, do_freq, do_ar, ar_order,
                do_hjorth, do_icc, do_wavelet,
                wav_name, wav_level, ssc_thr, fft_pad2, fs, W,
                corr_channels=corr_channels, corr_stride=corr_stride,
                do_tkeo_bands=do_tkeo_bands,
                do_cov_features=do_cov_features,
            )
            all_feat_chunks.append(chunk_flat)
            if feat_names is None:
                feat_names = names
            for _, s, e in window_indices:
                wins.append((s, e))
            chunk_windows = []
            window_indices = []

    if chunk_windows:
        wd = np.stack(chunk_windows, axis=0)
        chunk_flat, names = _extract_chunk(
            wd, p2, do_freq, do_ar, ar_order,
            do_hjorth, do_icc, do_wavelet,
            wav_name, wav_level, ssc_thr, fft_pad2, fs, W,
            corr_channels=corr_channels, corr_stride=corr_stride,
            do_tkeo_bands=do_tkeo_bands,
            do_cov_features=do_cov_features,
        )
        all_feat_chunks.append(chunk_flat)
        if feat_names is None:
            feat_names = names
        for _, s, e in window_indices:
            wins.append((s, e))

    if not all_feat_chunks:
        return np.array([]), [], [], []

    feat_flat = np.vstack(all_feat_chunks).astype(np.float32, copy=False)
    debug_print(f"Streaming: {feat_flat.shape}, {len(wins)} windows")
    return feat_flat, wins, snr_per_channel or [0.0] * n_channels, feat_names


# =====================================================================
# In-memory extraction (v14.0 — Active Signal Detection)
# =====================================================================

def extract_features_per_channel(emg, config_dict):
    """
    Extract features from EMG signal in-memory.

    v14.0 changes:
      - Active Signal Detection (removes rest/noise before windowing)
      - corr_stride for faster correlation on full channels
      - Smart downsampling OFF by default
      - Full correlation channels by default

    v16.0 changes:
      - Euclidean Alignment (domain adaptation after filtering)
      - TKEO band energy features
      - Covariance features (Riemannian-inspired)
    """
    # Step 0: Convert to float32
    if emg.dtype != np.float32:
        emg = np.ascontiguousarray(emg, dtype=np.float32)
        debug_print(f"Converted to float32: {emg.shape}, {emg.nbytes / 1e6:.0f}MB")

    # Extract adaptive processing flags
    original_fs = config_dict.get('sampling_rate', 1000)
    effective_fs = original_fs

    # Pop all flags for this function
    p = config_dict.copy()
    (do_freq, ssc_thr, fft_pad2, do_ar, ar_order, do_hjorth,
     do_icc, sub_n, do_wavelet, wav_name, wav_level,
     chunk_size, corr_channels, corr_stride,
     downsample_large, downsample_threshold,
     use_asd, asd_threshold,
     use_ea, do_tkeo_bands, do_cov_features) = _pop_processing_flags(p)

    # v14.0: Smart downsampling (OFF by default, only if explicitly set)
    bandpass = config_dict.get('bandpass', [20, 450])
    bandpass_high = bandpass[1] if isinstance(bandpass, (list, tuple)) else 450
    min_fs_for_bandpass = 2 * bandpass_high

    if (downsample_large and emg.shape[0] > downsample_threshold
            and (original_fs // 2) >= min_fs_for_bandpass):
        old_N = emg.shape[0]
        emg = emg[::2]
        effective_fs = original_fs // 2
        debug_print(
            f"[v14.0] Smart downsample: {old_N} -> {emg.shape[0]} samples, "
            f"fs: {original_fs} -> {effective_fs}Hz"
        )
        config_dict = dict(config_dict)
        config_dict['sampling_rate'] = effective_fs
    elif downsample_large and emg.shape[0] > downsample_threshold:
        debug_print(
            f"[v14.0] Skipping downsample: {original_fs}Hz // 2 = {original_fs // 2}Hz "
            f"< 2 * bandpass_high ({min_fs_for_bandpass}Hz) — would cause aliasing"
        )

    # v15.0: Active Signal Detection (before filtering)
    # CRITICAL FIX: Skip ASD when remove_class_zero=False
    # because low-energy samples ARE the rest class we want to preserve.
    if use_asd and emg.shape[0] > 1000:
        remove_class_zero = config_dict.get('remove_class_zero', False)
        if remove_class_zero:
            emg, active_mask = active_signal_detection(emg, threshold=asd_threshold)
            debug_print(f"[v15.0] Active signal detection: {emg.shape[0]} samples retained")
        else:
            debug_print("[v15.0] Active signal detection SKIPPED "
                         "(remove_class_zero=False — preserving rest class)")

    # Memory threshold for streaming
    IN_MEMORY_THRESHOLD = 300 * 1024 * 1024  # 300MB

    if emg.nbytes > IN_MEMORY_THRESHOLD:
        debug_print(
            f"Large EMG ({emg.nbytes / 1e6:.0f}MB > {IN_MEMORY_THRESHOLD / 1e6:.0f}MB) "
            f"-> streaming mode."
        )
        N, C = emg.shape
        fs = effective_fs
        # v15.0: Compute W from window_size_ms (not hardcoded window_size)
        W = _compute_window_samples(config_dict, fs)
        step = max(1, int(W * (1 - config_dict.get('overlap', 0.5))))

        def window_generator():
            for start in range(0, N - W + 1, step):
                yield start, start + W, emg[start:start + W, :]

        return extract_features_streaming(
            window_generator(), config_dict, fs, C
        )

    debug_print("Feature extraction v16.0 (in-memory, stride-trick, float32) ...")
    try:
        p2 = _build_filter_config(p, effective_fs)

        cfg = EMGConfig(**p2)
        fs = cfg.sampling_rate

        sos = butter(
            4, [cfg.cutoff_low, cfg.cutoff_high],
            btype='band', output='sos', fs=fs
        )

        gc.collect()

        try:
            emg_filt = sosfiltfilt(sos, emg, axis=0)
        except MemoryError:
            debug_print(
                "[mem] Full-array filtering failed (MemoryError), "
                "retrying channel-by-channel..."
            )
            emg_filt = _filter_emg_channels(emg, sos)

        debug_print(f"Filtered: {emg_filt.shape}")

        # v16.0: Euclidean Alignment (domain adaptation — He & Wu 2020)
        # Applied AFTER filtering, BEFORE windowing
        if use_ea:
            emg_filt = euclidean_alignment(emg_filt)
            debug_print(f"[v16.0] Euclidean Alignment applied: {emg_filt.shape}")

        N, C = emg_filt.shape
        W = cfg.window_size
        step = max(1, int(W * (1 - cfg.overlap)))
        n_all = max(1, (N - W) // step + 1)
        wins = [(i * step, i * step + W) for i in range(n_all)]

        if sub_n > 1:
            wins = wins[::sub_n]

        nw = len(wins)
        debug_print(f"Windows: {nw} (chunk={chunk_size}, overlap={cfg.overlap:.2f})")

        # ── STRIDE-TRICK WINDOWING (zero-copy window view) ────────────
        shape = (n_all, W, C)
        strides = (emg_filt.strides[0] * step, emg_filt.strides[0], emg_filt.strides[1])
        all_windows = np.lib.stride_tricks.as_strided(
            emg_filt, shape=shape, strides=strides
        )
        # v35.0: Materialize a copy so del emg_filt actually frees memory
        # stride_tricks creates a VIEW that holds a reference to emg_filt
        all_windows = all_windows.copy()
        if sub_n > 1:
            all_windows = all_windows[::sub_n]

        del emg_filt
        gc.collect()

        all_feat_chunks = []
        feat_names = None

        for chunk_start in range(0, nw, chunk_size):
            chunk_end = min(chunk_start + chunk_size, nw)

            try:
                wd = np.ascontiguousarray(
                    all_windows[chunk_start:chunk_end].transpose(0, 2, 1)
                )
                chunk_flat, names = _extract_chunk(
                    wd, p2, do_freq, do_ar, ar_order,
                    do_hjorth, do_icc, do_wavelet,
                    wav_name, wav_level, ssc_thr, fft_pad2, fs, W,
                    corr_channels=corr_channels,
                    corr_stride=corr_stride,
                    do_tkeo_bands=do_tkeo_bands,
                    do_cov_features=do_cov_features,
                )
                del wd
                all_feat_chunks.append(chunk_flat)
                if feat_names is None:
                    feat_names = names
            except MemoryError:
                debug_print(
                    f"[mem] MemoryError at chunk {chunk_start}-{chunk_end}, "
                    f"retrying with tiny chunks + gc"
                )
                sub_chunk_size = max(32, (chunk_end - chunk_start) // 4)
                for sub_start in range(chunk_start, chunk_end, sub_chunk_size):
                    sub_end = min(sub_start + sub_chunk_size, chunk_end)
                    try:
                        wd = np.ascontiguousarray(
                            all_windows[sub_start:sub_end].transpose(0, 2, 1)
                        )
                        chunk_flat, names = _extract_chunk(
                            wd, p2, do_freq, do_ar, ar_order,
                            do_hjorth, do_icc, do_wavelet,
                            wav_name, wav_level, ssc_thr, fft_pad2, fs, W,
                            corr_channels=corr_channels,
                            corr_stride=corr_stride,
                            do_tkeo_bands=do_tkeo_bands,
                            do_cov_features=do_cov_features,
                        )
                        del wd
                        all_feat_chunks.append(chunk_flat)
                        if feat_names is None:
                            feat_names = names
                    except MemoryError:
                        debug_print(f"[mem] Tiny chunk also failed at {sub_start}")
                        raise

        feat_flat = np.vstack(all_feat_chunks).astype(np.float32, copy=False)
        debug_print(f"Feature matrix: {feat_flat.shape}")

        snr = [0.0] * C

        return feat_flat, wins, snr, feat_names

    except MemoryError as me:
        debug_print(
            f"[mem] In-memory path failed completely ({me}), "
            f"falling back to streaming mode..."
        )
        gc.collect()

        N, C = emg.shape
        fs = config_dict.get('sampling_rate', 1000)
        # v15.0: Compute W from window_size_ms (not hardcoded window_size)
        W = _compute_window_samples(config_dict, fs)
        step = max(1, int(W * (1 - config_dict.get('overlap', 0.5))))

        def window_generator():
            for start in range(0, N - W + 1, step):
                yield start, start + W, emg[start:start + W, :]

        return extract_features_streaming(
            window_generator(), config_dict, fs, C
        )

    except Exception as exc:
        import traceback
        debug_print(f"!!! {exc}")
        traceback.print_exc(file=sys.stderr)
        raise


# =====================================================================
# Feature dimension verification
# =====================================================================

def verify_feature_dimension(C, ar_order=0, do_hjorth=True, do_freq=True,
                              do_wavelet=False, wav_level=4,
                              do_icc=True, corr_channels=0,
                              do_tkeo_bands=False, do_cov_features=False):
    """v16.0: Full correlation by default, plus TKEO bands and covariance features."""
    base_per_channel = 16
    if do_hjorth:
        base_per_channel += 3
    if do_freq:
        base_per_channel += 7
    if ar_order > 0:
        base_per_channel += ar_order
    if do_wavelet:
        base_per_channel += 2 * (wav_level + 1)

    if do_icc and C > 1:
        eff_C = min(corr_channels, C) if corr_channels > 0 else C
        icc_count = (eff_C * (eff_C - 1)) // 2
    else:
        eff_C = C
        icc_count = 0

    # v16.0: TKEO band features add 4 features per channel
    tkeo_band_count = C * 4 if do_tkeo_bands else 0

    # v16.0: Covariance features add C*(C+1)/2 features
    cov_count = C * (C + 1) // 2 if (do_cov_features and C > 1) else 0

    total = base_per_channel * C + icc_count + tkeo_band_count + cov_count
    extra = []
    if tkeo_band_count:
        extra.append(f"TKEO_bands={tkeo_band_count}")
    if cov_count:
        extra.append(f"Cov={cov_count}")
    extra_str = f", {', '.join(extra)}" if extra else ""
    corr_ch_str = str(eff_C) if (do_icc and C > 1) else ('off' if not do_icc else str(C))
    return total, f"{total}D (C={C}, per_ch={base_per_channel}, ICC={icc_count}, corr_ch={corr_ch_str}{extra_str})"
