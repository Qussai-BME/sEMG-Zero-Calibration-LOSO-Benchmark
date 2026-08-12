#!/usr/bin/env python3
"""
core_engine.py - EMG Signal Processing Core (FINAL FIX)
Always returns 2D arrays, even for single channel.
"""

import numpy as np
import logging
from scipy import signal
from scipy.fft import fft, fftfreq
import json
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


@dataclass
class EMGConfig:
    """Configuration parameters following biomedical standards."""
     # ... existing fields ...
    r_threshold: float = 0.0   # for ZCR and SSC threshold
    sampling_rate: int = 2000
    cutoff_low: float = 20.0
    cutoff_high: float = 450.0
    filter_order: int = 4
    notch_freq: float = 50.0
    window_size: int = 200
    overlap: float = 0.5
    filter_type: str = 'butterworth'
    noise_estimation_method: str = 'percentile'
    noise_percentile: float = 5.0
    manual_noise_floor: Optional[float] = None
    psd_method: str = 'welch'
    chunk_duration: Optional[float] = None

    def validate(self):
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if self.cutoff_high >= self.sampling_rate / 2:
            raise ValueError("Cutoff high must be < Nyquist frequency")
        if self.filter_order < 1 or self.filter_order > 10:
            raise ValueError("Filter order must be 1-10")
        if not (0 < self.overlap < 1):
            raise ValueError("Overlap must be between 0 and 1")
        return True


class EMGFeatureExtractor:

    def __init__(self, config: EMGConfig):
        self.config = config
        self.config.validate()
        self.filters = self._design_filters()
        logger.info(f"EMG Engine initialized with config: {config}")

    def _design_filters(self) -> Dict:
        nyquist = self.config.sampling_rate / 2
        Wn = [self.config.cutoff_low / nyquist, self.config.cutoff_high / nyquist]

        try:
            if self.config.filter_type == 'butterworth':
                b_band, a_band = signal.butter(self.config.filter_order, Wn, btype='band')
            elif self.config.filter_type == 'chebyshev':
                b_band, a_band = signal.cheby1(self.config.filter_order, 0.5, Wn, btype='band')
            elif self.config.filter_type == 'bessel':
                b_band, a_band = signal.bessel(self.config.filter_order, Wn, btype='band')
            elif self.config.filter_type == 'elliptic':
                b_band, a_band = signal.ellip(self.config.filter_order, 0.5, 40, Wn, btype='band')
            else:
                logger.warning(f"Unknown filter type '{self.config.filter_type}', using Butterworth.")
                b_band, a_band = signal.butter(self.config.filter_order, Wn, btype='band')
        except Exception as e:
            logger.error(f"Filter design failed: {e}")
            raise ValueError(f"Could not design {self.config.filter_type} filter: {e}")

        b_notch, a_notch = signal.iirnotch(self.config.notch_freq / nyquist, 30.0)
        return {'bandpass': (b_band, a_band), 'notch': (b_notch, a_notch)}

    def _estimate_noise_floor(self, signal_segment: np.ndarray) -> float:
        """
        signal_segment: 1D array.
        """
        try:
            if self.config.noise_estimation_method == 'manual' and self.config.manual_noise_floor is not None:
                return float(self.config.manual_noise_floor)

            window_len = int(0.2 * self.config.sampling_rate)
            if window_len < 10:
                window_len = min(10, len(signal_segment) // 4)
            if len(signal_segment) < window_len:
                return float(np.std(signal_segment))

            rms_vals = []
            for i in range(0, len(signal_segment) - window_len, window_len // 2):
                seg = signal_segment[i:i+window_len]
                rms_vals.append(np.sqrt(np.mean(seg**2)))

            if len(rms_vals) == 0:
                return float(np.std(signal_segment))

            if self.config.noise_estimation_method == 'median':
                noise_floor = np.median(rms_vals)
            else:
                noise_floor = np.percentile(rms_vals, self.config.noise_percentile)

            return float(noise_floor)
        except Exception as e:
            logger.error(f"Noise floor estimation failed: {e}")
            return 0.01

    def preprocess(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Always returns a 2D array of shape (samples, channels).
        """
        try:
            # Convert to 2D if needed
            if raw_signal.ndim == 1:
                raw_signal = raw_signal.reshape(-1, 1)
            elif raw_signal.ndim != 2:
                raise ValueError(f"Expected 1D or 2D array, got {raw_signal.ndim}D")

            n_samples, n_channels = raw_signal.shape

            if n_samples < self.config.window_size:
                raise ValueError(f"Signal length {n_samples} < window size {self.config.window_size}")

            b_band, a_band = self.filters['bandpass']
            b_notch, a_notch = self.filters['notch']

            filtered = np.zeros_like(raw_signal, dtype=np.float64)
            for ch in range(n_channels):
                f = signal.filtfilt(b_band, a_band, raw_signal[:, ch])
                f = signal.filtfilt(b_notch, a_notch, f)
                filtered[:, ch] = f

            # Always return 2D – even for single channel
            return filtered  # shape (n_samples, n_channels)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def extract_time_features(self, signal_segment: np.ndarray) -> Dict[str, float]:
        """
        signal_segment: 1D array (window,)
        """
        try:
            seg = signal_segment.astype(np.float64)
            features = {}
            features['MAV'] = float(np.mean(np.abs(seg)))
            features['RMS'] = float(np.sqrt(np.mean(seg**2)))
            zero_crossings = np.where(np.diff(np.signbit(seg)))[0]
            features['ZCR'] = float(len(zero_crossings) / len(seg))
            features['WL'] = float(np.sum(np.abs(np.diff(seg))))
            slopes = np.diff(seg)
            ssc = np.sum((slopes[:-1] * slopes[1:]) < 0)
            features['SSC'] = float(ssc / len(seg))
            return features
        except Exception as e:
            logger.error(f"Time feature extraction failed: {e}")
            raise

    def extract_frequency_features(self, signal_segment: np.ndarray, fs: int) -> Dict[str, float]:
        """
        signal_segment: 1D array (window,)
        """
        try:
            if self.config.psd_method == 'fft':
                n = len(signal_segment)
                fft_vals = fft(signal_segment)
                fft_abs = np.abs(fft_vals[:n//2])
                freqs = fftfreq(n, 1/fs)[:n//2]
                psd = fft_abs**2 / n
            else:
                freqs, psd = signal.welch(signal_segment, fs, nperseg=min(256, len(signal_segment)))

            total_power = np.sum(psd)
            if total_power == 0:
                return {'MDF': 0.0, 'MNF': 0.0}
            mnf = np.sum(freqs * psd) / total_power
            cum_power = np.cumsum(psd)
            mdf = freqs[np.searchsorted(cum_power, cum_power[-1] / 2)]
            return {'MDF': float(mdf), 'MNF': float(mnf)}
        except Exception as e:
            logger.error(f"Frequency feature extraction failed: {e}")
            return {'MDF': 0.0, 'MNF': 0.0}

    def estimate_memory_usage(self, n_samples: int, n_channels: int) -> float:
        raw_mb = n_samples * n_channels * 8 / (1024**2)
        filt_mb = raw_mb
        n_windows = n_samples // self.config.window_size
        feat_mb = n_windows * n_channels * 5 * 8 / (1024**2)
        return raw_mb + filt_mb + feat_mb

    def process_stream(self,
                       raw_signal: np.ndarray,
                       selected_channel: int = 0,
                       measure_time: bool = False,
                       compute_freq_features: bool = False) -> Dict:
        start_time = time.perf_counter() if measure_time else None

        try:
            # Ensure 2D
            if raw_signal.ndim == 1:
                raw_signal = raw_signal.reshape(-1, 1)
            elif raw_signal.ndim != 2:
                raise ValueError(f"raw_signal must be 1D or 2D, got {raw_signal.ndim}D")

            n_samples, n_channels = raw_signal.shape

            if HAS_PSUTIL:
                mem_est = self.estimate_memory_usage(n_samples, n_channels)
                avail = psutil.virtual_memory().available / (1024**2)
                if mem_est > 0.5 * avail:
                    logger.warning(f"Estimated memory {mem_est:.1f} MB may exceed available {avail:.1f} MB. Consider chunking.")

            # Chunking
            if self.config.chunk_duration is not None:
                chunk_samples = int(self.config.chunk_duration * self.config.sampling_rate)
                if chunk_samples < self.config.window_size:
                    chunk_samples = self.config.window_size
                n_chunks = int(np.ceil(n_samples / chunk_samples))
                logger.info(f"Processing in {n_chunks} chunks of ~{chunk_samples} samples")

                all_time_features = [[] for _ in range(n_channels)]
                all_freq_features = [[] for _ in range(n_channels)] if compute_freq_features else None
                all_timestamps = []

                for i in range(0, n_samples, chunk_samples):
                    chunk = raw_signal[i:i+chunk_samples, :]
                    old_chunk = self.config.chunk_duration
                    self.config.chunk_duration = None
                    chunk_res = self.process_stream(
                        chunk,
                        selected_channel=selected_channel,
                        measure_time=False,
                        compute_freq_features=compute_freq_features
                    )
                    self.config.chunk_duration = old_chunk

                    for ch in range(n_channels):
                        all_time_features[ch].extend(chunk_res['time_series']['features'][ch])
                        if compute_freq_features:
                            all_freq_features[ch].extend(chunk_res['time_series']['freq_features'][ch])
                    offset = i / self.config.sampling_rate
                    all_timestamps.extend([t + offset for t in chunk_res['time_series']['timestamps']])

                # Get filtered full signal (2D)
                filtered_full = self.preprocess(raw_signal)  # shape (n_samples, n_channels)
                # Select channel data
                if n_channels == 1:
                    ch_data = filtered_full[:, 0]  # now safe because filtered_full is 2D
                else:
                    ch_data = filtered_full[:, selected_channel]
                noise_floor = self._estimate_noise_floor(ch_data)

                sig_std = np.std(ch_data)
                if noise_floor > 0:
                    snr_db = 20 * np.log10(sig_std / noise_floor)
                else:
                    snr_db = 0.0

                output = {
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'sampling_rate': self.config.sampling_rate,
                        'window_size': self.config.window_size,
                        'overlap': self.config.overlap,
                        'filter_config': asdict(self.config),
                        'n_channels': n_channels,
                        'selected_channel': selected_channel,
                        'processed_in_chunks': True,
                        'chunk_duration': self.config.chunk_duration
                    },
                    'signal_quality': {
                        'estimated_noise_floor': float(noise_floor),
                        'mean_snr': float(snr_db),
                        'artifact_detected': False
                    },
                    'time_series': {
                        'timestamps': all_timestamps,
                        'features': all_time_features
                    },
                    'summary_statistics': {}
                }
                if compute_freq_features:
                    output['time_series']['freq_features'] = all_freq_features

                for ch in range(n_channels):
                    mav_vals = [f['MAV'] for f in all_time_features[ch]]
                    rms_vals = [f['RMS'] for f in all_time_features[ch]]
                    output['summary_statistics'][f'channel_{ch}'] = {
                        'mean_activation': float(np.mean(mav_vals)) if mav_vals else 0.0,
                        'peak_activation': float(np.max(rms_vals)) if rms_vals else 0.0,
                        'fatigue_index': 0.0
                    }

                if measure_time:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    output['benchmark'] = {'processing_time_ms': elapsed_ms}

                return output

            else:
                # No chunking
                filtered = self.preprocess(raw_signal)  # shape (n_samples, n_channels)
                # Select channel data
                if n_channels == 1:
                    ch_data = filtered[:, 0]
                else:
                    ch_data = filtered[:, selected_channel]
                noise_floor = self._estimate_noise_floor(ch_data)

                step = int(self.config.window_size * (1 - self.config.overlap))
                if step < 1:
                    step = 1
                n_windows = max(1, (len(ch_data) - self.config.window_size) // step + 1)

                time_features_all = []
                freq_features_all = [] if compute_freq_features else None

                for ch in range(n_channels):
                    ch_time_feats = []
                    ch_freq_feats = []
                    for i in range(n_windows):
                        start = i * step
                        end = start + self.config.window_size
                        if end > len(filtered):
                            break
                        window = filtered[start:end, ch]  # 1D slice
                        tf = self.extract_time_features(window)
                        ch_time_feats.append(tf)
                        if compute_freq_features:
                            ff = self.extract_frequency_features(window, self.config.sampling_rate)
                            ch_freq_feats.append(ff)
                    time_features_all.append(ch_time_feats)
                    if compute_freq_features:
                        freq_features_all.append(ch_freq_feats)

                timestamps = [i * step / self.config.sampling_rate for i in range(len(time_features_all[0]))]

                sig_std = np.std(ch_data)
                if noise_floor > 0:
                    snr_db = 20 * np.log10(sig_std / noise_floor)
                else:
                    snr_db = 0.0

                output = {
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'sampling_rate': self.config.sampling_rate,
                        'window_size': self.config.window_size,
                        'overlap': self.config.overlap,
                        'filter_config': asdict(self.config),
                        'n_channels': n_channels,
                        'selected_channel': selected_channel,
                        'processed_in_chunks': False
                    },
                    'signal_quality': {
                        'estimated_noise_floor': float(noise_floor),
                        'mean_snr': float(snr_db),
                        'artifact_detected': False
                    },
                    'time_series': {
                        'timestamps': timestamps,
                        'features': time_features_all
                    },
                    'summary_statistics': {}
                }
                if compute_freq_features:
                    output['time_series']['freq_features'] = freq_features_all

                for ch in range(n_channels):
                    mav_vals = [f['MAV'] for f in time_features_all[ch]]
                    rms_vals = [f['RMS'] for f in time_features_all[ch]]
                    output['summary_statistics'][f'channel_{ch}'] = {
                        'mean_activation': float(np.mean(mav_vals)) if mav_vals else 0.0,
                        'peak_activation': float(np.max(rms_vals)) if rms_vals else 0.0,
                        'fatigue_index': 0.0
                    }

                if measure_time:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    output['benchmark'] = {'processing_time_ms': elapsed_ms}

                return output

        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            raise


class EMGSignalSimulator:

    @staticmethod
    def generate_contraction(duration: float,
                             sampling_rate: int,
                             intensity: float = 1.0,
                             n_channels: int = 1) -> np.ndarray:
        n_samples = int(duration * sampling_rate)
        t = np.linspace(0, duration, n_samples)
        envelope = np.ones(n_samples) * intensity
        envelope *= 0.8 + 0.2 * np.sin(2 * np.pi * 2 * t)

        noise = np.random.normal(0, 0.1 * intensity, (n_samples, n_channels))
        signal = envelope[:, None] * noise
        signal += np.random.normal(0, 0.01, (n_samples, n_channels))

        # Always return 2D if n_channels > 1, else 1D for backward compatibility
        if n_channels == 1:
            return signal.ravel()
        else:
            return signal