#!/usr/bin/env python3
"""
diagnose_v2.py — Comprehensive Diagnostic & Baseline Fix for MiniROCKET on EMG
==============================================================================
Based on literature analysis (2024-2025 SOTA on NinaPro DB2):

KEY FINDINGS from papers:
  1. Ovadia et al. 2024 (Nature Sci. Rep.): MiniROCKET on NinaPro DB5/DB7,
     intra-subject 4-1-1 split = 98.27%. BUT they used FULL MOVEMENT (5s),
     not 200ms windows. Also removed rest (99% accuracy for rest detection).
  
  2. Yang et al. 2025 (Nature Sci. Rep.): Cross-subject LOSO on DB2 = 89.4%
     using SSL + Adversarial DA (deep learning).
  
  3. Colot et al. 2024 (arXiv): Cross-subject LOSO, 4 classes = 79.5%
     with KPCA Subspace Alignment.
  
  4. Wu et al. 2025 (Int. J. Neural Sys.): EMG-ROCKET on HD-sEMG,
     cross-day = 84.3% with enhanced channel fusion.

DIAGNOSIS:
  - Our within-subject (40.56%) is LOW because:
    a) Data amplitude ~0.001 mV (very small) → PPV features near random
    b) 200-400ms windows may be too short for 40-49 classes
    c) Exercise offset mapping may be loading wrong data
  
  - Our cross-subject (2.3%) is below random because:
    a) PPV features encode subject identity, not movement patterns
    b) No domain adaptation

THIS SCRIPT:
  Phase 1: Data Quality Check (raw signal inspection per exercise)
  Phase 2: Window Size Sweep (200ms → 2000ms)
  Phase 3: Exercise Selection (E1 vs E1+E2 vs all)
  Phase 4: Full Movement Segmentation (Ovadia approach)
  
Usage:
  python diagnose_v2.py --config config.yaml --datasets ninapro_db2 --subjects 1
  python diagnose_v2.py --config config.yaml --datasets ninapro_db3 --subjects 1 2
"""

import sys
import os
import gc
import time
import argparse
import numpy as np
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import yaml

from minirocket import MiniRocketPipeline


# =====================================================================
#  Args & Config
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Diagnose MiniROCKET on EMG data")
    p.add_argument('--config', type=str, default=os.path.join(SCRIPT_DIR, 'config.yaml'))
    p.add_argument('--datasets', nargs='+', required=True,
                   choices=['ninapro_db2', 'ninapro_db3', 'ninapro_db7'])
    p.add_argument('--subjects', nargs='+', type=int, default=[1])
    p.add_argument('--num_kernels', type=int, default=2000)
    p.add_argument('--max_train', type=int, default=15000)
    p.add_argument('--phase', type=str, default='all',
                   choices=['all', 'data', 'window', 'exercise', 'full_movement'])
    p.add_argument('--no_movement_map', action='store_true',
                   help='Skip db3_to_db7 movement_map (standalone DB3 analysis, keeps Exercise D)')
    return p.parse_args()


def load_config(path):
    for candidate in [path, os.path.join(SCRIPT_DIR, 'config.yaml')]:
        if os.path.exists(candidate):
            with open(candidate, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Config not found: {path}")


# =====================================================================
#  Windowing Utilities
# =====================================================================
def create_windows(emg, window_size, overlap=0.5):
    """Create sliding windows from raw EMG. Returns (N_win, C, T_win)."""
    N, C = emg.shape
    step = max(1, int(window_size * (1.0 - overlap)))
    n_windows = (N - window_size) // step + 1
    if n_windows <= 0:
        return np.empty((0, C, window_size), dtype=np.float32)
    strides = (emg.strides[0] * step, emg.strides[1], emg.strides[0])
    windows = np.lib.stride_tricks.as_strided(
        emg, shape=(n_windows, C, window_size), strides=strides
    )
    return np.ascontiguousarray(windows, dtype=np.float32)


def assign_window_labels(windows, labels, overlap=0.5):
    """Assign label to each window based on center sample."""
    n_windows = windows.shape[0]
    window_size = windows.shape[2]
    step = max(1, int(window_size * (1.0 - overlap)))
    mids = np.arange(n_windows) * step + window_size // 2
    mids = np.clip(mids, 0, len(labels) - 1)
    return labels[mids]


def segment_movements(emg, labels, min_duration=500):
    """
    Segment continuous EMG into individual movement segments.
    Each segment starts when label changes from 0 to non-zero,
    and ends when it changes back to 0.
    
    Returns list of (segment_emg, label) tuples.
    """
    N, C = emg.shape
    segments = []
    seg_start = None
    seg_label = 0
    
    for i in range(N):
        lbl = labels[i]
        if lbl != 0 and seg_start is None:
            # Movement starts
            seg_start = i
            seg_label = lbl
        elif lbl == 0 and seg_start is not None:
            # Movement ends
            duration = i - seg_start
            if duration >= min_duration:
                segments.append((emg[seg_start:i].copy(), seg_label))
            seg_start = None
            seg_label = 0
        elif lbl != 0 and lbl != seg_label:
            # Label changed (different movement without rest gap)
            duration = i - seg_start
            if duration >= min_duration:
                segments.append((emg[seg_start:i].copy(), seg_label))
            seg_start = i
            seg_label = lbl
    
    # Handle last segment
    if seg_start is not None:
        duration = N - seg_start
        if duration >= min_duration:
            segments.append((emg[seg_start:N].copy(), seg_label))
    
    return segments


def pad_or_trim_segment(seg, target_length):
    """Pad with zeros or trim to reach target length."""
    C = seg.shape[1]
    current_len = seg.shape[0]
    if current_len == target_length:
        return seg
    elif current_len < target_length:
        pad = np.zeros((target_length - current_len, C), dtype=np.float32)
        return np.vstack([seg, pad])
    else:
        # Take center portion
        start = (current_len - target_length) // 2
        return seg[start:start + target_length]


# =====================================================================
#  Phase 1: Data Quality Check
# =====================================================================
def phase1_data_quality(raw_data, label_data, fs, ds_name):
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 1: DATA QUALITY CHECK", flush=True)
    print("=" * 70, flush=True)
    
    for idx, ((subj_id, emg), (_, labels)) in enumerate(zip(raw_data, label_data)):
        # v8.4: Safety alignment check (last resort — data_loaders should handle this)
        if emg.shape[0] != labels.shape[0]:
            min_len = min(emg.shape[0], labels.shape[0])
            print(f"    [v8.4 WARN] EMG/labels length mismatch: {emg.shape[0]} vs {labels.shape[0]}. "
                  f"Truncating to {min_len}.", flush=True)
            emg = emg[:min_len]
            labels = labels[:min_len]
            # Propagate to source arrays so later phases also use truncated data
            raw_data[idx] = (subj_id, emg)
            label_data[idx] = (subj_id, labels)

        print(f"\n  Subject {subj_id}:", flush=True)
        print(f"    Raw shape: {emg.shape} (samples x channels)", flush=True)
        print(f"    Duration: {emg.shape[0]/fs:.1f}s @ {fs}Hz", flush=True)
        print(f"    Channels: {emg.shape[1]}", flush=True)
        print(f"    EMG stats: mean={emg.mean():.6f}, std={emg.std():.6f}", flush=True)
        print(f"    EMG range: [{emg.min():.6f}, {emg.max():.6f}]", flush=True)
        print(f"    EMG abs-max: {np.abs(emg).max():.6f}", flush=True)
        
        # Per-channel stats
        for ch in range(min(emg.shape[1], 12)):
            ch_data = emg[:, ch]
            print(f"    CH{ch:2d}: mean={ch_data.mean():+.6f}, "
                  f"std={ch_data.std():.6f}, "
                  f"range=[{ch_data.min():+.6f}, {ch_data.max():+.6f}]",
                  flush=True)
        
        # Label distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"    Labels: {len(unique)} unique values", flush=True)
        for lbl, cnt in sorted(zip(unique, counts)):
            duration_s = cnt / fs
            pct = 100.0 * cnt / len(labels)
            if lbl == 0:
                print(f"      Label {lbl:3d} (REST): {cnt:8d} samples "
                      f"({duration_s:.1f}s, {pct:.1f}%)", flush=True)
            else:
                print(f"      Label {lbl:3d} (MOV):  {cnt:8d} samples "
                      f"({duration_s:.1f}s, {pct:.1f}%)", flush=True)
        
        # Check if data is predominantly rest
        rest_pct = 100.0 * np.sum(labels == 0) / len(labels)
        print(f"    Rest percentage: {rest_pct:.1f}%", flush=True)
        
        # Movement amplitude (excluding rest)
        mov_mask = labels != 0
        if mov_mask.sum() > 0:
            mov_emg = emg[mov_mask]
            print(f"    Movement EMG (excl rest): mean={mov_emg.mean():.6f}, "
                  f"std={mov_emg.std():.6f}", flush=True)
            print(f"    Movement EMG range: [{mov_emg.min():.6f}, {mov_emg.max():.6f}]",
                  flush=True)
            print(f"    Movement abs-max: {np.abs(mov_emg).max():.6f}", flush=True)
        
        # Per-exercise analysis
        print(f"    Per-exercise label ranges:", flush=True)
        # DB2: E1 (B) = labels 1-17, E2 (C) = 18-40, E3 (D) = 41-49
        if ds_name in ['ninapro_db2', 'ninapro_db3']:
            exercises = [
                ('E1/B', 1, 17), ('E2/C', 18, 40), ('E3/D', 41, 49)
            ]
            for ex_name, lo, hi in exercises:
                mask = (labels >= lo) & (labels <= hi)
                cnt = mask.sum()
                if cnt > 0:
                    ex_emg = emg[mask]
                    print(f"      {ex_name} (labels {lo}-{hi}): {cnt:8d} samples, "
                          f"std={ex_emg.std():.6f}, "
                          f"absmax={np.abs(ex_emg).max():.6f}", flush=True)
                else:
                    print(f"      {ex_name} (labels {lo}-{hi}): NO DATA", flush=True)
        elif ds_name == 'ninapro_db7':
            non_rest = labels[labels != 0]
            if len(non_rest) > 0:
                print(f"      Movements: labels {non_rest.min()}-{non_rest.max()} "
                      f"({len(np.unique(non_rest))} unique)", flush=True)


# =====================================================================
#  Phase 2: Window Size Sweep
# =====================================================================
def phase2_window_sweep(emg, labels, fs, num_kernels, max_train, overlap=0.75):
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 2: WINDOW SIZE SWEEP (with rest removed)", flush=True)
    print("=" * 70, flush=True)
    
    results = []
    window_ms_list = [100, 200, 400, 600, 800, 1000, 1500, 2000]
    
    for window_ms in window_ms_list:
        window_size = int(window_ms * fs / 1000)
        if window_size < 9:
            continue
        
        windows = create_windows(emg, window_size, overlap=overlap)
        win_labels = assign_window_labels(windows, labels, overlap=overlap)
        
        # Remove rest
        mask = win_labels != 0
        if mask.sum() == 0:
            print(f"  {window_ms:5d}ms: No movement windows!", flush=True)
            continue
        windows = windows[mask]
        win_labels = win_labels[mask]
        
        n_classes = len(np.unique(win_labels))
        n_windows = len(win_labels)
        random_chance = 1.0 / n_classes
        
        if n_windows < 100:
            print(f"  {window_ms:5d}ms: Only {n_windows} windows, skipping", flush=True)
            continue
        
        # Shuffle and split 80/20
        rng = np.random.RandomState(42)
        idx = np.arange(n_windows)
        rng.shuffle(idx)
        split = int(0.8 * len(idx))
        train_idx, test_idx = idx[:split], idx[split:]
        
        X_train = windows[train_idx]
        y_train = win_labels[train_idx]
        X_test = windows[test_idx]
        y_test = win_labels[test_idx]
        
        # Subsample if needed
        if len(y_train) > max_train:
            total = len(y_train)
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            indices = []
            for cls, count in zip(unique_classes, class_counts):
                cls_idx = np.where(y_train == cls)[0]
                n_sample = min(count, max(2, int(count * max_train / total)))
                chosen = rng.choice(cls_idx, size=n_sample, replace=False)
                indices.append(chosen)
            indices = np.concatenate(indices)
            rng.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        print(f"\n  --- Window: {window_ms}ms ({window_size} samples) ---", flush=True)
        print(f"      Classes: {n_classes}, Random: {random_chance*100:.1f}%", flush=True)
        print(f"      Train: {len(y_train):,}, Test: {len(y_test):,}", flush=True)
        
        t0 = time.time()
        try:
            pipe = MiniRocketPipeline(num_kernels=num_kernels)
            pipe.fit(X_train, y_train)
            train_acc = pipe.score(X_train, y_train)
            test_acc = pipe.score(X_test, y_test)
            dt = time.time() - t0
            
            ratio = test_acc / random_chance
            status = "OK" if test_acc > 0.5 else ("LOW" if test_acc > random_chance else "FAIL")
            
            print(f"      Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}% "
                  f"({ratio:.1f}x random) [{status}] ({dt:.1f}s)", flush=True)
            
            results.append({
                'window_ms': window_ms,
                'n_classes': n_classes,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'train_acc': train_acc,
                'test_acc': test_acc,
                'ratio': ratio,
                'time': dt,
            })
            
        except Exception as e:
            print(f"      ERROR: {e}", flush=True)
        
        del windows, X_train, X_test, pipe
        gc.collect()
    
    # Summary table
    if results:
        print(f"\n  {'Window':>8s} {'Classes':>7s} {'Train':>7s} {'Test':>8s} "
              f"{'Ratio':>6s} {'Time':>6s} {'Status':>6s}", flush=True)
        print(f"  {'-'*55}", flush=True)
        for r in results:
            status = "OK" if r['test_acc'] > 0.5 else (
                "LOW" if r['test_acc'] > (1.0/r['n_classes']) else "FAIL")
            print(f"  {r['window_ms']:6d}ms {r['n_classes']:6d}d "
                  f"{r['train_acc']*100:6.2f}% {r['test_acc']*100:7.2f}% "
                  f"{r['ratio']:5.1f}x {r['time']:5.1f}s {status:>6s}", flush=True)
        
        best = max(results, key=lambda x: x['test_acc'])
        print(f"\n  >>> BEST: {best['window_ms']}ms window = {best['test_acc']*100:.2f}% "
              f"({best['ratio']:.1f}x random)", flush=True)
    
    return results


# =====================================================================
#  Phase 3: Exercise Selection
# =====================================================================
def phase3_exercise_selection(raw_data, label_data, fs, num_kernels, max_train, ds_name):
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 3: EXERCISE SELECTION (400ms windows, overlap=0.75)", flush=True)
    print("=" * 70, flush=True)
    
    window_size = int(400 * fs / 1000)
    overlap = 0.75
    
    results = []
    
    # Define exercise filters
    if ds_name in ['ninapro_db2', 'ninapro_db3']:
        exercise_configs = [
            # الجديد (صحيح مع numpy):
           ("E1 only (17 mov)", lambda lbl: (lbl >= 1) & (lbl <= 17)),
           ("E2 only (23 mov)", lambda lbl: (lbl >= 18) & (lbl <= 40)),
           ("E1+E2 (40 mov)", lambda lbl: (lbl >= 1) & (lbl <= 40)),
           ("All exercises (49 mov)", lambda lbl: lbl != 0),
        ]
    elif ds_name == 'ninapro_db7':
        exercise_configs = [
            ("All movements (no rest)", lambda lbl: lbl != 0),
        ]
    else:
        exercise_configs = [("All (no rest)", lambda lbl: lbl != 0)]
    
    subj_id, emg = raw_data[0]
    labels = label_data[0][1]
    
    for ex_name, filter_fn in exercise_configs:
        # Filter labels
        mask = filter_fn(labels)
        filtered_labels = labels[mask]
        filtered_emg = emg[mask]
        
        n_classes = len(np.unique(filtered_labels))
        print(f"\n  --- {ex_name}: {n_classes} classes ---", flush=True)
        
        if n_classes < 2:
            print(f"      Skipping (need at least 2 classes)", flush=True)
            continue
        
        windows = create_windows(filtered_emg, window_size, overlap)
        win_labels = assign_window_labels(windows, filtered_labels, overlap)
        
        # Remove any residual rest
        mask2 = win_labels != 0
        windows = windows[mask2]
        win_labels = win_labels[mask2]
        
        n_win = len(win_labels)
        if n_win < 100:
            print(f"      Only {n_win} windows, skipping", flush=True)
            continue
        
        random_chance = 1.0 / n_classes
        
        # Shuffle and split 80/20
        rng = np.random.RandomState(42)
        idx = np.arange(n_win)
        rng.shuffle(idx)
        split = int(0.8 * len(idx))
        train_idx, test_idx = idx[:split], idx[split:]
        
        X_train = windows[train_idx]
        y_train = win_labels[train_idx]
        X_test = windows[test_idx]
        y_test = win_labels[test_idx]
        
        # Subsample
        if len(y_train) > max_train:
            total = len(y_train)
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            indices = []
            for cls, count in zip(unique_classes, class_counts):
                cls_idx = np.where(y_train == cls)[0]
                n_sample = min(count, max(2, int(count * max_train / total)))
                chosen = rng.choice(cls_idx, size=n_sample, replace=False)
                indices.append(chosen)
            indices = np.concatenate(indices)
            rng.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        print(f"      Train: {len(y_train):,}, Test: {len(y_test):,}", flush=True)
        print(f"      Random chance: {random_chance*100:.1f}%", flush=True)
        
        t0 = time.time()
        try:
            pipe = MiniRocketPipeline(num_kernels=num_kernels)
            pipe.fit(X_train, y_train)
            train_acc = pipe.score(X_train, y_train)
            test_acc = pipe.score(X_test, y_test)
            dt = time.time() - t0
            
            ratio = test_acc / random_chance
            status = "OK" if test_acc > 0.5 else ("LOW" if test_acc > random_chance else "FAIL")
            
            print(f"      Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}% "
                  f"({ratio:.1f}x random) [{status}] ({dt:.1f}s)", flush=True)
            
            results.append({
                'exercise': ex_name,
                'n_classes': n_classes,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'ratio': ratio,
            })
            
        except Exception as e:
            print(f"      ERROR: {e}", flush=True)
        
        del windows, X_train, X_test, pipe
        gc.collect()
    
    if results:
        print(f"\n  {'Exercise':>25s} {'Classes':>7s} {'Train':>7s} "
              f"{'Test':>8s} {'Ratio':>6s}", flush=True)
        print(f"  {'-'*58}", flush=True)
        for r in results:
            print(f"  {r['exercise']:25s} {r['n_classes']:6d}d "
                  f"{r['train_acc']*100:6.2f}% {r['test_acc']*100:7.2f}% "
                  f"{r['ratio']:5.1f}x", flush=True)
        
        best = max(results, key=lambda x: x['test_acc'])
        print(f"\n  >>> BEST: {best['exercise']} = {best['test_acc']*100:.2f}%", flush=True)
    
    return results


# =====================================================================
#  Phase 4: Full Movement Segmentation (Ovadia 2024 approach)
# =====================================================================
def phase4_full_movement(emg, labels, fs, num_kernels, max_train):
    """
    Ovadia et al. 2024 approach:
    - Segment each movement from the continuous signal
    - Use the ENTIRE movement as one sample (no sliding windows)
    - Remove rest class
    
    This is how they achieved 98.27% on DB5.
    """
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 4: FULL MOVEMENT SEGMENTATION (Ovadia 2024 approach)", flush=True)
    print("=" * 70, flush=True)
    
    # Segment movements
    segments = segment_movements(emg, labels, min_duration=int(200 * fs / 1000))
    
    print(f"  Total segments found: {len(segments)}", flush=True)
    
    if len(segments) == 0:
        print("  ERROR: No movement segments found!", flush=True)
        return None
    
    # Statistics
    lengths = [s[0].shape[0] for s in segments]
    seg_labels = [s[1] for s in segments]
    unique_labels = sorted(set(seg_labels))
    
    print(f"  Unique movements: {len(unique_labels)}", flush=True)
    print(f"  Segment lengths: min={min(lengths)/fs:.2f}s, "
          f"max={max(lengths)/fs:.2f}s, "
          f"mean={np.mean(lengths)/fs:.2f}s, "
          f"median={np.median(lengths)/fs:.2f}s", flush=True)
    
    # Per-class segment count
    label_counts = Counter(seg_labels)
    for lbl in sorted(unique_labels):
        print(f"    Movement {lbl:3d}: {label_counts[lbl]} segments", flush=True)
    
    # Test with different fixed lengths
    results = []
    target_ms_list = [500, 1000, 2000, 3000, 5000]
    
    for target_ms in target_ms_list:
        target_length = int(target_ms * fs / 1000)
        
        # Pad/trim all segments to target length
        X_all = []
        y_all = []
        for seg_emg, seg_lbl in segments:
            padded = pad_or_trim_segment(seg_emg, target_length)
            X_all.append(padded)
            y_all.append(seg_lbl)
        
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.int32)
        
        n_classes = len(np.unique(y_all))
        n_samples = len(y_all)
        random_chance = 1.0 / n_classes
        
        if n_samples < 20:
            print(f"\n  {target_ms:5d}ms: Only {n_samples} segments, skipping", flush=True)
            continue
        
        # Shuffle and split 80/20
        rng = np.random.RandomState(42)
        idx = np.arange(n_samples)
        rng.shuffle(idx)
        split = int(0.8 * len(idx))
        train_idx, test_idx = idx[:split], idx[split:]
        
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        
        # Subsample
        if len(y_train) > max_train:
            total = len(y_train)
            unique_cls, cls_counts = np.unique(y_train, return_counts=True)
            indices = []
            for cls, count in zip(unique_cls, cls_counts):
                cls_idx = np.where(y_train == cls)[0]
                n_sample = min(count, max(2, int(count * max_train / total)))
                chosen = rng.choice(cls_idx, size=n_sample, replace=False)
                indices.append(chosen)
            indices = np.concatenate(indices)
            rng.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        print(f"\n  --- Target length: {target_ms}ms ({target_length} samples) ---", flush=True)
        print(f"      Classes: {n_classes}, Random: {random_chance*100:.1f}%", flush=True)
        print(f"      Train: {len(y_train)}, Test: {len(y_test)}", flush=True)
        
        t0 = time.time()
        try:
            pipe = MiniRocketPipeline(num_kernels=num_kernels)
            pipe.fit(X_train, y_train)
            train_acc = pipe.score(X_train, y_train)
            test_acc = pipe.score(X_test, y_test)
            dt = time.time() - t0
            
            ratio = test_acc / random_chance
            status = "OK" if test_acc > 0.5 else ("LOW" if test_acc > random_chance else "FAIL")
            
            print(f"      Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}% "
                  f"({ratio:.1f}x random) [{status}] ({dt:.1f}s)", flush=True)
            
            results.append({
                'target_ms': target_ms,
                'n_classes': n_classes,
                'n_segments': n_samples,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'ratio': ratio,
                'time': dt,
            })
            
        except Exception as e:
            print(f"      ERROR: {e}", flush=True)
        
        del X_all, X_train, X_test, pipe
        gc.collect()
    
    # Summary
    if results:
        print(f"\n  {'Length':>8s} {'Classes':>7s} {'Segs':>6s} "
              f"{'Train':>7s} {'Test':>8s} {'Ratio':>6s} {'Time':>6s}", flush=True)
        print(f"  {'-'*55}", flush=True)
        for r in results:
            print(f"  {r['target_ms']:6d}ms {r['n_classes']:6d}d "
                  f"{r['n_segments']:5d}  "
                  f"{r['train_acc']*100:6.2f}% {r['test_acc']*100:7.2f}% "
                  f"{r['ratio']:5.1f}x {r['time']:5.1f}s", flush=True)
        
        best = max(results, key=lambda x: x['test_acc'])
        print(f"\n  >>> BEST: {best['target_ms']}ms segments = "
              f"{best['test_acc']*100:.2f}% ({best['ratio']:.1f}x random)", flush=True)
    
    return results


# =====================================================================
#  Main
# =====================================================================
def main():
    args = parse_args()
    config = load_config(args.config)
    
    for ds_key in args.datasets:
        ds_cfg = config.get('datasets', {}).get(ds_key, {})
        data_path = ds_cfg.get('path', '')
        fs = ds_cfg.get('sampling_rate', 2000)
        
        print(f"\n{'#'*70}", flush=True)
        print(f"  Dataset: {ds_key.upper()}", flush=True)
        print(f"  Path: {data_path}", flush=True)
        print(f"  FS: {fs}Hz", flush=True)
        print(f"  Subjects: {args.subjects}", flush=True)
        print(f"  Kernels: {args.num_kernels}", flush=True)
        print(f"  Phase: {args.phase}", flush=True)
        print(f"{'#'*70}", flush=True)
        
        # Load data
        from data_loaders import load_ninapro_db
        
        movement_map = None
        if ds_key == 'ninapro_db3' and not args.no_movement_map:
            movement_map = config.get('db3_to_db7_movement_map')
        
        loader = load_ninapro_db(
            db_version=ds_key.replace('ninapro_', '').upper(),
            data_path=data_path,
            subjects=args.subjects,
            movement_map=movement_map,
            remove_class_zero=False,
        )
        
        raw_data = []
        label_data = []
        for emg, labels, meta in loader:
            subj_id = meta['subject_id']
            raw_data.append((subj_id, np.asarray(emg, dtype=np.float64)))
            label_data.append((subj_id, labels.astype(np.int32)))
        
        if not raw_data:
            print("  ERROR: No data loaded!", flush=True)
            continue
        
        subj_id, emg = raw_data[0]
        labels = label_data[0][1]

        # ---- Self-test: Label Verification ----
        print("\n" + "~" * 70, flush=True)
        print("  SELF-TEST: LABEL VERIFICATION", flush=True)
        print("~" * 70, flush=True)
        unique = np.unique(labels)
        n_unique = len(unique)
        n_mov = len(unique[unique != 0])
        n_rest_pct = 100.0 * np.sum(labels == 0) / len(labels)

        # Check for label gaps
        mov_only = unique[unique != 0]
        if len(mov_only) > 1:
            expected_range = set(range(int(mov_only.min()), int(mov_only.max()) + 1))
            actual = set(int(x) for x in mov_only)
            gaps = expected_range - actual
            if gaps:
                print(f"  WARNING: Label gaps detected! Missing: {sorted(gaps)}", flush=True)
            else:
                print(f"  Label continuity: OK (no gaps in {mov_only.min()}-{mov_only.max()})", flush=True)

        # Check per-exercise expected counts
        if ds_key in ['ninapro_db2', 'ninapro_db3']:
            ex_checks = [('E1/B', 1, 17, 17), ('E2/C', 18, 40, 23), ('E3/D', 41, 49, 9)]
            has_map = movement_map is not None
            # When movement_map is active, E3/D is INTENTIONALLY excluded (no DB7 equivalent)
            expected_total = 49 if (args.no_movement_map or ds_key == 'ninapro_db2') else 40

            if n_mov != expected_total:
                print(f"  WARNING: Expected {expected_total} movements, got {n_mov}", flush=True)
            else:
                print(f"  Movement count: OK ({n_mov}/{expected_total})", flush=True)

            for ex_name, lo, hi, exp_n in ex_checks:
                mask = (labels >= lo) & (labels <= hi)
                found = len(np.unique(labels[mask]))
                found -= 1 if 0 in np.unique(labels[mask]) else 0  # exclude rest
                if found < 0:
                    found = 0

                # v8.4: Better messaging for E3/D when movement_map excludes it
                if ex_name == 'E3/D' and has_map and found == 0:
                    status = "EXCLUDED (movement_map has no DB7 equivalent)"
                elif found == exp_n:
                    status = "OK"
                elif found < exp_n:
                    status = f"MISSING ({exp_n - found}) — check raw data file"
                else:
                    status = f"EXTRA ({found - exp_n})"
                print(f"    {ex_name}: {found}/{exp_n} movements [{status}]", flush=True)

        print(f"  Total: {n_unique} unique labels ({n_mov} movements + rest)", flush=True)
        print(f"  Rest: {n_rest_pct:.1f}%", flush=True)
        print(f"  Movement map: {'DISABLED' if args.no_movement_map else ('ACTIVE' if movement_map else 'N/A')}", flush=True)
        print("~" * 70, flush=True)

        # Run phases
        if args.phase in ['all', 'data']:
            phase1_data_quality(raw_data, label_data, fs, ds_key)
        
        if args.phase in ['all', 'window']:
            phase2_window_sweep(emg, labels, fs, args.num_kernels, args.max_train)
        
        if args.phase in ['all', 'exercise']:
            phase3_exercise_selection(raw_data, label_data, fs, args.num_kernels,
                                      args.max_train, ds_key)
        
        if args.phase in ['all', 'full_movement']:
            phase4_full_movement(emg, labels, fs, args.num_kernels, args.max_train)
        
        del raw_data, label_data
        gc.collect()


if __name__ == '__main__':
    main()
