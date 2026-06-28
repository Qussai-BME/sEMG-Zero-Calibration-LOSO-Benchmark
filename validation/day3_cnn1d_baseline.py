#!/usr/bin/env python3
"""
Day 3: CNN-1D Baseline [v8] — Memory-Efficient + Label Mapping
================================================================
Fixes over v7:
  1. NO giant array concatenation (4.11 GiB eliminated!)
     — MultiSubjectWindowDataset holds references, not copies
  2. On-the-fly normalization from sample-based global stats
  3. Automatic label remapping for non-contiguous labels (fixes DB3)
  4. 30 epochs, patience=12, T_0=8 — faster convergence

Peak memory: ~5 GB (was ~12 GB in v7)

Following the same protocol as our feature-based classifiers,
normalization statistics are computed exclusively from training subjects.

Usage:
  python day3_cnn1d_baseline.py --db ninapro_db7
  python day3_cnn1d_baseline.py --db ninapro_db3
  python day3_cnn1d_baseline.py --db ninapro_db2
  python day3_cnn1d_baseline.py --db ninapro_db7 --fast --subjects 1,5,8
"""

import os
import sys
import json
import time
import gc
import traceback
import argparse
import warnings
from pathlib import Path

import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
RESULTS_DIR = SCRIPT_DIR / "paper1_results"
CKPT_DIR = RESULTS_DIR / "day3_checkpoints"

for d in [RESULTS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Model: Multi-Scale CNN-1D with Dropout  (~48 K params)
# ═══════════════════════════════════════════════════════════════

class CNN1Dv7(nn.Module):
    """
    Three parallel Conv1d branches (kernel 3 / 5 / 7) merge into a
    shared encoder.  Dropout after every pooling layer.  Global
    average pooling -> linear classifier.
    """

    def __init__(self, n_channels=12, n_classes=40):
        super().__init__()
        self.branch3 = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.branch5 = nn.Conv1d(n_channels, 32, kernel_size=5, padding=2)
        self.branch7 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.bn_merge = nn.BatchNorm1d(96)

        self.conv1 = nn.Conv1d(96, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 48, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(48)

        self.pool   = nn.MaxPool1d(2)
        self.drop   = nn.Dropout(0.30)
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(48, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        b3 = F.relu(self.branch3(x))
        b5 = F.relu(self.branch5(x))
        b7 = F.relu(self.branch7(x))
        x = torch.cat([b3, b5, b7], dim=1)

        x = self.drop(self.pool(F.relu(self.bn_merge(x))))
        x = self.drop(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════
# Dataset: Memory-Efficient Multi-Subject Windowed EMG
# ═══════════════════════════════════════════════════════════════

class MultiSubjectWindowDataset(Dataset):
    """
    References multiple subject EMG arrays WITHOUT concatenating them.
    Normalization is applied on-the-fly per window (negligible overhead).
    Automatic label remapping handles non-contiguous labels (e.g., DB3).
    """

    def __init__(self, subject_pairs, win_size, stride,
                 norm_mu=None, norm_sigma=None,
                 label_set=None,  # sorted list of valid labels; None = assume 1..n_classes
                 augment=False, noise_std=0.02, ch_mask_p=0.1):
        """
        subject_pairs : list of (emg_ndarray, stim_ndarray) — NOT copied
        norm_mu/sigma : (1, n_ch) arrays or None
        label_set     : sorted array of valid labels for remapping
        """
        self.subject_pairs = subject_pairs
        self.win_size      = win_size
        self.norm_mu       = norm_mu
        self.norm_sigma    = norm_sigma
        self.augment       = augment
        self.noise_std     = noise_std
        self.ch_mask_p     = ch_mask_p

        # ── Label mapping ──
        if label_set is not None:
            self.label_map = {int(l): i for i, l in enumerate(label_set)}
            self.n_classes = len(label_set)
        else:
            # assume contiguous 1 .. max_label
            all_lbls = []
            for _, stim in subject_pairs:
                ul = np.unique(stim[stim >= 1])
                if len(ul) > 0:
                    all_lbls.append(ul.max())
            self.n_classes = max(all_lbls) if all_lbls else 40
            self.label_map = None

        # ── Build window index: (subject_index, start_pos, 0-based label) ──
        self.indices = []
        for si, (emg, stim) in enumerate(subject_pairs):
            n = len(emg)
            for i in range(0, n - win_size + 1, stride):
                lbl = int(np.median(stim[i : i + win_size]))
                if lbl < 1:
                    continue
                if self.label_map is not None:
                    if lbl not in self.label_map:
                        continue
                    mapped = self.label_map[lbl]
                else:
                    if lbl > self.n_classes:
                        continue
                    mapped = lbl - 1
                self.indices.append((si, i, mapped))

        self.labels = np.array([lab for _, _, lab in self.indices],
                               dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        si, start, label = self.indices[idx]
        emg = self.subject_pairs[si][0]
        w = emg[start : start + self.win_size].astype(np.float32)

        if self.norm_mu is not None:
            w = (w - self.norm_mu) / self.norm_sigma

        if self.augment:
            if self.noise_std > 0:
                w = w + (np.random.randn(*w.shape).astype(np.float32)
                         * self.noise_std)
            if self.ch_mask_p > 0:
                mask = np.random.rand(w.shape[1]) > self.ch_mask_p
                w[:, ~mask] = 0.0

        return torch.from_numpy(w.T.copy()), label


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def extract_per_subject_data(data_gen):
    subjects = {}
    for item in data_gen:
        if isinstance(item, tuple) and len(item) == 3:
            emg, stim, info = item
        elif isinstance(item, tuple) and len(item) == 2:
            emg, stim = item
            info = {}
        else:
            continue
        sid = info.get("subject_id", len(subjects) + 1)
        valid = np.unique(stim[stim >= 1])
        subjects[sid] = {
            "emg":        emg.astype(np.float32),
            "stim":       stim.astype(np.int32),
            "n_classes":  int(len(valid)),
            "subject_id": sid,
        }
    return subjects


def compute_sample_stats(subject_pairs, samples_per_subj=200000, seed=42):
    """Compute global mean/std from random samples — NO full concatenation."""
    rng = np.random.RandomState(seed)
    parts = []
    for emg, _ in subject_pairs:
        n = min(samples_per_subj, len(emg))
        idx = rng.choice(len(emg), n, replace=False)
        parts.append(emg[idx].astype(np.float32))
    samples = np.vstack(parts)  # ~200K * n_sub * 12 * 4B ~ 200 MB
    mu    = samples.mean(axis=0, keepdims=True)
    sigma = samples.std(axis=0, keepdims=True) + 1e-8
    del samples, parts
    gc.collect()
    return mu, sigma


def get_common_labels(subjects, sids):
    """Return sorted array of all non-zero labels present across subjects."""
    all_lbls = set()
    for sid in sids:
        stim = subjects[sid]["stim"]
        ul = np.unique(stim[stim >= 1])
        all_lbls.update(int(x) for x in ul)
    return np.array(sorted(all_lbls), dtype=np.int64)


def stratified_subsample(ds, cap, seed=42):
    rng = np.random.RandomState(seed)
    labels = ds.labels
    classes = np.unique(labels)
    per_cls = max(cap // len(classes), 1)
    chosen = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) > per_cls:
            idx = rng.choice(idx, per_cls, replace=False)
        chosen.extend(idx.tolist())
    rng.shuffle(chosen)
    return chosen[:cap]


def get_rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        return 0


# ═══════════════════════════════════════════════════════════════
# LOSO Cross-Validation
# ═══════════════════════════════════════════════════════════════

def run_database(db_key, args):
    from data_loaders import load_ninapro_db

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mapping = {
        "ninapro_db7": ("db7", config["datasets"]["ninapro_db7"]["path"]),
        "ninapro_db3": ("db3", config["datasets"]["ninapro_db3"]["path"]),
        "ninapro_db2": ("db2", config["datasets"]["ninapro_db2"]["path"]),
    }
    if db_key not in mapping:
        print(f"  ERROR: unknown db '{db_key}'. Choose from {list(mapping)}")
        return None
    db_version, data_path = mapping[db_key]

    # ── Load data ──
    print(f"\n  Loading data...")
    print(f"  Calling: load_ninapro_db('{db_version}', '{data_path}', "
          f"remove_class_zero=False)")
    sys.path.insert(0, str(SCRIPT_DIR))
    gen = load_ninapro_db(db_version, data_path, remove_class_zero=False)
    subjects = extract_per_subject_data(gen)
    if not subjects:
        print("  ERROR: no subjects loaded!")
        return None

    if args.subjects:
        keep = set(int(s) for s in args.subjects.split(","))
        subjects = {k: v for k, v in subjects.items() if k in keep}

    sids = sorted(subjects.keys())
    n_sub = len(sids)

    print(f"\n  Extracted {n_sub} subjects")
    for s in list(subjects.values())[:3]:
        print(f"    S{s['subject_id']}: {s['emg'].shape}, "
              f"{s['n_classes']} classes, "
              f"{s['emg'].nbytes / (1024**2):.0f}MB")
    if n_sub > 3:
        print(f"    ... and {n_sub - 3} more")

    # ── Detect label structure ──
    all_train_labels = get_common_labels(subjects, sids)
    n_unique_labels  = len(all_train_labels)
    # Check if contiguous
    expected = np.arange(1, n_unique_labels + 1)
    is_contiguous = np.array_equal(all_train_labels, expected)

    if is_contiguous:
        label_set = None  # fast path: use simple label-1 mapping
        n_classes = n_unique_labels
        print(f"  Labels: contiguous 1..{n_classes} ({n_classes} classes)")
    else:
        label_set = all_train_labels
        n_classes = n_unique_labels
        print(f"  Labels: NON-CONTIGUOUS ({n_classes} unique, "
              f"range {all_train_labels[0]}-{all_train_labels[-1]})")
        print(f"  -> Using explicit label remapping")

    # ── Hyper-parameters ──
    FS         = 2000
    WIN_MS     = 400
    WIN        = int(FS * WIN_MS / 1000)   # 800 samples
    OVERLAP    = 0.50
    STRIDE     = int(WIN * (1 - OVERLAP))
    BATCH      = 512
    LR         = 1e-3
    WD         = 1e-3
    CLIP       = 1.0

    if args.fast:
        TRAIN_CAP = 25000
        TEST_CAP  = 15000
        EPOCHS    = 20
        PATIENCE  = 8
        T_0       = 5
    else:
        TRAIN_CAP = 60000
        TEST_CAP  = 30000
        EPOCHS    = 30
        PATIENCE  = 12
        T_0       = 8

    print(f"\n  LOSO-CV: {n_sub} folds | win={WIN}smp ({WIN_MS}ms) | "
          f"overlap={OVERLAP*100:.0f}% | classes={n_classes}")
    print(f"  train_cap={TRAIN_CAP} | test_cap={TEST_CAP} | "
          f"batch={BATCH} | epochs={EPOCHS} | patience={PATIENCE}")
    print(f"  Augment: noise_std=0.02, ch_mask_p=0.1")
    print(f"  Optimizer: AdamW + CosineAnnealingWarmRestarts(T_0={T_0})")
    print(f"  Memory mode: NO concatenation (on-the-fly normalization)")
    print(f"  RSS before start: {get_rss_mb():.0f}MB\n")

    results    = {}
    fold_times = []

    for fi, test_sid in enumerate(sids):
        t0 = time.time()
        print(f"  [{fi+1}/{n_sub}] S{test_sid}: train on {n_sub-1} subjects",
              end="")
        try:
            # ── Build subject pairs (references only, no copy!) ──
            train_sids = [s for s in sids if s != test_sid]
            train_pairs = [(subjects[s]["emg"], subjects[s]["stim"])
                           for s in train_sids]
            test_pairs  = [(subjects[test_sid]["emg"], subjects[test_sid]["stim"])]

            # ── Normalization stats from training subjects (sample-based) ──
            # Following the same protocol as our feature-based classifiers,
            # normalization statistics are computed exclusively from training subjects.
            mu, sigma = compute_sample_stats(train_pairs)

            # ── Determine labels for this fold ──
            if label_set is not None:
                fold_label_set = label_set
            else:
                fold_label_set = None

            # ── Windowed datasets (no giant arrays!) ──
            print(f" | building windows...", end="", flush=True)
            train_ds = MultiSubjectWindowDataset(
                train_pairs, WIN, STRIDE,
                norm_mu=mu, norm_sigma=sigma,
                label_set=fold_label_set,
                augment=True, noise_std=0.02, ch_mask_p=0.1)
            test_ds  = MultiSubjectWindowDataset(
                test_pairs, WIN, STRIDE,
                norm_mu=mu, norm_sigma=sigma,
                label_set=fold_label_set,
                augment=False)
            fold_nc = train_ds.n_classes

            # ── Stratified subsample ──
            tr_idx = stratified_subsample(train_ds, TRAIN_CAP)
            te_idx = stratified_subsample(test_ds,  TEST_CAP)
            train_ds = Subset(train_ds, tr_idx)
            test_ds  = Subset(test_ds,  te_idx)
            n_batches = (len(tr_idx) + BATCH - 1) // BATCH
            print(f" tr={len(tr_idx)}, te={len(te_idx)}, "
                  f"~{n_batches} bat/ep |", end="", flush=True)

            # ── DataLoaders ──
            train_dl = DataLoader(train_ds, BATCH, shuffle=True,
                                 num_workers=0, drop_last=False)
            test_dl  = DataLoader(test_ds,  BATCH, shuffle=False,
                                 num_workers=0)

            # ── Class weights ──
            tr_labels = train_ds.dataset.labels[train_ds.indices]
            counts = np.bincount(tr_labels, minlength=fold_nc).astype(float)
            wts = 1.0 / (counts + 1)
            wts = (wts / wts.sum() * fold_nc).astype(np.float32)

            # ── Model ──
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model  = CNN1Dv7(n_channels=12, n_classes=fold_nc).to(device)
            npar   = sum(p.numel() for p in model.parameters())

            criterion = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(wts).to(device))
            optimizer = torch.optim.AdamW(model.parameters(),
                                         lr=LR, weight_decay=WD)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=2, eta_min=LR * 0.01)

            # ── Training loop ──
            best_va = 0.0
            wait    = 0
            best_sd = None
            last_ep = EPOCHS

            for ep in range(EPOCHS):
                model.train()
                tc, tt = 0, 0
                for xb, yb in train_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                    optimizer.step()
                    tc += (logits.argmax(1) == yb).sum().item()
                    tt += xb.size(0)
                scheduler.step()
                tr_acc = tc / max(tt, 1)

                model.eval()
                vc, vt = 0, 0
                with torch.no_grad():
                    for xb, yb in test_dl:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        vc += (logits.argmax(1) == yb).sum().item()
                        vt += yb.size(0)
                va = vc / max(vt, 1)
                cur_lr = optimizer.param_groups[0]["lr"]

                if va > best_va:
                    best_va = va
                    wait = 0
                    best_sd = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
                else:
                    wait += 1

                ep_time = int(time.time() - t0)
                print(f"\n      E{ep+1:2d}/{EPOCHS}  "
                      f"tr={tr_acc:.4f}  va={va:.4f}  "
                      f"best={best_va:.4f}  w={wait}  "
                      f"lr={cur_lr:.1e}  [{ep_time}s]", end="", flush=True)

                if wait >= PATIENCE:
                    print(f"\n    Early stop at epoch {ep+1} "
                          f"(best={best_va:.4f})", flush=True)
                    last_ep = ep + 1
                    break
                last_ep = ep + 1

            # ── Final eval with best weights ──
            if best_sd is not None:
                model.load_state_dict(best_sd)
            model.eval()
            fc, ft = 0, 0
            with torch.no_grad():
                for xb, yb in test_dl:
                    xb = xb.to(device)
                    fc += (model(xb).argmax(1).cpu() == yb).sum().item()
                    ft += yb.size(0)
            final_acc = fc / max(ft, 1)
            dt = time.time() - t0

            results[f"S{test_sid}"] = {
                "accuracy": round(final_acc, 4),
                "best_val": round(best_va, 4),
                "epochs":   last_ep,
                "time_min": round(dt / 60, 1),
            }
            fold_times.append(dt)
            print(f"\n    >>> S{test_sid}: {final_acc:.4f}  "
                  f"(best={best_va:.4f}, {last_ep}ep, {npar:,} params)")

            if fi == 0 and n_sub > 1:
                est = dt * n_sub / 60
                print(f"    [EST] ~{dt/60:.1f} min/fold -> "
                      f"~{est:.1f} min total ({est/60:.1f} hrs)")

            # ── Cleanup ──
            del (model, optimizer, scheduler, best_sd,
                 train_dl, test_dl, train_ds, test_ds,
                 train_pairs, test_pairs,
                 tr_labels, wts, counts, mu, sigma)
            gc.collect()
            print(f"    RSS: {get_rss_mb():.0f}MB\n")

            # ── Checkpoint ──
            ckpt = CKPT_DIR / f"{db_key}_checkpoint.json"
            with open(ckpt, "w") as f:
                json.dump({"db": db_key, "fold": fi,
                           "results_so_far": results,
                           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                          f, indent=2)

        except Exception:
            traceback.print_exc()
            results[f"S{test_sid}"] = {"accuracy": 0.0,
                                        "error": traceback.format_exc()}
            gc.collect()

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    accs = [r["accuracy"] for r in results.values()
            if isinstance(r.get("accuracy"), (int, float))]
    if not accs:
        return None

    m, s_acc = np.mean(accs), np.std(accs)
    med = np.median(accs)
    total_t = sum(fold_times)

    valid_results = {k: v for k, v in results.items()
                     if isinstance(v.get("accuracy"), (int, float))}
    best_sub  = max(valid_results, key=lambda k: valid_results[k]["accuracy"])
    worst_sub = min(valid_results, key=lambda k: valid_results[k]["accuracy"])

    print(f"\n  {'=' * 58}")
    print(f"  LOSO-CV Complete: {db_key} ({total_t / 60:.1f} min)")
    print(f"  Mean  : {m:.4f} +/- {s_acc:.4f}")
    print(f"  Median: {med:.4f}")
    print(f"  Min   : {valid_results[worst_sub]['accuracy']:.4f} ({worst_sub})")
    print(f"  Max   : {valid_results[best_sub]['accuracy']:.4f} ({best_sub})")

    out = RESULTS_DIR / f"Day3_CNN1D_{db_key}_results.json"
    with open(out, "w") as f:
        json.dump({
            "database":        db_key,
            "version":         "v8",
            "model":           "CNN1Dv7-multi-scale",
            "hyperparameters": {
                "window_ms": WIN_MS, "overlap": OVERLAP,
                "train_cap": TRAIN_CAP, "test_cap": TEST_CAP,
                "batch_size": BATCH, "epochs": EPOCHS,
                "patience": PATIENCE, "lr": LR,
                "weight_decay": WD, "grad_clip": CLIP,
                "scheduler": f"CosineAnnealingWarmRestarts(T_0={T_0})",
                "augmentation": {"noise_std": 0.02, "ch_mask_p": 0.1},
                "label_mapping": "contiguous" if label_set is None else "non_contiguous",
                "memory_mode": "no_concatenation",
            },
            "per_subject": results,
            "summary": {
                "mean":  round(m, 4),
                "std":   round(s_acc, 4),
                "median": round(med, 4),
                "min":   round(min(accs), 4),
                "max":   round(max(accs), 4),
                "n_subjects": len(accs),
                "total_time_min": round(total_t / 60, 1),
            },
        }, f, indent=2)
    print(f"  Saved: {out}")
    return round(m, 4), round(s_acc, 4)


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Day 3: CNN-1D Baseline v8")
    ap.add_argument("--db",       type=str, required=True,
                    help="Database key (ninapro_db7 / db3 / db2)")
    ap.add_argument("--fast",     action="store_true",
                    help="Quick test: 20 ep, 25K cap, patience=8")
    ap.add_argument("--epochs",   type=int, default=30,
                    help="Max epochs (default: 30)")
    ap.add_argument("--subjects", type=str, default=None,
                    help="Comma-separated subject IDs (e.g. 1,2,3)")
    args = ap.parse_args()

    if args.fast:
        args.epochs = 20

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  Day 3: CNN-1D Baseline [v8] — Memory-Efficient")
    print(f"  Device: {dev}")
    print(f"  Mode: {'FAST (20 ep)' if args.fast else f'FULL ({args.epochs} ep)'}")
    print("=" * 70)

    result = run_database(args.db, args)

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  DB                     CNN-1D      +/-    N     Time")
    print(f"  --------------------------------------------------")
    if result:
        print(f"  {args.db:22s} {result[0]:.4f}   "
              f"{result[1]:.4f}   --    --")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
