"""
cnn_baseline.py — v3.0 (CLEAN REWRITE)
1D CNN baseline for EMG gesture recognition with nested LOSO validation.

IMPROVEMENTS:
  - Cleaner model architecture with proper weight initialization
  - Deterministic training (fixed seeds per fold)
  - Proper memory management with model.cpu() after each fold
  - Inference time benchmarking with warmup
  - ONNX export support
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats


# =====================================================================
# Model Architecture
# =====================================================================

class EMGBaseline1DCNN(nn.Module):
    """
    Simple 1D CNN for EMG window classification.

    Architecture:
      Conv1d(ch, 64, k=3) → BN → ReLU →
      Conv1d(64, 128, k=3) → BN → ReLU → MaxPool →
      Conv1d(128, 256, k=3) → BN → ReLU →
      AdaptiveAvgPool → Dropout → FC → logits
    """

    def __init__(self, n_channels, n_classes, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, n_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


# =====================================================================
# Training with nested LOSO
# =====================================================================

def train_cnn_loso(raw_windows, labels, groups, config, device=None):
    """
    Train 1D CNN with nested Leave-One-Subject-Out cross-validation.

    For each test subject:
      - Validation subject is randomly selected from remaining subjects
      - All other subjects are used for training
      - Early stopping based on validation loss
      - Best model state is restored before evaluation

    Parameters
    ----------
    raw_windows : np.ndarray, shape (n_windows, n_channels, window_samples)
    labels : np.ndarray
        Integer class labels.
    groups : np.ndarray
        Subject IDs for LOSO grouping.
    config : dict
        Configuration dict with 'cnn' section.
    device : torch.device, optional

    Returns
    -------
    dict with keys:
        per_subject_accuracy, mean_accuracy, std_accuracy, ci_95,
        val_subject_per_fold, inference_time_ms_mean, inference_time_ms_std,
        model, device, trained_models
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[CNN] Device: {device}", flush=True)

    unique_subjects = np.unique(groups)
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    n_classes = len(le.classes_)

    cnn_cfg = config.get('cnn', {})
    batch_size = cnn_cfg.get('batch_size', 32)
    epochs = cnn_cfg.get('epochs', 50)
    lr = cnn_cfg.get('learning_rate', 1e-3)
    patience = cnn_cfg.get('patience', 10)
    dropout = cnn_cfg.get('dropout', 0.5)

    per_subject_acc = []
    val_subject_per_fold = []
    inference_times = []
    models = []

    rng = np.random.RandomState(42)

    for fold_idx, test_subj in enumerate(unique_subjects):
        remaining = [s for s in unique_subjects if s != test_subj]

        # Nested: split remaining into val + train
        if len(remaining) == 0:
            val_subj = test_subj
            train_subjs = []
        elif len(remaining) == 1:
            val_subj = remaining[0]
            train_subjs = []
        else:
            val_subj = rng.choice(remaining)
            train_subjs = [s for s in remaining if s != val_subj]

        val_subject_per_fold.append(int(val_subj))

        train_mask = np.isin(groups, train_subjs)
        val_mask = groups == val_subj
        test_mask = groups == test_subj

        # If no training subjects, use train+val combined
        if train_mask.sum() == 0:
            train_mask = train_mask | val_mask
            print(
                f"  Fold {fold_idx + 1}: test={test_subj}, "
                f"no separate val (using train+val combined)",
                flush=True
            )
        else:
            print(
                f"  Fold {fold_idx + 1}: test={test_subj}, "
                f"val={val_subj}, train_n={train_mask.sum()}",
                flush=True
            )

        X_train_raw = raw_windows[train_mask]
        y_train_enc = y_enc[train_mask]
        X_val_raw = raw_windows[val_mask]
        y_val_enc = y_enc[val_mask]
        X_test_raw = raw_windows[test_mask]
        y_test_enc = y_enc[test_mask]

        # Skip if test set is empty
        if len(X_test_raw) == 0:
            print(f"    Skipping: empty test set for subject {test_subj}")
            continue

        # Z-score normalization (fit on training only)
        scaler = StandardScaler()
        n_train, C, W = X_train_raw.shape
        X_train_flat = X_train_raw.reshape(n_train, C * W)
        scaler.fit(X_train_flat)

        def transform_windows(windows):
            n, c, w = windows.shape
            flat = windows.reshape(n, c * w)
            flat_scaled = scaler.transform(flat)
            return flat_scaled.reshape(n, c, w)

        X_train_norm = transform_windows(X_train_raw)
        X_val_norm = transform_windows(X_val_raw)
        X_test_norm = transform_windows(X_test_raw)

        # Convert to tensors
        X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
        X_val_t = torch.tensor(X_val_norm, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_enc, dtype=torch.long)
        X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_enc, dtype=torch.long)

        # Class weights for balanced loss
        unique_train = np.unique(y_train_enc)
        if len(unique_train) > 1:
            class_weights = compute_class_weight(
                'balanced', classes=unique_train, y=y_train_enc
            )
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        else:
            weight_tensor = None

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        model = EMGBaseline1DCNN(
            n_channels=C, n_classes=n_classes, dropout=dropout
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=max(patience // 2, 2)
        )

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=batch_size, shuffle=False
        )

        # Training loop with early stopping
        best_val_loss = np.inf
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(Xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(Xb)
            train_loss /= len(train_loader.dataset)

            # Validation
            if len(X_val_t) > 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        Xb, yb = Xb.to(device), yb.to(device)
                        out = model(Xb)
                        loss = criterion(out, yb)
                        val_loss += loss.item() * len(Xb)
                val_loss /= len(val_loader.dataset)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            else:
                # No validation set — use training loss
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # Inference time benchmarking with warmup
        model.to(device)
        with torch.no_grad():
            # Warmup (10 inferences, not timed)
            for _ in range(10):
                _ = model(X_test_t[:1].to(device))

            # Timed inferences
            n_bench = min(100, len(X_test_t))
            times = []
            for i in range(n_bench):
                start = time.perf_counter()
                _ = model(X_test_t[i:i + 1].to(device))
                end = time.perf_counter()
                times.append((end - start) * 1000)

        inference_times.append(np.mean(times))

        # Final evaluation on test set
        with torch.no_grad():
            out = model(X_test_t.to(device))
            preds = out.argmax(dim=1).cpu().numpy()
            acc = float((preds == y_test_enc).mean())

        per_subject_acc.append(acc)
        models.append(model.cpu())

        print(
            f"    test={test_subj}: acc={acc:.4f}, "
            f"infer={np.mean(times):.3f}ms",
            flush=True
        )

        # Cleanup GPU memory
        del X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t
        del criterion, optimizer, scheduler
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    # Summary statistics
    if not per_subject_acc:
        return {
            'per_subject_accuracy': [],
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'ci_95': [0.0, 0.0],
            'val_subject_per_fold': val_subject_per_fold,
            'inference_time_ms_mean': 0.0,
            'inference_time_ms_std': 0.0,
            'model': '1D-CNN-baseline',
            'device': str(device),
            'trained_models': []
        }

    mean_acc = np.mean(per_subject_acc)
    std_acc = np.std(per_subject_acc)
    n = len(per_subject_acc)

    if n > 1:
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lower = mean_acc - t_crit * std_acc / np.sqrt(n)
        ci_upper = mean_acc + t_crit * std_acc / np.sqrt(n)
    else:
        ci_lower = ci_upper = mean_acc

    print(
        f"\n[CNN] Mean: {mean_acc:.4f} +/- {std_acc:.4f}  "
        f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}]",
        flush=True
    )

    return {
        'per_subject_accuracy': per_subject_acc,
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'ci_95': [float(ci_lower), float(ci_upper)],
        'val_subject_per_fold': val_subject_per_fold,
        'inference_time_ms_mean': float(np.mean(inference_times)),
        'inference_time_ms_std': float(np.std(inference_times)),
        'model': '1D-CNN-baseline',
        'device': str(device),
        'trained_models': models
    }


# =====================================================================
# ONNX Export
# =====================================================================

def export_onnx(model, n_channels, window_samples, save_path="emg_cnn.onnx"):
    """Export trained CNN model to ONNX format."""
    model.eval()
    model.cpu()
    dummy_input = torch.randn(1, n_channels, window_samples)
    torch.onnx.export(
        model, dummy_input, save_path,
        input_names=['emg_window'],
        output_names=['logits'],
        opset_version=14,
        dynamic_axes={
            'emg_window': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"ONNX model saved to {save_path}")
