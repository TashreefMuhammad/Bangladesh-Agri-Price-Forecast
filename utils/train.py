"""
Training loop, evaluation, and metrics for all deep learning models.

All three DL models (BiLSTM, VanillaTransformer, T2V_Transformer) share
the same training loop — the only difference is whether the time index
tensor `t` is passed to the model forward() call.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import copy


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'MAE':  mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    n_epochs:     int  = 100,
    lr:           float = 1e-3,
    patience:     int  = 20,
    use_time_idx: bool = True,
    verbose:      bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Train a deep learning model with early stopping.

    Parameters
    ----------
    model         : nn.Module (BiLSTM, VanillaTransformer, or T2V_Transformer)
    train_loader  : DataLoader yielding (x, t, y) batches
    val_loader    : DataLoader yielding (x, t, y) batches
    device        : torch.device
    n_epochs      : maximum training epochs
    lr            : initial learning rate
    patience      : early stopping patience (epochs without val improvement)
    use_time_idx  : if True, passes `t` to model.forward(x, t)
                    if False, passes only x (for BiLSTM and VanillaTransformer)
    verbose       : print epoch progress

    Returns
    -------
    best_model : model with lowest validation loss (deep copy)
    history    : dict with 'train_loss' and 'val_loss' lists
    """
    model = model.to(device)
    criterion = nn.HuberLoss(delta=1.0)   # more robust to price outliers than MSE
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=10, min_lr=1e-6)

    best_val_loss  = float('inf')
    best_model     = None
    patience_count = 0
    history        = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, n_epochs + 1):
        # --- Training ---
        model.train()
        train_losses = []
        for x, t, y in train_loader:
            x, t, y = x.to(device), t.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, t) if use_time_idx else model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, t, y in val_loader:
                x, t, y = x.to(device), t.to(device), y.to(device)
                pred = model(x, t) if use_time_idx else model(x)
                val_losses.append(criterion(pred, y).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{n_epochs} | "
                  f"train={train_loss:.5f} | val={val_loss:.5f} | lr={lr_now:.2e}")

        # --- Early stopping ---
        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            best_model     = copy.deepcopy(model)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best val={best_val_loss:.5f})")
                break

    return best_model, history


# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------

def evaluate_model(
    model:       nn.Module,
    test_loader: DataLoader,
    scaler,
    device:      torch.device,
    use_time_idx: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on test set and return (predictions, ground_truth)
    both in original (unscaled) price units.
    """
    model.eval()
    all_preds = []
    all_true  = []

    with torch.no_grad():
        for x, t, y in test_loader:
            x, t = x.to(device), t.to(device)
            pred = model(x, t) if use_time_idx else model(x)
            # pred: (batch, pred_len, 1) -> flatten to (batch * pred_len,)
            all_preds.append(pred.cpu().numpy().reshape(-1))
            all_true.append(y.numpy().reshape(-1))

    preds = np.concatenate(all_preds)
    truth = np.concatenate(all_true)

    # Inverse-scale
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    truth = scaler.inverse_transform(truth.reshape(-1, 1)).flatten()

    return preds, truth


# ---------------------------------------------------------------------------
# Naïve persistence evaluation (no DataLoader needed)
# ---------------------------------------------------------------------------

def evaluate_naive(test_df, scaler, pred_len: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate naïve persistence on the test set.
    Predict last value of each input window for all pred_len steps.
    """
    from .data import scale
    vals = scale(test_df, scaler)
    n = len(vals)
    preds, truth = [], []

    for i in range(n - seq_len - pred_len + 1):
        last_val = vals[i + seq_len - 1, 0]
        pred_window = np.full(pred_len, last_val)
        true_window = vals[i + seq_len: i + seq_len + pred_len, 0]
        preds.append(pred_window)
        truth.append(true_window)

    preds = scaler.inverse_transform(np.concatenate(preds).reshape(-1,1)).flatten()
    truth = scaler.inverse_transform(np.concatenate(truth).reshape(-1,1)).flatten()
    return preds, truth