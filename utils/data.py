"""
Data loading, preprocessing, and PyTorch Dataset for agricultural price CSVs.

Expected CSV format (one file per commodity):
    date,price
    2020-07-22,105.0
    2020-07-23,106.5
    ...

The data pipeline:
    1. Load CSV, parse dates, sort
    2. Forward-fill any residual gaps (already done in students' preprocessing,
       but we re-apply as a safety net)
    3. Train/val/test split — strict temporal order, no shuffle
    4. MinMaxScaler fit on TRAIN only (no data leakage)
    5. Sliding-window Dataset for seq_len → pred_len pairs
    6. Separate time index tensor t in [0, 1] for Time2Vec
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# Constants — default split ratios and window sizes
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10   # implicit: 1 - TRAIN - VAL

DEFAULT_SEQ_LEN  = 90   # 3 months lookback
DEFAULT_PRED_LEN = 14   # 2 weeks ahead


# ---------------------------------------------------------------------------
# Raw data loader
# ---------------------------------------------------------------------------

def load_commodity_csv(path: str) -> pd.DataFrame:
    """
    Load a commodity CSV and return a DataFrame with DatetimeIndex and a
    single 'price' column (retail mid-price), sorted ascending, daily gaps
    forward-filled.

    Handles two formats:
    - 7-column AgriPriceBD format (date, product name, measurement,
      wholesale price minimum, wholesale price maximum, retail price minimum,
      retail price maximum): computes retail mid-price =
      (retail price minimum + retail price maximum) / 2
    - 2-column legacy format (date, price): uses price column directly.
    """
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').set_index('date')

    # Detect 7-column AgriPriceBD format and compute retail mid-price
    cols_lower = [c.lower() for c in df.columns]
    if 'retail price minimum' in cols_lower and 'retail price maximum' in cols_lower:
        retail_min = df.columns[cols_lower.index('retail price minimum')]
        retail_max = df.columns[cols_lower.index('retail price maximum')]
        df = pd.DataFrame(
            {'price': (df[retail_min] + df[retail_max]) / 2},
            index=df.index
        )
    else:
        # Fall back: use first column with 'price' in name, or last numeric
        price_col = [c for c in df.columns if 'price' in c.lower()]
        if not price_col:
            price_col = [df.select_dtypes(include=[np.number]).columns[-1]]
        df = df[[price_col[0]]].rename(columns={price_col[0]: 'price'})

    # Reindex to daily, forward-fill gaps
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(full_idx).ffill().bfill()
    df.index.name = 'date'

    # Range validation (paper Section 3.1): flag values outside 0.1–500 BDT/kg
    PRICE_MIN, PRICE_MAX = 0.1, 500.0
    n_flagged = ((df['price'] < PRICE_MIN) | (df['price'] > PRICE_MAX)).sum()
    if n_flagged > 0:
        import warnings
        warnings.warn(
            f"{path}: {n_flagged} price values outside [{PRICE_MIN}, {PRICE_MAX}] BDT/kg. "
            f"These are retained as-is per paper Section 3.1 transparency policy.",
            UserWarning,
        )

    return df


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict temporal split — no shuffle, no leakage."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# ---------------------------------------------------------------------------
# Scaler — fit on train only
# ---------------------------------------------------------------------------

def fit_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[['price']].values)
    return scaler


def scale(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(df[['price']].values).astype(np.float32)


def inverse_scale(arr: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """arr shape: (n,) or (n, 1)"""
    arr = arr.reshape(-1, 1)
    return scaler.inverse_transform(arr).flatten()


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PriceWindowDataset(Dataset):
    """
    Sliding-window dataset producing (x, t, y) triples.

    x : (seq_len, 1)  — normalised price values (input window)
    t : (seq_len, 1)  — normalised time indices in [0, 1] (for Time2Vec)
    y : (pred_len, 1) — normalised price values (target)

    The time index is the absolute position within the FULL series,
    normalised to [0, 1]. This lets Time2Vec learn global temporal
    patterns (e.g. multi-year seasonality) rather than just within-window
    relative positions.
    """

    def __init__(
        self,
        scaled_values: np.ndarray,
        global_indices: np.ndarray,
        seq_len:  int = DEFAULT_SEQ_LEN,
        pred_len: int = DEFAULT_PRED_LEN,
    ):
        """
        Parameters
        ----------
        scaled_values  : (N, 1) float32 array of normalised prices
        global_indices : (N,)   float32 array of time indices in [0, 1]
        seq_len        : input window length
        pred_len       : forecast horizon
        """
        self.values  = scaled_values
        self.t_index = global_indices
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.n_samples = len(scaled_values) - seq_len - pred_len + 1

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        x_start = idx
        x_end   = idx + self.seq_len
        y_end   = x_end + self.pred_len

        x = torch.tensor(self.values[x_start:x_end],    dtype=torch.float32)  # (seq, 1)
        t = torch.tensor(self.t_index[x_start:x_end].reshape(-1, 1),
                         dtype=torch.float32)                                  # (seq, 1)
        y = torch.tensor(self.values[x_end:y_end],      dtype=torch.float32)  # (pred, 1)
        return x, t, y


# ---------------------------------------------------------------------------
# Full preprocessing pipeline for one commodity
# ---------------------------------------------------------------------------

def prepare_commodity(
    path:     str,
    seq_len:  int = DEFAULT_SEQ_LEN,
    pred_len: int = DEFAULT_PRED_LEN,
    batch_size: int = 32,
) -> Dict:
    """
    End-to-end data preparation for one commodity CSV.

    Returns a dict with:
        'train_loader', 'val_loader', 'test_loader'
        'scaler'                   — for inverse transforming predictions
        'test_dates'               — DatetimeIndex for plotting
        'test_values_raw'          — unscaled test prices (for metrics)
        'n_train', 'n_val', 'n_test'
        'total_len'
    """
    df = load_commodity_csv(path)
    total_len = len(df)

    train_df, val_df, test_df = temporal_split(df)
    scaler = fit_scaler(train_df)

    # Global time index — position in FULL series normalised to [0,1]
    # This is critical for Time2Vec to discover inter-year patterns.
    all_indices = np.arange(total_len, dtype=np.float32) / (total_len - 1)

    train_size = len(train_df)
    val_size   = len(val_df)

    train_vals = scale(train_df, scaler)
    val_vals   = scale(val_df,   scaler)
    test_vals  = scale(test_df,  scaler)

    train_idx = all_indices[:train_size]
    val_idx   = all_indices[train_size:train_size + val_size]
    test_idx  = all_indices[train_size + val_size:]

    train_ds = PriceWindowDataset(train_vals, train_idx, seq_len, pred_len)
    val_ds   = PriceWindowDataset(val_vals,   val_idx,   seq_len, pred_len)
    test_ds  = PriceWindowDataset(test_vals,  test_idx,  seq_len, pred_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return {
        'train_loader':      train_loader,
        'val_loader':        val_loader,
        'test_loader':       test_loader,
        'scaler':            scaler,
        'test_dates':        test_df.index,
        'test_values_raw':   test_df['price'].values,
        'train_df':          train_df,
        'val_df':            val_df,
        'test_df':           test_df,
        'n_train':           len(train_df),
        'n_val':             len(val_df),
        'n_test':            len(test_df),
        'total_len':         total_len,
    }