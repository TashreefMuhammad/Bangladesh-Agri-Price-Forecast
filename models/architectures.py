"""
Model architectures for agricultural price forecasting.

Four models are implemented here, forming a clean ablation ladder:
  1. BiLSTM            — deep learning baseline (no temporal encoding)
  2. Transformer       — standard Transformer with fixed sinusoidal PE
  3. T2V_Transformer   — ablation model: Transformer with Time2Vec encoding (Kazemi et al., 2019)
  4. NaivePersistence  — statistical floor baseline (not trained)

Design philosophy for Colab free-tier compatibility:
  - All deep models use d_model=64, n_heads=4, n_layers=2 by default
  - Parameter counts are kept well below 500K
  - No custom CUDA kernels; runs on CPU if needed
"""

import math
import torch
import torch.nn as nn
from .time2vec import Time2Vec


# ---------------------------------------------------------------------------
# 1. Naïve Persistence
# ---------------------------------------------------------------------------

class NaivePersistence:
    """
    Predict the last observed value for all future steps.
    Not a neural model — used as a sanity-check floor in evaluation.
    """
    def predict(self, x: torch.Tensor, pred_len: int) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (batch, seq_len, 1)  — input window
        pred_len : int

        Returns
        -------
        (batch, pred_len, 1)
        """
        last = x[:, -1:, :]                          # (batch, 1, 1)
        return last.expand(-1, pred_len, -1)          # (batch, pred_len, 1)


# ---------------------------------------------------------------------------
# 2. BiLSTM
# ---------------------------------------------------------------------------

class BiLSTM(nn.Module):
    """
    Bidirectional LSTM forecaster.

    Architecture:
        Input projection → BiLSTM stack → last hidden → linear head

    Note: bidirectional is valid for forecasting when the full input
    window is available at inference time (batch mode, not streaming).
    """

    def __init__(
        self,
        input_dim:  int = 1,
        hidden_dim: int = 64,
        n_layers:   int = 2,
        pred_len:   int = 14,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # bidirectional doubles the hidden size
        self.head = nn.Linear(hidden_dim * 2, pred_len)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, 1)
        t : ignored — present for API consistency with Transformer models

        Returns
        -------
        (batch, pred_len, 1)
        """
        out, _ = self.lstm(x)                        # (batch, seq_len, hidden*2)
        out = self.dropout(out[:, -1, :])            # take last timestep
        out = self.head(out)                         # (batch, pred_len)
        return out.unsqueeze(-1)                     # (batch, pred_len, 1)


# ---------------------------------------------------------------------------
# 3. Vanilla Transformer (fixed sinusoidal PE)
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Standard fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)                         # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VanillaTransformer(nn.Module):
    """
    Standard encoder-only Transformer for time series forecasting.
    Uses fixed sinusoidal positional encoding.

    This is the ablation baseline — identical to T2V_Transformer except
    for the temporal encoding layer.
    """

    def __init__(
        self,
        input_dim:  int = 1,
        d_model:    int = 64,
        n_heads:    int = 4,
        n_layers:   int = 2,
        d_ff:       int = 256,
        pred_len:   int = 14,
        seq_len:    int = 90,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len

        # Project scalar price to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = SinusoidalPE(d_model, max_len=seq_len + 10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN: more stable on small data
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, 1)   price values, normalised
        t : ignored

        Returns
        -------
        (batch, pred_len, 1)
        """
        h = self.input_proj(x)      # (batch, seq_len, d_model)
        h = self.pos_enc(h)
        h = self.encoder(h)         # (batch, seq_len, d_model)
        h = self.dropout(h[:, -1, :])   # CLS-like: use last token
        out = self.head(h)          # (batch, pred_len)
        return out.unsqueeze(-1)    # (batch, pred_len, 1)


# ---------------------------------------------------------------------------
# 4. Time2Vec Transformer (Kazemi et al., 2019 — ablation against fixed PE)
# ---------------------------------------------------------------------------

class T2V_Transformer(nn.Module):
    """
    Ablation model: Transformer with Time2Vec positional encoding (Kazemi et al., 2019).

    Evaluated against VanillaTransformer to isolate the contribution of learnable
    temporal encoding vs. fixed sinusoidal encoding.
    """

    def __init__(
        self,
        input_dim:  int = 1,
        d_model:    int = 64,
        n_heads:    int = 4,
        n_layers:   int = 2,
        d_ff:       int = 256,
        pred_len:   int = 14,
        t2v_dim:    int = 32,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len

        # Value embedding
        self.input_proj = nn.Linear(input_dim, d_model)

        # Time2Vec temporal encoding
        self.time2vec   = Time2Vec(out_dim=t2v_dim)
        self.t2v_proj   = nn.Linear(t2v_dim, d_model)   # project to d_model for addition

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, 1)   price values, normalised
        t : (batch, seq_len, 1)   time indices, normalised to [0, 1]

        Returns
        -------
        (batch, pred_len, 1)
        """
        # Value path
        val_emb = self.input_proj(x)            # (batch, seq_len, d_model)

        # Temporal path
        t2v_emb = self.time2vec(t)              # (batch, seq_len, t2v_dim)
        t2v_emb = self.t2v_proj(t2v_emb)        # (batch, seq_len, d_model)

        # Combine (additive, same as standard PE convention)
        h = self.dropout(val_emb + t2v_emb)     # (batch, seq_len, d_model)

        h = self.encoder(h)                     # (batch, seq_len, d_model)
        h = self.dropout(h[:, -1, :])           # last token
        out = self.head(h)                      # (batch, pred_len)
        return out.unsqueeze(-1)                # (batch, pred_len, 1)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(name: str, config: dict) -> nn.Module:
    """
    Convenience factory.

    Parameters
    ----------
    name   : one of 'bilstm', 'transformer', 't2v_transformer'
    config : dict of hyperparameters (pred_len, seq_len, d_model, ...)
    """
    models = {
        'bilstm':          BiLSTM,
        'transformer':     VanillaTransformer,
        't2v_transformer': T2V_Transformer,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(models.keys())}")
    return models[name](**{k: v for k, v in config.items()
                           if k in models[name].__init__.__code__.co_varnames})