"""
Time2Vec: Learning a Vector Representation of Time
Kazemi et al. (2019) - https://arxiv.org/abs/1907.05321

Replaces fixed sinusoidal positional encoding with learnable periodic embeddings.
The key idea: one linear term captures trend, (k-1) sinusoidal terms capture
periodic patterns of different frequencies and phases.

For agricultural price data this is particularly appropriate because:
  - harvest cycles (annual)
  - Ramadan/Eid demand spikes (lunar, ~354-day cycle)
  - monsoon seasonality (~120-day cycle)
are all learnable from data rather than hardcoded.
"""

import torch
import torch.nn as nn
import math


class Time2Vec(nn.Module):
    """
    Time2Vec embedding layer.

    Given a scalar time index t, produces a k-dimensional vector:
        [w0*t + b0,                         <- linear (trend) term
         sin(w1*t + b1),                    <- periodic term 1
         sin(w2*t + b2),                    <- periodic term 2
         ...
         sin(w_{k-1}*t + b_{k-1})]         <- periodic term k-1

    Parameters
    ----------
    out_dim : int
        Total output dimension k. First dimension is linear, rest are sinusoidal.
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

        # Linear (trend) component
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1, 1))

        # Periodic components — shape (1, out_dim - 1)
        self.w = nn.Parameter(torch.randn(1, out_dim - 1))
        self.b = nn.Parameter(torch.randn(1, out_dim - 1))

        self._init_weights()

    def _init_weights(self):
        """
        Initialise periodic weights to cover a range of frequencies.
        Without this, all sinusoids start at similar frequencies and
        training is slow to discover distinct periodicities.
        """
        nn.init.uniform_(self.w0, -0.1, 0.1)
        nn.init.zeros_(self.b0)

        # Space initial frequencies logarithmically so short and long
        # cycles are both represented from the start.
        k = self.out_dim - 1
        if k > 0:
            freqs = torch.exp(
                torch.linspace(math.log(0.01), math.log(1.0), k)
            ).unsqueeze(0)          # (1, k)
            with torch.no_grad():
                self.w.copy_(freqs)
            nn.init.uniform_(self.b, -math.pi, math.pi)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : torch.Tensor  shape (batch, seq_len, 1)
            Normalised time indices in [0, 1].

        Returns
        -------
        torch.Tensor  shape (batch, seq_len, out_dim)
        """
        # Linear term: (batch, seq_len, 1)
        linear = self.w0 * t + self.b0

        # Periodic terms: broadcast over batch and seq_len
        # t: (batch, seq_len, 1), w/b: (1, out_dim-1) -> (batch, seq_len, out_dim-1)
        periodic = torch.sin(t * self.w + self.b)

        return torch.cat([linear, periodic], dim=-1)   # (batch, seq_len, out_dim)
