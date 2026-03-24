"""
Publication-quality plotting utilities.

All figures follow a consistent style suitable for journal submission:
  - 300 DPI
  - Clean whitegrid style
  - Colorblind-friendly palette
  - Proper axis labels with units
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional

# Journal-quality defaults
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'legend.fontsize':  10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'figure.dpi':       150,     # screen preview; save at 300
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
})

# Colorblind-safe palette
PALETTE = {
    'truth':        '#2d3436',
    'naive':        '#b2bec3',
    'sarima':       '#0984e3',
    'prophet':      '#00b894',
    'bilstm':       '#e17055',
    'transformer':  '#6c5ce7',
    't2v_transformer': '#d63031',
}

COMMODITY_LABELS = {
    'garlic':        'Garlic',
    'chickpea':      'Chickpea',
    'green_chilli':  'Green Chilli',
    'cucumber':      'Cucumber',
    'sweet_pumpkin': 'Sweet Pumpkin',
}


# ---------------------------------------------------------------------------
# 1. Forecast comparison plot (one commodity, all models)
# ---------------------------------------------------------------------------

def plot_forecasts(
    dates:      pd.DatetimeIndex,
    truth:      np.ndarray,
    forecasts:  Dict[str, np.ndarray],
    commodity:  str,
    pred_len:   int = 14,
    save_path:  Optional[str] = None,
):
    """
    Plot ground truth vs. all model forecasts for one commodity.
    Uses the last `4 * pred_len` days of context for visual clarity.
    """
    n = min(len(truth), 4 * pred_len)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(dates[-n:], truth[-n:],
            color=PALETTE['truth'], linewidth=1.8,
            label='Ground Truth', zorder=5)

    for model_name, preds in forecasts.items():
        if preds is None or len(preds) == 0:
            continue
        color = PALETTE.get(model_name, '#fdcb6e')
        ax.plot(dates[-n:][:len(preds[-n:])], preds[-n:],
                color=color, linewidth=1.2, linestyle='--',
                label=_model_label(model_name), alpha=0.85)

    ax.set_title(f'{COMMODITY_LABELS.get(commodity, commodity)} Price Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (BDT/kg)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Metrics summary table (all commodities × all models)
# ---------------------------------------------------------------------------

def build_results_table(results: Dict) -> pd.DataFrame:
    """
    Build a publication-ready results table.

    Parameters
    ----------
    results : nested dict  {commodity: {model: {MAE, RMSE, MAPE}}}

    Returns
    -------
    pd.DataFrame with MultiIndex columns (model, metric)
    """
    records = []
    for commodity, model_results in results.items():
        row = {'Commodity': COMMODITY_LABELS.get(commodity, commodity)}
        for model, metrics in model_results.items():
            for metric, value in metrics.items():
                row[f'{_model_label(model)}_{metric}'] = round(value, 3)
        records.append(row)

    df = pd.DataFrame(records).set_index('Commodity')
    return df


def print_results_table(results: Dict):
    """Pretty-print the results table to console."""
    df = build_results_table(results)
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string())
    print("="*80 + "\n")


# ---------------------------------------------------------------------------
# 3. Training loss curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    histories:  Dict[str, Dict],
    commodity:  str,
    save_path:  Optional[str] = None,
):
    """Plot train/val loss curves for all DL models."""
    dl_models = ['bilstm', 'transformer', 't2v_transformer']
    fig, axes = plt.subplots(1, len(dl_models), figsize=(12, 3.5), sharey=False)

    for ax, model_name in zip(axes, dl_models):
        if model_name not in histories:
            ax.set_visible(False)
            continue
        h = histories[model_name]
        epochs = range(1, len(h['train_loss']) + 1)
        ax.plot(epochs, h['train_loss'], label='Train', color='#2d3436', linewidth=1.2)
        ax.plot(epochs, h['val_loss'],   label='Val',   color='#e17055', linewidth=1.2)
        ax.set_title(_model_label(model_name))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Huber Loss')
        ax.legend()
        ax.grid(alpha=0.3)
        sns.despine(ax=ax)

    fig.suptitle(f'Training Curves — {COMMODITY_LABELS.get(commodity, commodity)}',
                 fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Ablation bar chart (BiLSTM vs Transformer vs T2V_Transformer)
# ---------------------------------------------------------------------------

def plot_ablation(
    results:    Dict,
    metric:     str = 'RMSE',
    save_path:  Optional[str] = None,
):
    """
    Bar chart comparing BiLSTM, Transformer, T2V_Transformer across all commodities.
    Used in the ablation section of the paper.
    """
    ablation_models = ['bilstm', 'transformer', 't2v_transformer']
    commodities = list(results.keys())
    n_commodities = len(commodities)
    n_models = len(ablation_models)

    x = np.arange(n_commodities)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, model_name in enumerate(ablation_models):
        values = []
        for commodity in commodities:
            v = results[commodity].get(model_name, {}).get(metric, np.nan)
            values.append(v)
        bars = ax.bar(x + i * width, values, width,
                      label=_model_label(model_name),
                      color=PALETTE.get(model_name, '#fdcb6e'),
                      alpha=0.85, edgecolor='white')
        # Value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [COMMODITY_LABELS.get(c, c) for c in commodities],
        rotation=15, ha='right'
    )
    ax.set_ylabel(metric)
    ax.set_title(f'Ablation Study: Temporal Encoding ({metric})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Seasonality decomposition (for data analysis section)
# ---------------------------------------------------------------------------

def plot_decomposition(df: pd.DataFrame, commodity: str, save_path: Optional[str] = None):
    """STL decomposition plot for a single commodity."""
    from statsmodels.tsa.seasonal import STL

    stl = STL(df['price'], period=365, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    components = [
        (df['price'],   'Observed'),
        (res.trend,     'Trend'),
        (res.seasonal,  'Seasonal'),
        (res.resid,     'Residual'),
    ]
    for ax, (series, label) in zip(axes, components):
        ax.plot(df.index, series, linewidth=0.8, color='#2d3436')
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.2)
        sns.despine(ax=ax)

    axes[0].set_title(
        f'STL Decomposition — {COMMODITY_LABELS.get(commodity, commodity)} Price (BDT/kg)'
    )
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _model_label(model_name: str) -> str:
    labels = {
        'naive':           'Naïve Persistence',
        'sarima':          'SARIMA',
        'prophet':         'Prophet',
        'bilstm':          'BiLSTM',
        'transformer':     'Transformer',
        't2v_transformer': 'T2V-Transformer',
    }
    return labels.get(model_name, model_name)