"""
Classical baseline models: SARIMA and Facebook Prophet.

Both are fit per-commodity on the training set only.
Evaluation is done via a rolling one-step-ahead forecast
over the test period, which is the most rigorous evaluation
protocol for classical time series models.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from .train import compute_metrics


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------

def run_sarima(
    train_df:   pd.DataFrame,
    test_df:    pd.DataFrame,
    pred_len:   int = 14,
    order:      tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 0, 7),   # weekly seasonality default
    auto_order: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Fit SARIMA on training data, forecast over test period.

    If auto_order=True, uses pmdarima.auto_arima to find optimal (p,d,q)(P,D,Q,m).
    Otherwise uses the provided order arguments.

    Returns
    -------
    preds       : np.ndarray of predictions
    truth       : np.ndarray of actual values
    metrics     : dict with MAE, RMSE, MAPE
    """
    try:
        import pmdarima as pm
        _has_pmdarima = True
    except ImportError:
        _has_pmdarima = False

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train_vals = train_df['price'].values
    test_vals  = test_df['price'].values

    if auto_order and _has_pmdarima:
        print("  Running auto_arima (this may take ~2-3 minutes)...")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            auto_model = pm.auto_arima(
                train_vals,
                seasonal=True, m=7,
                stepwise=True, information_criterion='aic',
                max_p=3, max_q=3, max_P=2, max_Q=2,
                suppress_warnings=True, error_action='ignore',
            )
        order          = auto_model.order
        seasonal_order = auto_model.seasonal_order
        print(f"  Best SARIMA order: {order} x {seasonal_order}")
    else:
        print(f"  Using SARIMA{order}x{seasonal_order}")

    # Fit on training data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = SARIMAX(
            train_vals,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    # Rolling forecast over test set
    # For each window, forecast pred_len steps ahead
    history = list(train_vals)
    preds, truth = [], []

    n_test = len(test_vals)
    step = pred_len   # non-overlapping windows

    for start in range(0, n_test - pred_len + 1, step):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fc = model.apply(history).forecast(pred_len)
        preds.append(fc)
        truth.append(test_vals[start: start + pred_len])
        # Update history with actual observations
        history.extend(test_vals[start: start + pred_len].tolist())
        # Refit model on expanded history every window (improves accuracy)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = SARIMAX(
                history,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

    preds = np.concatenate(preds)
    truth = np.concatenate(truth)
    metrics = compute_metrics(truth, preds)
    return preds, truth, metrics


# ---------------------------------------------------------------------------
# Facebook Prophet
# ---------------------------------------------------------------------------

def run_prophet(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    pred_len: int = 14,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Fit Facebook Prophet on training data, forecast over test period.

    Prophet is particularly relevant for this domain because:
      - it handles missing days gracefully (already forward-filled here)
      - it has explicit yearly and weekly seasonality components
      - it can model holiday effects (Ramadan/Eid can be added as custom holidays)

    Returns
    -------
    preds, truth, metrics
    """
    try:
        from prophet import Prophet
    except ImportError:
        from fbprophet import Prophet

    # Prophet expects columns: ds (datetime), y (value)
    train_prophet = train_df.reset_index().rename(
        columns={'date': 'ds', 'price': 'y'}
    )
    test_prophet = test_df.reset_index().rename(
        columns={'date': 'ds', 'price': 'y'}
    )

    # Bangladesh-specific holidays (major demand spikes for food commodities)
    # These shift garlic/chickpea/chilli demand significantly
    bd_holidays = make_bangladesh_holidays(
        train_df.index.min().year,
        test_df.index.max().year
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=bd_holidays,
        changepoint_prior_scale=0.05,    # conservative — avoids overfitting on ~1400 pts
        seasonality_prior_scale=10.0,
    )
    model.fit(train_prophet)

    # Forecast over entire test period
    future = model.make_future_dataframe(
        periods=len(test_df), freq='D', include_history=False
    )
    # Use actual test dates
    future['ds'] = test_df.index

    forecast = model.predict(future)
    preds = forecast['yhat'].values
    truth = test_df['price'].values[:len(preds)]

    # Clip predictions to non-negative (prices can't be negative)
    preds = np.clip(preds, 0, None)

    metrics = compute_metrics(truth, preds)
    return preds, truth, metrics


def make_bangladesh_holidays(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Approximate Ramadan start dates and Eid holidays for Bangladesh.
    These cause significant demand spikes for garlic, chickpea, and chilli.

    Dates are approximate — lunar calendar shifts ~11 days earlier each year.
    """
    # Approximate Ramadan start dates (1st day of Ramadan, Gregorian)
    ramadan_starts = {
        2020: '2020-04-24', 2021: '2021-04-13', 2022: '2022-04-02',
        2023: '2023-03-23', 2024: '2024-03-11', 2025: '2025-03-01',
    }
    eid_ul_fitr = {
        2020: '2020-05-24', 2021: '2021-05-13', 2022: '2022-05-02',
        2023: '2023-04-21', 2024: '2024-04-10', 2025: '2025-03-31',
    }
    eid_ul_adha = {
        2020: '2020-07-31', 2021: '2021-07-20', 2022: '2022-07-09',
        2023: '2023-06-28', 2024: '2024-06-17', 2025: '2025-06-07',
    }

    rows = []
    for year in range(start_year, end_year + 1):
        if year in ramadan_starts:
            rows.append({'holiday': 'Ramadan',    'ds': ramadan_starts[year], 'lower_window': 0, 'upper_window': 29})
        if year in eid_ul_fitr:
            rows.append({'holiday': 'Eid_ul_Fitr', 'ds': eid_ul_fitr[year],  'lower_window': -3, 'upper_window': 3})
        if year in eid_ul_adha:
            rows.append({'holiday': 'Eid_ul_Adha', 'ds': eid_ul_adha[year],  'lower_window': -3, 'upper_window': 3})

    if not rows:
        return pd.DataFrame(columns=['holiday', 'ds', 'lower_window', 'upper_window'])

    df = pd.DataFrame(rows)
    df['ds'] = pd.to_datetime(df['ds'])
    return df
