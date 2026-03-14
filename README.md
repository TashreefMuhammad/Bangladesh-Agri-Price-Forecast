# Agricultural Price Forecasting: T2V-Transformer

Daily retail price forecasting for 5 Bangladeshi commodities using a novel
Time2Vec-Transformer architecture benchmarked against classical and deep learning baselines.

## Project Structure

```
Bangladesh-Agri-Price-Forecast/
├── models/
│   ├── time2vec.py          ← Time2Vec layer (Kazemi et al. 2019)
│   ├── architectures.py     ← BiLSTM, VanillaTransformer, T2V_Transformer, NaivePersistence
│   └── __init__.py
├── utils/
│   ├── data.py              ← Data loading, preprocessing, PyTorch Dataset
│   ├── train.py             ← Training loop, early stopping, metrics (MAE/RMSE/MAPE)
│   ├── baselines.py         ← SARIMA (auto_arima) + Facebook Prophet with BD holidays
│   ├── plots.py             ← Publication-quality figures
│   └── __init__.py
└── notebooks/
    └── main_experiment.ipynb ← Master Colab notebook — run this
```

## Commodities

| Commodity     | CSV path                                  |
|---------------|-------------------------------------------|
| Garlic        | `Dataset/Garlic/garlic.csv`               |
| Chickpea      | `Dataset/Chickpea/Chickpea.csv`           |
| Green Chilli  | `Dataset/Green_chilli/Green_chilli.csv`   |
| Cucumber      | `Dataset/Cucumber/Cucumber.csv`           |
| Sweet Pumpkin | `Dataset/Sweet_pumpkin/Sweet_pumpkin.csv` |

Each CSV requires columns: `date` (YYYY-MM-DD), `price` (BDT/kg)

## Models

| Model | Description | Role |
|-------|-------------|------|
| Naïve Persistence | Last value carried forward | Floor baseline |
| SARIMA | Auto-selected via pmdarima | Classical seasonal baseline |
| Prophet | With Bangladesh holidays (Ramadan, Eid) | Decomposition baseline |
| BiLSTM | 2-layer bidirectional LSTM | DL baseline |
| Transformer | Fixed sinusoidal PE, 2 encoder layers | Ablation baseline |
| **T2V-Transformer** | **Time2Vec PE, 2 encoder layers** | **Proposed model** |

## Ablation Design

The comparison between `Transformer` and `T2V-Transformer` is a controlled experiment:
**identical architecture**, only the temporal encoding changes. This cleanly isolates
the contribution of learnable periodic embeddings (Time2Vec) versus fixed sinusoidal PE.

## Quick Start (Colab)

1. Upload `agri_forecast/` and `Dataset/` to your Colab runtime
2. Open `notebooks/main_experiment.ipynb`
3. Run Cell 1 to install dependencies
4. Run all cells — expected runtime ~25 min on a T4 GPU

## Key Design Decisions

**Why not Informer?**
Informer's ProbSparse attention was designed for 17,000+ hourly sequences. On
~1,780 daily points it collapses to mean prediction (flat line). We note this
in Related Work as motivation for a lightweight architecture.

**Why Time2Vec + Transformer?**
Agricultural prices carry strong periodic signals: harvest cycles, Ramadan/Eid
demand spikes, monsoon-driven supply shocks. Time2Vec's learnable sinusoidal
components discover these from data rather than requiring manual feature engineering.
The combination is Time2Vec's native application (Kazemi et al. designed it as
a Transformer PE replacement).

**Why lightweight (d_model=64)?**
~1,780 training points. A large Transformer would overfit badly. The parameter
count comparison (Cell 12) shows T2V-Transformer is not winning due to size.

## Requirements

```
torch >= 2.0
statsmodels
pmdarima
prophet
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install: `pip install pmdarima prophet statsmodels scikit-learn seaborn`

## Citation

If you use this code, please cite:
```
[To be filled after submission]
```
