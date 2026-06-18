# VMD-Informer: Offshore Wind Speed Forecasting

Multi-horizon wind speed forecasting for offshore environments using Variational Mode Decomposition (VMD) combined with deep learning models including a custom Informer architecture with ProbSparse attention. Multi-height offshore Doppler LIDAR wind speed dataset is used.

---

## Overview

Wind speed is decomposed into Intrinsic Mode Functions (IMFs) using VMD before being fed into sequence models. This decomposition reduces non-stationarity and makes the forecasting task easier for the model.

**Data:** Offshore LIDAR and buoy meteorological data from Morro Bay, California (DOE A2E dataset). 10-minute average data for all variables. A few days of sample data are included in `data_LIDAR/` and `data_BUOY/` for testing.  
**Forecast targets:** Wind speed at 6 heights (40–140 m)  
**Forecast horizons:** 36-step, 72-step, 144-step ahead  

```
Raw netCDF (LIDAR + Buoy)
        │
        ▼
raw_data_preprocessing.py  →  wind_data_cleaned.csv
                               met_data_merged.csv
        │
        ▼
VMD.py                     →  vmd_series/vmd_results_wspd{height}.npz
        │
        ▼
datagen.py                 →  Train / Val / Test PyTorch Datasets
        │
        ▼
hyperparam_optuna.py       →  outputs/optuna/best_params_{model}.json
        │
        ▼
train_final.py             →  outputs/checkpoints/best_{model}.pt
```

---

## Models

| Model | Description |
|---|---|
| `GRUForecaster` | Vanilla GRU with MLP head |
| `LSTMForecaster` | Vanilla LSTM with MLP head |
| `BiLSTMForecaster` | Bidirectional LSTM |
| `GRUEncoderDecoder` | GRU encoder-decoder with autoregressive decoding |
| `AttentionForecaster` | GRU encoder + multi-head attention |
| `TransformerForecaster` | Transformer encoder with learned positional encoding |
| `CNNLSTM` | 1D CNN feature extractor + LSTM |
| `GRUEncoderDecoderAttn` | GRU encoder-decoder with additive (Bahdanau) attention |
| `Informer` | ProbSparse self-attention + convolutional distilling (O(L log L)) |

---

## Project Structure

```
VMD_Informer_Git/
├── data_LIDAR/                  # LIDAR netCDF files (one per day)
├── data_BUOY/                   # Buoy netCDF files (one per day)
├── vmd_series/                  # VMD output (.npz per height)
├── outputs/
│   ├── optuna/                  # best_params_{model}.json
│   └── checkpoints/             # best_{model}.pt
├── raw_data_preprocessing.py    # netCDF → cleaned CSV
├── VMD.py                       # VMD decomposition + K selection
├── datagen.py                   # PyTorch Dataset builder
├── models.py                    # All model definitions
├── train.py                     # Generic training loop utility
├── hyperparam_optuna.py         # Optuna hyperparameter search
├── train_final.py               # Final training with best params
└── main.py                      # End-to-end pipeline
```

---

## Setup

```bash
conda create -n vmd-informer python=3.10
conda activate vmd-informer
pip install numpy pandas netCDF4 torch vmdpy scikit-learn optuna matplotlib
```

Place your data folders in the project root:
```
VMD_Informer_Git/
├── data_LIDAR/   ← LIDAR .nc files here
└── data_BUOY/    ← Buoy *10m.a2e.nc files here
```

---

## Usage

**Full pipeline (default: 100 m height, all models):**
```bash
python main.py
```

**Select height and models:**
```bash
python main.py --height 80 --models gru lstm informer
```

**Skip steps you have already run:**
```bash
# Skip preprocessing and VMD — use existing CSV and .npz files
python main.py --skip-preprocessing --skip-vmd

# Skip Optuna — use existing outputs/optuna/*.json
python main.py --skip-preprocessing --skip-vmd --skip-optuna
```

**Run steps individually:**
```bash
python raw_data_preprocessing.py
python VMD.py
python hyperparam_optuna.py
python train_final.py
```

After a full run, outputs are saved as:
```
outputs/
├── optuna/
│   ├── best_params_gru.json
│   ├── best_params_informer.json
│   └── ...
└── checkpoints/
    ├── best_gru.pt
    ├── best_informer.pt
    └── ...
```

---





## Data Source

LIDAR and buoy data from the [U.S. DOE Atmosphere to Electrons (A2E)](https://a2e.energy.gov/) program, Morro Bay, California offshore site.
