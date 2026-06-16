    
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import torch



def datagen(height: int):
    class VMDDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    VALID_HEIGHTS = {40, 60, 80, 100, 120, 140}

    if height not in VALID_HEIGHTS:
        raise ValueError(f"height must be one of {sorted(VALID_HEIGHTS)}, got {height}")
         

    data    = pd.read_csv('wind_data_cleaned.csv')
    loaded  = np.load(f'vmd_series/vmd_results_wspd{height}.npz')
    u       = loaded['u']

    K_BEST=14
    mode_cols = [f'mode_{i+1}' for i in range(K_BEST)]
    for i in range(K_BEST):
        data[mode_cols[i]] = u[i]

    GAP_THRESHOLD     = 30
    SEQ_LEN           = 144
    HORIZONS          = [36, 72, 144]
    max_h             = max(HORIZONS)
    vmd_feature_names = mode_cols

    X_vmd, y_vmd, y_times_vmd = [], [], []
    data['segment'] = (data['time_diff'] > GAP_THRESHOLD).cumsum()
    for _, segment in data.groupby('segment'):
        segment = segment.sort_index()
        if len(segment) < SEQ_LEN + max_h:
            continue
        values = segment[vmd_feature_names].values
        target = segment[f'wspd{height}'].values
        times  = segment.index.to_numpy()
        for i in range(0, len(segment) - SEQ_LEN - max_h + 1):
            X_vmd.append(values[i : i + SEQ_LEN])
            future_vals, future_times = [], []
            for h in HORIZONS:
                idx = i + SEQ_LEN + h - 1
                future_vals.append(target[idx])
                future_times.append(times[idx])
            y_vmd.append(future_vals)
            y_times_vmd.append(future_times)

    X_vmd       = np.array(X_vmd,       dtype=np.float32)
    y_vmd       = np.array(y_vmd,       dtype=np.float32)
    y_times_vmd = np.array(y_times_vmd)
    print(f"X_vmd: {X_vmd.shape} | y_vmd: {y_vmd.shape}")

    # Scale (fit on train only) 
    N        = len(X_vmd)
    n_train  = int(N * 0.70)                              

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_flat   = X_vmd.reshape(-1, X_vmd.shape[-1])
    scaler_X.fit(X_flat[:n_train * SEQ_LEN])              # fit on train only
    X_scaled = scaler_X.transform(X_flat).reshape(X_vmd.shape).astype(np.float32)

    scaler_y.fit(y_vmd[:n_train])
    y_scaled = scaler_y.transform(y_vmd).astype(np.float32)

    # joblib.dump(scaler_X, 'scaler_X.pkl')
    # joblib.dump(scaler_y, 'scaler_y.pkl')
    print(f"Scaling done — X: {X_scaled.shape} | y: {y_scaled.shape}")

    N       = len(X_scaled)
    n_train = int(N * 0.70)
    n_val   = int(N * 0.15)
    n_test  = N - n_train - n_val

    X_train, y_train = X_scaled[:n_train],                y_scaled[:n_train]
    X_val,   y_val   = X_scaled[n_train:n_train+n_val],   y_scaled[n_train:n_train+n_val]
    X_test,  y_test  = X_scaled[n_train+n_val:],          y_scaled[n_train+n_val:]
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_dataset = VMDDataset(X_train, y_train)
    val_dataset   = VMDDataset(X_val,   y_val)
    test_dataset  = VMDDataset(X_test,  y_test)

    return train_dataset, val_dataset, test_dataset