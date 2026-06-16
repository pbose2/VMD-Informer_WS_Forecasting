import json
import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader
from datagen import dataset_gen, VMDDataset
from models import (
    GRUForecaster, LSTMForecaster, BiLSTMForecaster,
    GRUEncoderDecoder, AttentionForecaster, TransformerForecaster,
    CNNLSTM, GRUEncoderDecoderAttn, Informer
)

N_TRIALS = 50
TIMEOUT  = 600
OPT_EPOCHS = 30


def _trial_train_loop(model, train_loader, val_loader, lr, trial, device):
    """Short training loop used inside every Optuna objective."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(OPT_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


def _make_loaders(X, y, batch_size):
    N       = len(X)
    n_train = int(N * 0.70)
    n_val   = int(N * 0.15)
    train_ds = VMDDataset(X[:n_train],               y[:n_train])
    val_ds   = VMDDataset(X[n_train:n_train + n_val], y[n_train:n_train + n_val])
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False))


def _run_study(objective_fn, X, y, name):
    study = optuna.create_study(
        direction = 'minimize',
        sampler   = optuna.samplers.TPESampler(seed=42),
        pruner    = optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(lambda trial: objective_fn(trial, X, y), n_trials=N_TRIALS, timeout=TIMEOUT)
    print(f"\n[{name}] Best val loss: {study.best_value:.6f}")
    print(f"[{name}] Best params:   {study.best_params}")
    with open(f'best_params_{name}.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    return study


# ── Objectives ────────────────────────────────────────────────────────────────

def gru_objective(trial, X, y):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
    num_layers  = trial.suggest_int('num_layers', 1, 4)
    dropout     = trial.suggest_float('dropout', 0.0, 0.5)
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = _make_loaders(X, y, batch_size)
    model = GRUForecaster(X.shape[-1], hidden_size, num_layers, y.shape[-1], dropout).to(device)
    return _trial_train_loop(model, train_loader, val_loader, lr, trial, device)


def lstm_objective(trial, X, y):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers  = trial.suggest_int('num_layers', 1, 3)
    dropout     = trial.suggest_float('dropout', 0.0, 0.5)
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = _make_loaders(X, y, batch_size)
    model = LSTMForecaster(X.shape[-1], hidden_size, num_layers, y.shape[-1], dropout).to(device)
    return _trial_train_loop(model, train_loader, val_loader, lr, trial, device)


def bilstm_objective(trial, X, y):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers  = trial.suggest_int('num_layers', 1, 3)
    dropout     = trial.suggest_float('dropout', 0.0, 0.5)
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = _make_loaders(X, y, batch_size)
    model = BiLSTMForecaster(X.shape[-1], hidden_size, num_layers, y.shape[-1], dropout).to(device)
    return _trial_train_loop(model, train_loader, val_loader, lr, trial, device)


def gruencdec_objective(trial, X, y):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
    num_layers  = trial.suggest_int('num_layers', 1, 4)
    dropout     = trial.suggest_float('dropout', 0.0, 0.5)
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = _make_loaders(X, y, batch_size)
    model = GRUEncoderDecoder(X.shape[-1], hidden_size, num_layers, y.shape[-1], dropout).to(device)
    return _trial_train_loop(model, train_loader, val_loader, lr, trial, device)


def attn_objective(trial, X, y):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers  = trial.suggest_int('num_layers', 1, 4)
    dropout     = trial.suggest_float('dropout', 0.0, 0.5)
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    num_heads   = trial.suggest_categorical('num_heads', [h for h in [1, 2, 4, 8] if hidden_size % h == 0])
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = _make_loaders(X, y, batch_size)
    model = AttentionForecaster(X.shape[-1], hidden_size, num_layers, num_heads, y.shape[-1], dropout).to(device)
    return _trial_train_loop(model, train_loader, val_loader, lr, trial, device)


def transformer_objective(trial, X, y):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers  = trial.suggest_int('num_layers', 1, 4)
    dropout     = trial.suggest_float('dropout', 0.0, 0.5)
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    num_heads   = trial.suggest_categorical('num_heads', [h for h in [1, 2, 4, 8] if hidden_size % h == 0])
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = _make_loaders(X, y, batch_size)
    model = TransformerForecaster(X.shape[-1], hidden_size, num_layers, num_heads,
                                  y.shape[-1], dropout, max_len=X.shape[1]).to(device)
    return _trial_train_loop(model, train_loader, val_loader, lr, trial, device)


def cnnlstm_objective(trial, X, y):
    cnn_channels = trial.suggest_int('cnn_channels', 16, 128)
    kernel_size  = trial.suggest_int('kernel_size', 2, 5)
    lstm_hidden  = trial.suggest_int('lstm_hidden', 32, 256, step=32)
    lstm_layers  = trial.suggest_int('lstm_layers', 1, 3)
    dropout      = trial.suggest_float('dropout', 0.0, 0.5)
    lr           = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = _make_loaders(X, y, batch_size)
    model = CNNLSTM(X.shape[-1], cnn_channels, kernel_size, lstm_hidden,
                    lstm_layers, y.shape[-1], dropout).to(device)
    return _trial_train_loop(model, train_loader, val_loader, lr, trial, device)


def gruattn_objective(trial, X, y):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
    num_layers  = trial.suggest_int('num_layers', 1, 4)
    dropout     = trial.suggest_float('dropout', 0.0, 0.5)
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = _make_loaders(X, y, batch_size)
    model = GRUEncoderDecoderAttn(X.shape[-1], hidden_size, num_layers, y.shape[-1], dropout).to(device)
    return _trial_train_loop(model, train_loader, val_loader, lr, trial, device)


def informer_objective(trial, X, y):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers  = trial.suggest_int('num_layers', 2, 4)
    dropout     = trial.suggest_float('dropout', 0.0, 0.5)
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    factor      = trial.suggest_int('factor', 3, 10)
    distil      = trial.suggest_categorical('distil', [True, False])
    num_heads   = trial.suggest_categorical('num_heads', [h for h in [1, 2, 4, 8] if hidden_size % h == 0])
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = _make_loaders(X, y, batch_size)
    model = Informer(X.shape[-1], hidden_size, num_layers, num_heads, y.shape[-1],
                     dropout, factor, max_len=X.shape[1], distil=distil).to(device)
    return _trial_train_loop(model, train_loader, val_loader, lr, trial, device)


# ── Run all studies ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = dataset_gen(height=100)

    # Pull numpy arrays back out for Optuna (which builds its own splits internally)
    X = train_dataset.X.numpy()
    y = train_dataset.y.numpy()

    _run_study(gru_objective,         X, y, 'gru')
    _run_study(lstm_objective,        X, y, 'lstm')
    _run_study(bilstm_objective,      X, y, 'bilstm')
    _run_study(gruencdec_objective,   X, y, 'gruencdec')
    _run_study(attn_objective,        X, y, 'attn')
    _run_study(transformer_objective, X, y, 'transformer')
    _run_study(cnnlstm_objective,     X, y, 'cnnlstm')
    _run_study(gruattn_objective,     X, y, 'gruattn')
    _run_study(informer_objective,    X, y, 'informer')
