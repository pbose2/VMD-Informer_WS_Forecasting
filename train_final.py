import json
import os
import torch
from torch.utils.data import DataLoader
from datagen import dataset_gen
from train import train_model, plot_loss

from models import (
    GRUForecaster, LSTMForecaster, BiLSTMForecaster,
    GRUEncoderDecoder, AttentionForecaster, TransformerForecaster,
    CNNLSTM, GRUEncoderDecoderAttn, Informer
)

HEIGHT         = 100
EPOCHS         = 100
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OPTUNA_DIR     = 'outputs/optuna'
CHECKPOINT_DIR = 'outputs/checkpoints'


def load_params(name):
    path = os.path.join(OPTUNA_DIR, f'best_params_{name}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run hyperparam_optuna.py first — {path} not found.")
    with open(path) as f:
        return json.load(f)


def build_model(name, p, input_size, output_size, seq_len):
    if name == 'gru':
        return GRUForecaster(input_size, p['hidden_size'], p['num_layers'], output_size, p['dropout'])
    if name == 'lstm':
        return LSTMForecaster(input_size, p['hidden_size'], p['num_layers'], output_size, p['dropout'])
    if name == 'bilstm':
        return BiLSTMForecaster(input_size, p['hidden_size'], p['num_layers'], output_size, p['dropout'])
    if name == 'gruencdec':
        return GRUEncoderDecoder(input_size, p['hidden_size'], p['num_layers'], output_size, p['dropout'])
    if name == 'attn':
        return AttentionForecaster(input_size, p['hidden_size'], p['num_layers'],
                                   p['num_heads'], output_size, p['dropout'])
    if name == 'transformer':
        return TransformerForecaster(input_size, p['hidden_size'], p['num_layers'],
                                     p['num_heads'], output_size, p['dropout'], max_len=seq_len)
    if name == 'cnnlstm':
        return CNNLSTM(input_size, p['cnn_channels'], p['kernel_size'],
                       p['lstm_hidden'], p['lstm_layers'], output_size, p['dropout'])
    if name == 'gruattn':
        return GRUEncoderDecoderAttn(input_size, p['hidden_size'], p['num_layers'], output_size, p['dropout'])
    if name == 'informer':
        return Informer(input_size, p['hidden_size'], p['num_layers'], p['num_heads'],
                        output_size, p['dropout'], p['factor'], max_len=seq_len, distil=p['distil'])
    raise ValueError(f"Unknown model name: {name}")


def train_one(name, train_dataset, val_dataset):
    p          = load_params(name)
    input_size = train_dataset.X.shape[-1]
    output_size = train_dataset.y.shape[-1]
    seq_len    = train_dataset.X.shape[1]

    train_loader = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=p['batch_size'], shuffle=False)

    model = build_model(name, p, input_size, output_size, seq_len).to(DEVICE)
    print(f"\n{'='*60}\nTraining {name.upper()} with params: {p}\n{'='*60}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model, history = train_model(
        model, train_loader, val_loader,
        lr        = p['lr'],
        epochs    = EPOCHS,
        save_path = os.path.join(CHECKPOINT_DIR, f'best_{name}.pt'),
        device    = DEVICE
    )
    plot_loss(history, title=f'{name.upper()} Training')
    return model


if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = dataset_gen(height=HEIGHT)

    models_to_train = [
        'gru', 'lstm', 'bilstm', 'gruencdec', 'attn',
        'transformer', 'cnnlstm', 'gruattn', 'informer'
    ]

    for name in models_to_train:
        try:
            train_one(name, train_dataset, val_dataset)
        except FileNotFoundError as e:
            print(f"Skipping {name}: {e}")
