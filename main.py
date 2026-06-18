import argparse
import subprocess
import sys

ALL_MODELS = ['gru', 'lstm', 'bilstm', 'gruencdec', 'attn',
              'transformer', 'cnnlstm', 'gruattn', 'informer']


def run_script(path):
    print(f"\n{'='*60}\nRunning: {path}\n{'='*60}")
    result = subprocess.run([sys.executable, path], check=False)
    if result.returncode != 0:
        print(f"ERROR: {path} failed (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)


def step_preprocess(args):
    if args.skip_preprocessing:
        print("[SKIP] Preprocessing")
        return
    run_script('raw_data_preprocessing.py')


def step_vmd(args):
    if args.skip_vmd:
        print("[SKIP] VMD decomposition")
        return
    run_script('VMD.py')


def step_optuna(args):
    if args.skip_optuna:
        print("[SKIP] Optuna hyperparameter search")
        return

    from hyperparam_optuna import _run_study
    from datagen import dataset_gen
    from hyperparam_optuna import (
        gru_objective, lstm_objective, bilstm_objective,
        gruencdec_objective, attn_objective, transformer_objective,
        cnnlstm_objective, gruattn_objective, informer_objective
    )

    objectives = {
        'gru':         gru_objective,
        'lstm':        lstm_objective,
        'bilstm':      bilstm_objective,
        'gruencdec':   gruencdec_objective,
        'attn':        attn_objective,
        'transformer': transformer_objective,
        'cnnlstm':     cnnlstm_objective,
        'gruattn':     gruattn_objective,
        'informer':    informer_objective,
    }

    print(f"\n{'='*60}\nOptuna search — height={args.height}\n{'='*60}")
    train_dataset, _, _ = dataset_gen(height=args.height)
    X = train_dataset.X.numpy()
    y = train_dataset.y.numpy()

    for name in args.models:
        _run_study(objectives[name], X, y, name)


def step_train(args):
    if args.skip_training:
        print("[SKIP] Final training")
        return

    from train_final import train_one
    from datagen import dataset_gen

    print(f"\n{'='*60}\nFinal training — height={args.height}\n{'='*60}")
    train_dataset, val_dataset, _ = dataset_gen(height=args.height)

    for name in args.models:
        try:
            train_one(name, train_dataset, val_dataset)
        except FileNotFoundError as e:
            print(f"[SKIP] {name}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description='VMD-Informer pipeline')
    parser.add_argument('--height', type=int, default=100,
                        choices=[40, 60, 80, 100, 120, 140],
                        help='Wind height to forecast in metres (default: 100)')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help='Models to run (default: all)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip raw_data_preprocessing.py (use existing CSVs)')
    parser.add_argument('--skip-vmd',           action='store_true',
                        help='Skip VMD.py (use existing vmd_series/*.npz)')
    parser.add_argument('--skip-optuna',        action='store_true',
                        help='Skip Optuna (use existing outputs/optuna/*.json)')
    parser.add_argument('--skip-training',      action='store_true',
                        help='Skip final training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Models : {args.models}")
    print(f"Height : {args.height} m")

    step_preprocess(args)
    step_vmd(args)
    step_optuna(args)
    step_train(args)

    print("\nPipeline complete.")
    print(f"  Hyperparameters : outputs/optuna/best_params_{{model}}.json")
    print(f"  Model weights   : outputs/checkpoints/best_{{model}}.pt")
