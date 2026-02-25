"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


try:
    import wandb
except Exception:  # optional runtime integration
    wandb = None

from ann.neural_network import NeuralNetwork


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("-d", "--dataset", required=True, choices=["mnist", "fashion", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-o", "--optimizer", default="adam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=2)
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, default=[128, 64])
    parser.add_argument("-a", "--activation", default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l", "--loss", default="cross_entropy", choices=["cross_entropy", "mse"])
    parser.add_argument("-wi", "--weight_init", default="xavier", choices=["random", "xavier", "zeros"])
    parser.add_argument("--wandb_project", default="da6401_assignment_1")
    parser.add_argument("--model_save_path", default="artifacts/best_model.npy")
    parser.add_argument("--config_save_path", default="artifacts/config.json")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _ensure_relative_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        raise ValueError("Please provide a relative path, not an absolute path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main():
    """Main training function."""
    args = parse_arguments()

    from utils.data_loader import load_dataset

    data = load_dataset(args.dataset, seed=args.seed)
    model = NeuralNetwork(args)

    run = None
    if wandb is not None:
        run = wandb.init(project=args.wandb_project, config=vars(args), reinit=True)

    history = model.train(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_callback=(wandb.log if run is not None else None),
    )

    test_metrics = model.evaluate(data["X_test"], data["y_test"])

    model_path = _ensure_relative_path(args.model_save_path)
    config_path = _ensure_relative_path(args.config_save_path)

    state = model.export_state()
    state["num_layers"] = args.num_layers
    state["hidden_size"] = list(args.hidden_size)
    np.save(model_path, state, allow_pickle=True)

    config = vars(args).copy()
    config["best_test_metrics"] = {k: v for k, v in test_metrics.items() if k != "logits"}
    config["training_history"] = history
    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved config: {config_path}")
    print(
        f"Test metrics | loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f} "
        f"f1={test_metrics['f1']:.4f} precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f}"
    )

    if run is not None:
        wandb.log({f"test_{k}": v for k, v in test_metrics.items() if k != "logits"})
        run.finish()


if __name__ == "__main__":
    main()
