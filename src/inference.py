"""
Inference Script
Evaluate trained models on test sets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ann.neural_network import NeuralNetwork


class InferenceArgs:
    def __init__(self, config: dict):
        self.activation = config["activation"]
        self.loss = config["loss"]
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        self.weight_init = config.get("weight_init", "xavier")
        self.optimizer = config.get("optimizer", "adam")
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.weight_decay = config.get("weight_decay", 0.0)
        self.seed = config.get("seed", 42)


def parse_arguments():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument("--model_path", required=True, help="Path to saved model weights (.npy), relative path only")
    parser.add_argument("--config_path", default="", help="Optional path to config.json")
    parser.add_argument("-d", "--dataset", required=True, choices=["mnist", "fashion", "fashion_mnist"])
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    return parser.parse_args()


def _load_config(args):
    if args.config_path:
        with open(args.config_path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    state = np.load(args.model_path, allow_pickle=True).item()
    return {
        "activation": state["activation"],
        "loss": state["loss"],
        "num_layers": state["num_layers"],
        "hidden_size": state["hidden_size"],
        "weight_init": "xavier",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "seed": 42,
    }


def load_model(model_path, config):
    """Load trained model from disk."""
    model_file = Path(model_path)
    if model_file.is_absolute():
        raise ValueError("Please provide a relative model path, not an absolute path.")

    state = np.load(model_file, allow_pickle=True).item()
    model = NeuralNetwork(InferenceArgs(config))
    model.load_state(state)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data."""
    return model.evaluate(X_test, y_test)


def main():
    """Main inference function."""
    args = parse_arguments()
    config = _load_config(args)
    model = load_model(args.model_path, config)
    from utils.data_loader import load_dataset

    data = load_dataset(args.dataset)

    metrics = evaluate_model(model, data["X_test"], data["y_test"])
    summary = {k: v for k, v in metrics.items() if k != "logits"}

    print("Inference metrics:")
    print(json.dumps(summary, indent=2))
    return metrics


if __name__ == "__main__":
    main()
