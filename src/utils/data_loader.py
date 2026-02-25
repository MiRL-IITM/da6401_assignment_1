"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

from __future__ import annotations

import numpy as np
from keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split


def _one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    y = y.astype(int)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def load_dataset(dataset_name: str, val_size: float = 0.1, seed: int = 42):
    dataset_name = dataset_name.lower()
    if dataset_name in {"mnist"}:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name in {"fashion", "fashion_mnist"}:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("dataset must be one of: mnist, fashion, fashion_mnist")

    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float64) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float64) / 255.0

    y_train_oh = _one_hot(y_train)
    y_test_oh = _one_hot(y_test)

    X_train, X_val, y_train_oh, y_val_oh = train_test_split(
        X_train,
        y_train_oh,
        test_size=val_size,
        random_state=seed,
        stratify=y_train,
    )

    return {
        "X_train": X_train,
        "y_train": y_train_oh,
        "X_val": X_val,
        "y_val": y_val_oh,
        "X_test": X_test,
        "y_test": y_test_oh,
    }
