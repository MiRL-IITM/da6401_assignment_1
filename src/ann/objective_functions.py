"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

from __future__ import annotations

import numpy as np


def cross_entropy_loss(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray) -> float:
    eps = 1e-12
    clipped = np.clip(y_pred_probs, eps, 1.0 - eps)
    return float(-np.mean(np.sum(y_true_onehot * np.log(clipped), axis=1)))


def cross_entropy_with_softmax_gradient(
    y_true_onehot: np.ndarray, y_pred_probs: np.ndarray
) -> np.ndarray:
    batch_size = y_true_onehot.shape[0]
    return (y_pred_probs - y_true_onehot) / batch_size


def mse_loss(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray) -> float:
    return float(np.mean((y_true_onehot - y_pred_probs) ** 2))


def mse_gradient(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray) -> np.ndarray:
    batch_size = y_true_onehot.shape[0]
    return 2.0 * (y_pred_probs - y_true_onehot) / batch_size


def compute_loss(loss_name: str, y_true_onehot: np.ndarray, y_pred_probs: np.ndarray) -> float:
    if loss_name == "cross_entropy":
        return cross_entropy_loss(y_true_onehot, y_pred_probs)
    if loss_name == "mse":
        return mse_loss(y_true_onehot, y_pred_probs)
    raise ValueError(f"Unsupported loss: {loss_name}")


def compute_output_gradient(
    loss_name: str, y_true_onehot: np.ndarray, y_pred_probs: np.ndarray
) -> np.ndarray:
    if loss_name == "cross_entropy":
        return cross_entropy_with_softmax_gradient(y_true_onehot, y_pred_probs)
    if loss_name == "mse":
        return mse_gradient(y_true_onehot, y_pred_probs)
    raise ValueError(f"Unsupported loss: {loss_name}")
