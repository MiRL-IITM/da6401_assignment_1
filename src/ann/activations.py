"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def sigmoid_derivative(activated_x: np.ndarray) -> np.ndarray:
    return activated_x * (1.0 - activated_x)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(activated_x: np.ndarray) -> np.ndarray:
    return 1.0 - (activated_x ** 2)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return relu, relu_derivative
    if name == "sigmoid":
        return sigmoid, sigmoid_derivative
    if name == "tanh":
        return tanh, tanh_derivative
    raise ValueError(f"Unsupported activation: {name}")
