"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

from __future__ import annotations

import numpy as np


class NeuralLayer:
    """Fully-connected layer with optional activation."""

    def __init__(self, in_dim: int, out_dim: int, activation: str | None, weight_init: str):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_name = activation
        self.weight_init = weight_init

        self.W, self.b = self._init_params(in_dim, out_dim, weight_init)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self.input_cache = None
        self.z_cache = None
        self.a_cache = None

    @staticmethod
    def _init_params(in_dim: int, out_dim: int, method: str):
        method = method.lower()
        if method == "xavier":
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            W = np.random.uniform(-limit, limit, size=(in_dim, out_dim))
            b = np.zeros((1, out_dim))
            return W, b
        if method == "zeros":
            W = np.zeros((in_dim, out_dim))
            b = np.zeros((1, out_dim))
            return W, b
        # random
        W = np.random.randn(in_dim, out_dim) * 0.01
        b = np.zeros((1, out_dim))
        return W, b

    def forward_linear(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        self.z_cache = x @ self.W + self.b
        return self.z_cache

    def backward_linear(self, grad_out: np.ndarray) -> np.ndarray:
        batch_size = self.input_cache.shape[0]
        self.grad_W = (self.input_cache.T @ grad_out) / batch_size
        self.grad_b = np.sum(grad_out, axis=0, keepdims=True) / batch_size
        grad_input = grad_out @ self.W.T
        return grad_input
