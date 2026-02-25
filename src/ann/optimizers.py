"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(
        self,
        name: str,
        learning_rate: float,
        weight_decay: float,
        layers_count: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.name = name.lower()
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.v_w = [None] * layers_count
        self.v_b = [None] * layers_count
        self.s_w = [None] * layers_count
        self.s_b = [None] * layers_count

    def _regularized_grads(self, layer):
        grad_w = layer.grad_W + self.weight_decay * layer.W
        grad_b = layer.grad_b
        return grad_w, grad_b

    def step(self, layers: list):
        self.t += 1
        for idx, layer in enumerate(layers):
            grad_w, grad_b = self._regularized_grads(layer)

            if self.name == "sgd":
                layer.W -= self.lr * grad_w
                layer.b -= self.lr * grad_b
                continue

            if self.v_w[idx] is None:
                self.v_w[idx] = np.zeros_like(layer.W)
                self.v_b[idx] = np.zeros_like(layer.b)
                self.s_w[idx] = np.zeros_like(layer.W)
                self.s_b[idx] = np.zeros_like(layer.b)

            if self.name in {"momentum", "nag"}:
                self.v_w[idx] = self.beta1 * self.v_w[idx] + grad_w
                self.v_b[idx] = self.beta1 * self.v_b[idx] + grad_b
                if self.name == "nag":
                    layer.W -= self.lr * (self.beta1 * self.v_w[idx] + grad_w)
                    layer.b -= self.lr * (self.beta1 * self.v_b[idx] + grad_b)
                else:
                    layer.W -= self.lr * self.v_w[idx]
                    layer.b -= self.lr * self.v_b[idx]
                continue

            if self.name == "rmsprop":
                self.s_w[idx] = self.beta2 * self.s_w[idx] + (1.0 - self.beta2) * (grad_w ** 2)
                self.s_b[idx] = self.beta2 * self.s_b[idx] + (1.0 - self.beta2) * (grad_b ** 2)
                layer.W -= self.lr * grad_w / (np.sqrt(self.s_w[idx]) + self.epsilon)
                layer.b -= self.lr * grad_b / (np.sqrt(self.s_b[idx]) + self.epsilon)
                continue

            if self.name in {"adam", "nadam"}:
                self.v_w[idx] = self.beta1 * self.v_w[idx] + (1.0 - self.beta1) * grad_w
                self.v_b[idx] = self.beta1 * self.v_b[idx] + (1.0 - self.beta1) * grad_b
                self.s_w[idx] = self.beta2 * self.s_w[idx] + (1.0 - self.beta2) * (grad_w ** 2)
                self.s_b[idx] = self.beta2 * self.s_b[idx] + (1.0 - self.beta2) * (grad_b ** 2)

                v_w_hat = self.v_w[idx] / (1.0 - self.beta1 ** self.t)
                v_b_hat = self.v_b[idx] / (1.0 - self.beta1 ** self.t)
                s_w_hat = self.s_w[idx] / (1.0 - self.beta2 ** self.t)
                s_b_hat = self.s_b[idx] / (1.0 - self.beta2 ** self.t)

                if self.name == "nadam":
                    nesterov_w = self.beta1 * v_w_hat + ((1.0 - self.beta1) * grad_w) / (1.0 - self.beta1 ** self.t)
                    nesterov_b = self.beta1 * v_b_hat + ((1.0 - self.beta1) * grad_b) / (1.0 - self.beta1 ** self.t)
                    layer.W -= self.lr * nesterov_w / (np.sqrt(s_w_hat) + self.epsilon)
                    layer.b -= self.lr * nesterov_b / (np.sqrt(s_b_hat) + self.epsilon)
                else:
                    layer.W -= self.lr * v_w_hat / (np.sqrt(s_w_hat) + self.epsilon)
                    layer.b -= self.lr * v_b_hat / (np.sqrt(s_b_hat) + self.epsilon)
                continue

            raise ValueError(f"Unsupported optimizer: {self.name}")
