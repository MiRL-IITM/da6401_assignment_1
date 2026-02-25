"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .activations import get_activation, softmax
from .neural_layer import NeuralLayer
from .objective_functions import compute_loss, compute_output_gradient
from .optimizers import Optimizer


class NeuralNetwork:
    """Main model class that orchestrates training and inference."""

    def __init__(self, cli_args):
        np.random.seed(getattr(cli_args, "seed", 42))

        self.input_dim = 28 * 28
        self.output_dim = 10
        self.activation_name = cli_args.activation
        self.loss_name = cli_args.loss

        hidden_sizes = list(cli_args.hidden_size)
        if len(hidden_sizes) != cli_args.num_layers:
            raise ValueError("Length of --hidden_size must equal --num_layers")

        layer_dims = [self.input_dim] + hidden_sizes + [self.output_dim]
        self.layers = []
        for i in range(len(layer_dims) - 1):
            is_last = i == len(layer_dims) - 2
            activation = None if is_last else self.activation_name
            self.layers.append(
                NeuralLayer(
                    in_dim=layer_dims[i],
                    out_dim=layer_dims[i + 1],
                    activation=activation,
                    weight_init=cli_args.weight_init,
                )
            )

        self.hidden_activation, self.hidden_activation_derivative = get_activation(self.activation_name)
        self.optimizer = Optimizer(
            name=cli_args.optimizer,
            learning_rate=cli_args.learning_rate,
            weight_decay=cli_args.weight_decay,
            layers_count=len(self.layers),
        )

    def forward(self, X):
        a = X
        for layer in self.layers[:-1]:
            z = layer.forward_linear(a)
            a = self.hidden_activation(z)
            layer.a_cache = a
        logits = self.layers[-1].forward_linear(a)
        probs = softmax(logits)
        self.layers[-1].a_cache = probs
        return logits, probs

    def backward(self, y_true, y_pred_probs):
        grad = compute_output_gradient(self.loss_name, y_true, y_pred_probs)
        grad = self.layers[-1].backward_linear(grad)

        for layer in reversed(self.layers[:-1]):
            local_grad = grad * self.hidden_activation_derivative(layer.a_cache)
            grad = layer.backward_linear(local_grad)

    def update_weights(self):
        self.optimizer.step(self.layers)

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, log_callback=None):
        n_samples = X_train.shape[0]
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                xb = X_train[start:end]
                yb = y_train[start:end]

                _, probs = self.forward(xb)
                self.backward(yb, probs)
                self.update_weights()

            train_metrics = self.evaluate(X_train, y_train)
            val_metrics = self.evaluate(X_val, y_val)

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            epoch_payload = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            }
            if log_callback is not None:
                log_callback(epoch_payload)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
            )

        return history

    def evaluate(self, X, y):
        logits, probs = self.forward(X)
        loss = compute_loss(self.loss_name, y, probs)

        y_true = np.argmax(y, axis=1)
        y_pred = np.argmax(probs, axis=1)

        return {
            "logits": logits,
            "loss": float(loss),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def export_state(self):
        return {
            "weights": [layer.W for layer in self.layers],
            "biases": [layer.b for layer in self.layers],
            "activation": self.activation_name,
            "loss": self.loss_name,
        }

    def load_state(self, state):
        for layer, w, b in zip(self.layers, state["weights"], state["biases"]):
            layer.W = w
            layer.b = b
