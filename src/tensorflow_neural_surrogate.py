"""
TensorFlow/Keras neural-network surrogate candidate generator for black-box optimisation.

This module is intended to be used as an extra candidate source beside the
Gaussian Process ensemble. It trains a small TensorFlow/Keras MLP ensemble on
observed samples, then uses GradientTape with respect to the input x to perform
bounded gradient ascent inside the unit hypercube.

Typical notebook usage:

    from src.tensorflow_neural_surrogate import TensorFlowNeuralSurrogateCandidateGenerator

    nn_generator = TensorFlowNeuralSurrogateCandidateGenerator(
        n_ensemble=5,
        hidden_units=32,
        learning_rate=1e-3,
        max_epochs=1500,
        random_state=0,
    )

    nn_report = nn_generator.suggest(
        X=X,
        y=y,
        candidates=candidates,
        best_gp_input=best_input,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - useful message for notebooks
    raise ImportError(
        "tensorflow_neural_surrogate.py requires TensorFlow. Install tensorflow or skip the NN section."
    ) from exc


def _make_model(
    n_dimensions: int,
    hidden_units: int = 32,
    activation: str = "tanh",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
) -> tf.keras.Model:
    """Create and compile a small Keras MLP regressor."""
    regularizer = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None

    inputs = tf.keras.Input(shape=(n_dimensions,), dtype=tf.float32)
    x = tf.keras.layers.Dense(
        hidden_units,
        activation=activation,
        kernel_regularizer=regularizer,
    )(inputs)
    x = tf.keras.layers.Dense(
        hidden_units,
        activation=activation,
        kernel_regularizer=regularizer,
    )(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


@dataclass
class TensorFlowNeuralSurrogateCandidateGenerator:
    """
    Train a small TensorFlow NN ensemble and optimise candidates by input-gradient ascent.

    The defaults are deliberately conservative because the capstone datasets are
    small. Treat the resulting point as an extra candidate to compare against
    the GP/SVM pipeline, not as a replacement for it.
    """

    n_ensemble: int = 5 # number of small Keras MLP regressors to train in the ensemble
    hidden_units: int = 32 # number of hidden units in each layer of the MLP
    activation: str = "tanh" # activation function for the hidden layers
    learning_rate: float = 1e-3 # learning rate for the Adam optimizer
    weight_decay: float = 1e-3 # L2 regularization factor
    max_epochs: int = 1500 # maximum number of training epochs for each MLP
    patience: int = 200 # early stopping patience for training each MLP
    gradient_steps: int = 80 # number of gradient ascent steps to perform from each starting point
    gradient_step_size: float = 0.03 # step size for gradient ascent
    random_state: int = 0 # random seed for reproducibility
    verbose: bool = False # whether to print training progress for each MLP

    def __post_init__(self) -> None:
        self.models_: List[tf.keras.Model] = []
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.n_dimensions_: Optional[int] = None

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_dimensions).")
        return X

    @staticmethod
    def _as_1d(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).ravel()
        if y.ndim != 1:
            raise ValueError("y must be a 1D array or flattenable to 1D.")
        return y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TensorFlowNeuralSurrogateCandidateGenerator":
        """Fit an ensemble of small Keras MLP regressors."""
        X = self._as_2d(X)
        y = self._as_1d(y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if len(X) < 4:
            raise ValueError("At least 4 samples are recommended for the NN surrogate.")

        self.n_dimensions_ = X.shape[1]
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y) + 1e-8)
        y_scaled = ((y - self.y_mean_) / self.y_std_).astype(np.float32)

        self.models_ = []

        for model_idx in range(self.n_ensemble):
            seed = int(self.random_state + model_idx)
            np.random.seed(seed)
            tf.keras.utils.set_random_seed(seed)

            model = _make_model(
                n_dimensions=X.shape[1],
                hidden_units=self.hidden_units,
                activation=self.activation,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=self.patience,
                    min_delta=1e-7,
                    restore_best_weights=True,
                )
            ]

            history = model.fit(
                X,
                y_scaled,
                epochs=self.max_epochs,
                batch_size=len(X),
                shuffle=True,
                verbose=0,
                callbacks=callbacks,
            )

            self.models_.append(model)

            if self.verbose:
                final_loss = float(history.history["loss"][-1])
                print(f"TF NN model {model_idx + 1}/{self.n_ensemble}: train MSE={final_loss:.6g}")

        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        """Predict mean and ensemble standard deviation in original y units."""
        if not self.models_:
            raise RuntimeError("Call fit before predict.")

        X = self._as_2d(X)
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

        preds = []
        for model in self.models_:
            pred = model(X_tensor, training=False).numpy().reshape(-1)
            preds.append(pred)

        preds = np.asarray(preds)
        preds = preds * self.y_std_ + self.y_mean_
        mean = preds.mean(axis=0)
        std = preds.std(axis=0) + 1e-12

        if return_std:
            return mean, std
        return mean

    def _ensemble_objective(self, x_tensor: tf.Tensor) -> tf.Tensor:
        preds = [tf.reshape(model(x_tensor, training=False), (-1,)) for model in self.models_]
        return tf.reduce_mean(tf.stack(preds, axis=0), axis=0)

    def optimise_from_starts(self, starts: np.ndarray) -> Dict[str, np.ndarray]:
        """Run bounded gradient ascent from multiple starting points."""
        if not self.models_:
            raise RuntimeError("Call fit before optimise_from_starts.")

        starts = self._as_2d(starts)
        starts = np.clip(starts, 0.0, 1.0).astype(np.float32)
        x = tf.Variable(starts, dtype=tf.float32)

        trajectory_best_x = None
        trajectory_best_score = -np.inf

        for _ in range(self.gradient_steps):
            with tf.GradientTape() as tape:
                scores = self._ensemble_objective(x)
                objective = tf.reduce_sum(scores)

            grad = tape.gradient(objective, x)
            if grad is None:
                break

            grad_norm = tf.maximum(tf.norm(grad, axis=1, keepdims=True), 1e-8)
            x.assign(tf.clip_by_value(x + self.gradient_step_size * grad / grad_norm, 0.0, 1.0))

            current_scores = self._ensemble_objective(x).numpy()
            local_best_idx = int(np.argmax(current_scores))
            if float(current_scores[local_best_idx]) > trajectory_best_score:
                trajectory_best_score = float(current_scores[local_best_idx])
                trajectory_best_x = x.numpy()[local_best_idx].copy()

        final_points = x.numpy()
        mean, std = self.predict(final_points, return_std=True)
        best_idx = int(np.argmax(mean))

        if trajectory_best_x is not None:
            trajectory_mean, trajectory_std = self.predict(
                trajectory_best_x.reshape(1, -1),
                return_std=True,
            )
            if float(trajectory_mean[0]) > float(mean[best_idx]):
                return {
                    "candidate": trajectory_best_x,
                    "mean": float(trajectory_mean[0]),
                    "std": float(trajectory_std[0]),
                    "optimised_points": final_points,
                }

        return {
            "candidate": final_points[best_idx],
            "mean": float(mean[best_idx]),
            "std": float(std[best_idx]),
            "optimised_points": final_points,
        }

    def suggest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        candidates: Optional[np.ndarray] = None,
        best_gp_input: Optional[np.ndarray] = None,
        top_k_starts: int = 32,
        n_random_starts: int = 64,
    ) -> Dict[str, object]:
        """Fit the TensorFlow NN surrogate and return a gradient-optimised candidate report."""
        X = self._as_2d(X)
        y = self._as_1d(y)
        self.fit(X, y)

        rng = np.random.default_rng(self.random_state)
        starts = []

        starts.append(X[int(np.argmax(y))])

        if best_gp_input is not None:
            starts.append(np.asarray(best_gp_input, dtype=np.float32).ravel())

        if candidates is not None:
            candidates = self._as_2d(candidates)
            cand_mean, cand_std = self.predict(candidates, return_std=True)
            top_k = min(top_k_starts, len(candidates))
            top_idx = np.argsort(cand_mean)[-top_k:]
            starts.extend(candidates[top_idx])
        else:
            cand_mean = None
            cand_std = None

        random_starts = rng.random((n_random_starts, X.shape[1]), dtype=np.float32)
        starts.extend(random_starts)

        starts = np.unique(np.asarray(starts, dtype=np.float32), axis=0)
        opt_report = self.optimise_from_starts(starts)

        candidate = np.asarray(opt_report["candidate"], dtype=float)
        train_mean, train_std = self.predict(X, return_std=True)

        return {
            "candidate": candidate,
            "nn_mean": float(opt_report["mean"]),
            "nn_std": float(opt_report["std"]),
            "distance_to_best_observed": float(np.linalg.norm(candidate - X[int(np.argmax(y))])),
            "distance_to_gp_candidate": (
                None if best_gp_input is None else float(np.linalg.norm(candidate - np.asarray(best_gp_input)))
            ),
            "training_rmse": float(np.sqrt(np.mean((train_mean - y) ** 2))),
            "n_starts": int(len(starts)),
            "optimised_points": opt_report["optimised_points"],
            "candidate_pool_nn_mean": cand_mean,
            "candidate_pool_nn_std": cand_std,
        }


def print_tf_nn_candidate_report(report: Dict[str, object], round_digits: int = 6) -> None:
    """Pretty-print the TensorFlow NN candidate report inside a notebook."""
    candidate = np.asarray(report["candidate"])

    print("\n============= TENSORFLOW NEURAL SURROGATE CANDIDATE =============")
    print("TF NN candidate:")
    print(np.round(candidate, round_digits))
    print("TF NN predicted mean:", round(float(report["nn_mean"]), round_digits))
    print("TF NN ensemble std:", round(float(report["nn_std"]), round_digits))
    print("TF NN training RMSE:", round(float(report["training_rmse"]), round_digits))
    print("Distance to best observed:", round(float(report["distance_to_best_observed"]), round_digits))
    if report.get("distance_to_gp_candidate") is not None:
        print("Distance to GP/hybrid candidate:", round(float(report["distance_to_gp_candidate"]), round_digits))
    print("Gradient-ascent starts used:", int(report["n_starts"]))
    print("=================================================================\n")
