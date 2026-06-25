
# Neural-network surrogate candidate generator for black-box optimisation.

# This module is intended to be used as an extra candidate source beside the
# Gaussian Process ensemble. It trains a small PyTorch MLP on the observed
# samples, then uses backpropagation with respect to the input x to perform
# bounded gradient ascent inside the unit hypercube.
#
# Typical notebook usage:
#
#    from src.neural_surrogate import NeuralSurrogateCandidateGenerator
#
#    nn_generator = NeuralSurrogateCandidateGenerator(
#        n_ensemble=5,
#        hidden_units=32,
#        learning_rate=1e-3,
#        max_epochs=1500,
#        random_state=0,
#        device="cpu",
#        verbose=True,
#    )
#
#    nn_report = nn_generator.suggest(
#        X=X,
#        y=y,
#        candidates=candidates,
#        best_gp_input=best_input,
#        top_k_starts=32,
#    )

#    print(nn_report["candidate"])


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - useful message for notebooks
    raise ImportError(
        "neural_surrogate.py requires PyTorch. Install torch or skip the NN section."
    ) from exc


class _SmallMLP(nn.Module):
    # Small regression MLP used inside the surrogate ensemble.

    def __init__(self, n_dimensions: int, hidden_units: int = 32, activation: str = "tanh"):
        super().__init__()

        if activation == "relu":
            act = nn.ReLU
        elif activation == "gelu":
            act = nn.GELU
        else:
            act = nn.Tanh

        self.net = nn.Sequential(
            nn.Linear(n_dimensions, hidden_units),
            act(),
            nn.Linear(hidden_units, hidden_units),
            act(),
            nn.Linear(hidden_units, 1),
        )
        #self.net = nn.Sequential(
        #    nn.Linear(n_dimensions, hidden_units),
        #    act(),

        #    nn.Linear(hidden_units, hidden_units),
        #    act(),

        #    nn.Linear(hidden_units, hidden_units),
        #    act(),

        #    nn.Linear(hidden_units, hidden_units),
        #    act(),

        #    nn.Linear(hidden_units, 1),
        #)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class NeuralSurrogateCandidateGenerator:
    # Train a small NN ensemble and optimise candidates by input-gradient ascent.
    # Parameters are deliberately conservative because the capstone datasets are
    # small. Treat the resulting point as an extra candidate to compare against
    # the GP/SVM pipeline, not as a replacement for it.
   

    n_ensemble: int = 5              # Number of models in the ensemble
    hidden_units: int = 32           # Number of hidden units in the MLP
    activation: str = "tanh"         # Activation function: "tanh", "relu", or "gelu"
    learning_rate: float = 1e-3      # Learning rate for the AdamW optimiser
    weight_decay: float = 1e-3       # Weight decay for the AdamW optimiser
    max_epochs: int = 1500           # Maximum number of training epochs
    patience: int = 200              # Early stopping patience in epochs
    gradient_steps: int = 80         # Number of gradient ascent steps for candidate optimisation
    gradient_step_size: float = 0.03 # Step size for gradient ascent
    random_state: int = 0            # Random seed for reproducibility
    device: str = "cpu"              # Device for PyTorch: "cpu" or "cuda"
    verbose: bool = False            # Verbosity flag for training output

    # Internal attributes initialized after fitting 
    def __post_init__(self) -> None:
        # List of trained MLP models in the ensemble.
        self.models_: List[_SmallMLP] = []
        # Mean and standard deviation of the training targets, used for scaling predictions back to original units.
        self.y_mean_: float = 0.0
        # Standard deviation of the training targets, used for scaling predictions back to original units.
        self.y_std_: float = 1.0
        # Number of input dimensions, inferred from the training data during fitting.
        self.n_dimensions_: Optional[int] = None

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        # Convert input to a 2D array and validate shape.
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_dimensions).")
        return X

    @staticmethod
    def _as_1d(y: np.ndarray) -> np.ndarray:
        # Convert input to a 1D array and validate shape.
        y = np.asarray(y, dtype=np.float32).ravel()
        if y.ndim != 1:
            raise ValueError("y must be a 1D array or flattenable to 1D.")
        return y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralSurrogateCandidateGenerator":
        # Fit the NN ensemble to the provided training data (X, y).
        X = self._as_2d(X)
        y = self._as_1d(y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if len(X) < 4:
            raise ValueError("At least 4 samples are recommended for the NN surrogate.")

        # Store the number of input dimensions and the mean and standard deviation of the training targets for later use in prediction scaling.
        self.n_dimensions_ = X.shape[1]
        # Scale y to have zero mean and unit variance for more stable training. The original mean and std are stored to scale predictions back to original units.
        self.y_mean_ = float(np.mean(y))
        # Add a small constant to the standard deviation to prevent division by zero in case of zero variance.
        self.y_std_ = float(np.std(y) + 1e-8)
        # Scale the training targets to have zero mean and unit variance, which can help with training stability and convergence of the neural network models.
        y_scaled = (y - self.y_mean_) / self.y_std_

        # Convert data to PyTorch tensors on the specified device.
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.device)

        # Clear any existing models in the ensemble before training new ones.
        self.models_ = []

        # Train each model in the ensemble with early stopping.
        for model_idx in range(self.n_ensemble):
            # Set random seeds for reproducibility. Each model in the ensemble gets a different seed based on the 
            # base random state and the model index to ensure diversity among the models.
            seed = int(self.random_state + model_idx)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Initialize the small MLP model and move it to the specified device.
            model = _SmallMLP(
                n_dimensions=X.shape[1],
                hidden_units=self.hidden_units,
                activation=self.activation,
            ).to(self.device)

            # Set up the AdamW optimiser and MSE loss function.
            optimiser = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            loss_fn = nn.MSELoss()

            best_loss = np.inf
            stale_epochs = 0
            best_state = None

            # Train the model for a maximum number of epochs, applying early stopping based on validation loss.
            for epoch in range(self.max_epochs):
                model.train()
                optimiser.zero_grad(set_to_none=True)
                pred = model(X_tensor)
                loss = loss_fn(pred, y_tensor)
                loss.backward()
                optimiser.step()

                loss_value = float(loss.detach().cpu().item())
                if loss_value < best_loss - 1e-7:
                    # Update the best loss and save the model state if the current 
                    # loss is better than the best recorded loss.
                    best_loss = loss_value
                    stale_epochs = 0
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                else:
                    # Increment the stale epoch counter if the loss has not improved.
                    stale_epochs += 1

                if stale_epochs >= self.patience:
                    # Stop training early if the model has not improved for a number 
                    # of epochs equal to the patience parameter.
                    break

            # After training, load the best model state if available and set the model to evaluation mode.
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.models_.append(model)

            if self.verbose:
                print(f"NN model {model_idx + 1}/{self.n_ensemble}: train MSE={best_loss:.6g}")
    
        return self

    # Predict mean and ensemble standard deviation in original y units.
    def predict(self, X: np.ndarray, return_std: bool = True):
        """Predict mean and ensemble standard deviation in original y units."""
        if not self.models_:
            raise RuntimeError("Call fit before predict.")

        X = self._as_2d(X)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        preds = []
        with torch.no_grad():
            for model in self.models_:
                pred = model(X_tensor).detach().cpu().numpy()
                preds.append(pred)

        preds = np.asarray(preds)
        preds = preds * self.y_std_ + self.y_mean_
        mean = preds.mean(axis=0)
        std = preds.std(axis=0) + 1e-12

        if return_std:
            return mean, std
        return mean

    # Compute the ensemble objective for gradient ascent optimisation.
    def _ensemble_objective(self, x_tensor: torch.Tensor) -> torch.Tensor:
        preds = [model(x_tensor) for model in self.models_]
        return torch.stack(preds, dim=0).mean(dim=0)

    # Perform bounded gradient ascent from multiple starting points to find the best candidate.
    def optimise_from_starts(self, starts: np.ndarray) -> Dict[str, np.ndarray]:
        """Run bounded gradient ascent from multiple starting points."""
        if not self.models_:
            raise RuntimeError("Call fit before optimise_from_starts.")

        starts = self._as_2d(starts)
        starts = np.clip(starts, 0.0, 1.0)

        # Convert the starting points to a PyTorch tensor with gradients enabled for optimization.
        x = torch.tensor(starts, dtype=torch.float32, device=self.device, requires_grad=True)

        # Initialize variables to track the best point found during the gradient ascent trajectory.
        trajectory_best_x = None
        trajectory_best_score = -np.inf

        # Perform gradient ascent for a fixed number of steps, updating the 
        # input points based on the gradient of the ensemble objective.
        for _ in range(self.gradient_steps):
            # Compute the ensemble objective for the current input points and perform backpropagation to compute gradients.
            objective = self._ensemble_objective(x).sum()
            # Backpropagate to compute the gradient of the objective with respect to the input x.
            objective.backward()

            # Update the input points by taking a step in the direction of the gradient, normalizing to maintain 
            # consistent step sizes, and clamping to ensure the points remain within the unit hypercube.
            with torch.no_grad():
                # Extract the gradient, compute its norm, and update x by taking a step in the direction of the gradient.
                grad = x.grad
                # Compute the norm of the gradient for each point and normalize it to prevent excessively large updates.
                grad_norm = torch.linalg.norm(grad, dim=1, keepdim=True).clamp_min(1e-8)
                # Update x by taking a step in the direction of the gradient, scaled by the specified step size and normalized by the gradient norm.
                x += self.gradient_step_size * grad / grad_norm
                # Clamp x to ensure it remains within the bounds of the unit hypercube [0, 1]^d.
                x.clamp_(0.0, 1.0)

                # Evaluate the ensemble objective at the new points and update the best point found during the trajectory if the current score is better.
                scores = self._ensemble_objective(x).detach().cpu().numpy()
                # Find the index of the point with the highest score in the current batch.
                local_best_idx = int(np.argmax(scores))
                # If the best score found in the current batch is better than the best score found during the trajectory, update the trajectory best score and corresponding input point.
                if float(scores[local_best_idx]) > trajectory_best_score:
                    # Update the best score found during the trajectory and the corresponding input point.
                    trajectory_best_score = float(scores[local_best_idx])
                    # Store the best input point found during the trajectory by detaching it from the computation graph, 
                    # moving it to the CPU, converting it to a NumPy array, and copying it to ensure it is not modified 
                    # in-place during further optimization steps.
                    trajectory_best_x = x.detach().cpu().numpy()[local_best_idx].copy()

                # Zero the gradients for the next iteration to prevent accumulation.
                x.grad.zero_()

        # After completing the gradient ascent steps, evaluate the ensemble mean and standard deviation at the final optimized points, and identify the best point among them.
        final_points = x.detach().cpu().numpy()
        # Evaluate the ensemble mean and standard deviation at the final optimized points.
        mean, std = self.predict(final_points, return_std=True)
        # Identify the index of the point with the highest predicted mean in the final set of optimized points.
        best_idx = int(np.argmax(mean))

        # If the best point found during gradient ascent has a higher predicted mean 
        # than the best point in the final set, return it as the candidate.
        if trajectory_best_x is not None:
            trajectory_mean, trajectory_std = self.predict(trajectory_best_x.reshape(1, -1), return_std=True,)
            # Compare the best point found during the trajectory with the best point in the final set of optimized points, 
            # and return the one with the higher predicted mean as the candidate.
            if float(trajectory_mean[0]) > float(mean[best_idx]):
                return {
                    "candidate": trajectory_best_x,
                    "mean": float(trajectory_mean[0]),
                    "std": float(trajectory_std[0]),
                    "optimised_points": final_points,
                }
        
        # If the best point found during the trajectory does not have a higher predicted mean than the best point in 
        # the final set, return the best point from the final set as the candidate.
        return {
            "candidate": final_points[best_idx],
            "mean": float(mean[best_idx]),
            "std": float(std[best_idx]),
            "optimised_points": final_points,
        }

    # Suggest a candidate by fitting the NN surrogate and performing gradient ascent.
    def suggest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        candidates: Optional[np.ndarray] = None,
        best_gp_input: Optional[np.ndarray] = None,
        top_k_starts: int = 32,
        n_random_starts: int = 64,
    ) -> Dict[str, object]:
        """Fit the NN surrogate and return a gradient-optimised candidate report."""
        X = self._as_2d(X)
        y = self._as_1d(y)
        self.fit(X, y)

        rng = np.random.default_rng(self.random_state)
        starts = []

        # Always include the best observed point as a stable exploitation start.
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

        report = {
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
        return report

# Pretty-print the NN candidate report inside a notebook.
def print_nn_candidate_report(report: Dict[str, object], round_digits: int = 6) -> None:
    """Pretty-print the NN candidate report inside a notebook."""
    candidate = np.asarray(report["candidate"])

    print("\n================ NEURAL SURROGATE CANDIDATE ================")
    print("NN candidate:")
    print(np.round(candidate, round_digits))
    print("NN predicted mean:", round(float(report["nn_mean"]), round_digits))
    print("NN ensemble std:", round(float(report["nn_std"]), round_digits))
    print("NN training RMSE:", round(float(report["training_rmse"]), round_digits))
    print("Distance to best observed:", round(float(report["distance_to_best_observed"]), round_digits))
    if report.get("distance_to_gp_candidate") is not None:
        print("Distance to GP/hybrid candidate:", round(float(report["distance_to_gp_candidate"]), round_digits))
    print("Gradient-ascent starts used:", int(report["n_starts"]))
    print("============================================================\n")
