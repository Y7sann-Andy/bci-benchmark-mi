"""CSP + StandardScaler + PyTorch MLP wrapper.

Sklearn-style estimator for the benchmark: implements .fit(X, y) and
.predict(X) so it plugs into cross_validate_within_session via a factory.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler


class _MLP(nn.Module):
    """PyTorch MLP head sitting on top of CSP features.

    Input  : (batch, n_components)    -- CSP log-variance features
    Output : (batch, n_classes)       -- raw logits (no softmax — that lives in CrossEntropyLoss)
    """

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], n_classes: int):
        super().__init__()

        prev = in_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CSPMLPClassifier:
    """CSP + StandardScaler + MLP, as a single sklearn-style estimator."""

    def __init__(
        self,
        n_components: int = 4,
        hidden_dims: Sequence[int] = (32, 16),
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 16,
        device: str = "cpu",
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> CSPMLPClassifier:
        """Train CSP, scaler, and MLP on (X, y).

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
        y : ndarray, shape (n_trials,), integer labels starting at 0.

        Returns
        -------
        self
        """
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self.csp_ = CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False)
        X_csp = self.csp_.fit_transform(X, y)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_csp)

        X_t = torch.from_numpy(X_scaled).float().to(self.device)
        y_t = torch.from_numpy(y).long().to(self.device)

        n_classes = int(np.unique(y).size)
        self.model_ = _MLP(self.n_components, self.hidden_dims, n_classes).to(self.device)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.n_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model_(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for X.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        ndarray, shape (n_trials,)
        """
        X_csp = self.csp_.transform(X)
        X_scaled = self.scaler_.transform(X_csp)

        X_t = torch.from_numpy(X_scaled).float().to(self.device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t)
            y_pred = logits.argmax(dim=1).cpu().numpy()
        return y_pred
