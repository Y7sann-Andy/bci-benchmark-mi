"""EEGNet (Lawhern 2018) end-to-end wrapper.

Sklearn-style estimator for the benchmark: implements .fit(X, y) and
.predict(X) so it plugs into cross_validate_within_session via a factory.

No CSP / StandardScaler — EEGNet learns spatial+temporal features directly
from raw EEG. The internal BatchNorm layers handle normalization.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _EEGNet(nn.Module):
    """EEGNet (Lawhern 2018), end-to-end CNN for EEG classification.

    Input  : (batch, 1, n_channels, n_samples)   -- raw band-passed EEG epochs
    Output : (batch, n_classes)                  -- raw logits (no softmax)
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_t: int = 80,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernel_t), padding="same", bias = False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial_conv = nn.Conv2d(in_channels=F1, out_channels=F1*D, kernel_size=(n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_depthwise = nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 16), groups=F1*D, padding="same", bias=False)
        self.sep_pointwise = nn.Conv2d(in_channels=F1*D, out_channels=F2, kernel_size=(1,1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1,8))
        self.drop2 = nn.Dropout(dropout)

        flat_dim = F2 * 1 * (n_samples // 4 // 8)
        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


class EEGNetClassifier:
    """EEGNet end-to-end as an sklearn-style estimator."""

    def __init__(
        self,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_t: int = 80,
        dropout: float = 0.25,
        lr: float = 1e-3,
        n_epochs: int = 300,
        batch_size: int = 16,
        n_classes: int = None, # if None, infer from data in .fit
        scale: float = 1e6,
        device: str = "cpu",
        random_state: int = 42,
    ):
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_t = kernel_t
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.scale = scale
        self.device = device
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetClassifier":
        """Train EEGNet on (X, y).

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

        n_trials, n_channels, n_samples = X.shape
        n_classes = self.n_classes if self.n_classes is not None else int(np.unique(y).size)                                    
        self.n_classes_ = n_classes  


        X_t = torch.from_numpy(X * self.scale).float().unsqueeze(1).to(self.device)
        y_t = torch.from_numpy(y).long().to(self.device)

        self.model = _EEGNet(
            n_channels=n_channels, n_samples=n_samples, n_classes=self.n_classes_,
            F1=self.F1, D=self.D, F2=self.F2, kernel_t=self.kernel_t, dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.n_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
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
        X_t = torch.from_numpy(X * self.scale).float().unsqueeze(1).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            y_pred = logits.argmax(dim=1).cpu().numpy()
        return y_pred
    
