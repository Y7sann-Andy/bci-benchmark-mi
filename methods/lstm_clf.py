"""LSTM classifier for EEG MI (Hochreiter & Schmidhuber 1997).

Sklearn-style estimator for the benchmark: implements .fit(X, y) and
.predict(X) so it plugs into cross_validate_within_session via a factory.

Mirrors methods/eegnet.py structure:
    _LSTM           — internal nn.Module (architecture)
    LSTMClassifier  — public sklearn-style wrapper (fit/predict)

Notes
-----
- Forget-gate bias initialized to 1.0 (Jozefowicz 2015) to keep f_t near
  sigmoid(1) ≈ 0.73 at the start of training, mitigating vanishing gradient.
- Gradient clipping applied per-batch (max_norm default 1.0) to handle
  occasional gradient spikes from the gates' indirect paths (Pascanu 2013).
- Input scaled V → μV in LSTMClassifier.fit/predict (wrapper boundary, not
  inside _LSTM.forward) to keep the architecture data-scale agnostic.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTM(nn.Module):
    """LSTM architecture for EEG MI classification.

    Input  : (B, C, T) or (B, 1, C, T) EEG epochs already scaled to μV by the wrapper
    Output : (B, n_classes) logits

    Hidden state of the last timestep (top layer) is fed to a Linear classifier.
    """

    def __init__(
        self,
        n_channels: int,
        hidden_size: int = 32,
        n_classes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.5
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

        for name, param in self.lstm.named_parameters():                                                         
            if "bias" in name:                  
                n = param.size(0)
                param.data.fill_(0.0)                                                                            
                param.data[n // 4 : n // 2].fill_(1.0)  # forget gate bias to 1, input gate bias to 0(Jozefowicz 2015)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.permute(0,2,1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]
        h_last = self.dropout(h_last)
        logits = self.classifier(h_last)
        return logits


class LSTMClassifier:
    """LSTM end-to-end as an sklearn-style estimator.

    Mirrors EEGNetClassifier API for plug-in compatibility with
    cross_validate_within_session.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.5,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 16,
        clip_grad_norm: float = 1.0,
        n_classes: int = None,    # if None, infer from y in .fit
        scale: float = 1e6,
        device: str = "cpu",
        random_state: int = 42,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_grad_norm = clip_grad_norm
        self.n_classes = n_classes
        self.scale = scale
        self.device = device
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMClassifier":
        """Train LSTM on (X, y).

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
        X_t = torch.from_numpy(X * self.scale).float().to(self.device)
        y_t = torch.from_numpy(y).long().to(self.device)
        self.model = _LSTM(
            n_channels=n_channels, hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout, n_classes=self.n_classes_
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.n_epochs):
            for X_b, y_b in loader:
                optimizer.zero_grad()
                logits = self.model(X_b)
                loss = criterion(logits, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
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
        X_t = torch.from_numpy(X * self.scale).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds
