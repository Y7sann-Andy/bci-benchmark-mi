"""Per-session per-channel z-score normalization for EEG epochs.

Used by cross-session and channel-reduction runners to align EEGNet
preprocessing with the v2 within-CV baseline convention. Stabilizes
BatchNorm running statistics
across distribution-shifted inputs (cross-session / cross-subject),
trading ~2pt within-CV accuracy for substantial LOSO/cross-session
robustness.

Apply BEFORE passing X to the classifier pipeline:
    X_train = normalize_per_session(X_train)
    X_test  = normalize_per_session(X_test)   # independently — A1 convention

For cross-session this means T uses T's stats and E uses E's stats
independently — matches a deployment scenario where each new session
is normalized using its own calibration data.
"""
from __future__ import annotations

import numpy as np


def normalize_per_session(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score per session: mean/std over (trials, time) per channel.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_samples)
        EEG epochs from a single recording session.
    eps : float
        Small constant added to std to avoid division by zero on
        flat channels. Default 1e-8.

    Returns
    -------
    X_norm : ndarray, same shape as X
        Per-channel zero-mean unit-variance signals.
    """
    mean = X.mean(axis=(0, 2), keepdims=True)   # (1, n_channels, 1)
    std = X.std(axis=(0, 2), keepdims=True) + eps
    return (X - mean) / std
