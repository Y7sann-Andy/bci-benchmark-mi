"""CSP + LDA pipeline wrapper.

Sklearn-style estimator for the benchmark: implements .fit(X, y) and
.predict(X) so it plugs into cross_validate_within_session via a factory.

Composes mne.decoding.CSP (spatial filtering + log-variance features) with
sklearn LinearDiscriminantAnalysis (Fisher linear discriminant). The classical
CSP+LDA baseline for MI BCI (Lotte 2018 Section 3).

Multi-class (>2): mne.decoding.CSP uses one-vs-rest internally.
"""
from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class CSPLDAClassifier:
    """CSP feature extraction + LDA classifier as an sklearn-style estimator.

    CSP projects (n_channels, n_times) trials into n_components spatial
    filters that maximize the variance ratio between classes. log-variance
    of the projected signal becomes the feature vector for LDA.
    """

    def __init__(
        self,
        n_components: int = 4,
        reg: str | float | None = None,
    ):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPLDAClassifier":
        """Fit CSP on (X, y), then train LDA on the CSP feature vectors.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
        y : ndarray, shape (n_trials,), integer labels starting at 0.

        Returns
        -------
        self
        """
        self.classes_ = np.unique(y)
        self.csp_ = CSP(n_components=self.n_components, reg=self.reg, log=True)
        features = self.csp_.fit_transform(X, y)
        self.lda_ = LinearDiscriminantAnalysis()
        self.lda_.fit(features, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        ndarray, shape (n_trials,)
        """
        features = self.csp_.transform(X)
        preds = self.lda_.predict(features)
        return preds
