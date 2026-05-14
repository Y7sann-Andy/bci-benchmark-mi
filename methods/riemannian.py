"""Riemannian MDM pipeline wrapper.

Sklearn-style estimator: .fit(X, y) / .predict(X), plugs into
cross_validate_within_session via a factory.

Covariances (SPD featurization) -> MDM (Minimum Distance to Mean).
Each trial's covariance matrix is an SPD-manifold point; .fit computes
the per-class Riemannian (Frechet) mean; .predict assigns the class whose
mean is closest under the affine-invariant Riemannian metric.

No spatial filtering, no feature vector -- the learned model is K SPD matrices.
"""
from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM


class RiemannianMDMClassifier:
    """Covariance featurization + MDM as an sklearn-style estimator.

    Parameters
    ----------
    cov_estimator : str
        Covariance estimator passed to pyriemann Covariances. 'oas' (Oracle
        Approximating Shrinkage) regularizes toward a scaled identity, which
        helps at 22 channels x ~250 trials/subject; 'scm' is the plain sample
        covariance. Default 'oas'.
    metric : str
        Metric for the per-class mean and the distance in MDM. 'riemann' is
        the affine-invariant metric; 'logeuclid' / 'euclid' are available as
        ablations to quantify what affine invariance buys. Default 'riemann'.
    """

    def __init__(self, cov_estimator: str = "oas", metric: str = "riemann"):
        self.cov_estimator = cov_estimator
        self.metric = metric

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiemannianMDMClassifier":
        """Estimate trial covariances, then fit MDM (per-class Riemannian means).

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
        y : ndarray, shape (n_trials,), integer labels starting at 0.

        Returns
        -------
        self
        """
        self.classes_ = np.unique(y)
        self.cov_ = Covariances(estimator=self.cov_estimator)
        covs = self.cov_.fit_transform(X)
        self.mdm_ = MDM(metric=self.metric)
        self.mdm_.fit(covs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by nearest Riemannian mean.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        ndarray, shape (n_trials,)
        """
        covs = self.cov_.transform(X)
        return self.mdm_.predict(covs)
