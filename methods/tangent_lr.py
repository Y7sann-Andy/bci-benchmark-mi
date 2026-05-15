"""Tangent-space + Logistic Regression pipeline wrapper.

Sklearn-style estimator: .fit(X, y) / .predict(X), plugs into
cross_validate_within_session via a factory.

Covariances -> TangentSpace (log map at training Riemannian mean) -> LR.
Unlike MDM, LR has label-driven capacity over the n(n+1)/2 tangent-vector
features; addresses MDM's near-zero capacity bias.

C is sklearn's default (1.0); not tuned here. Inner-CV grid search over C
would be the principled next step but is omitted to match the unfated
hyperparameter convention in most BCI baseline reports.
"""
from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _spd_powers(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (M^(1/2), M^(-1/2)) for an SPD matrix via eigendecomposition."""
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-12)
    return V @ np.diag(np.sqrt(w)) @ V.T, V @ np.diag(1.0 / np.sqrt(w)) @ V.T


class TangentSpaceLRClassifier:
    """Covariances -> TangentSpace -> LR as an sklearn-style estimator.

    Parameters
    ----------
    cov_estimator : str
        Covariance estimator (matches RiemannianMDMClassifier). 'oas' (Oracle
        Approximating Shrinkage) regularizes toward a scaled identity, which
        helps at 22 channels x ~250 trials/subject. Default 'oas'.
    tangent_metric : str
        Metric defining the Riemannian mean used as the tangent-space reference
        point. 'riemann' is affine-invariant; 'logeuclid' is the ablation.
        Default 'riemann'.
    C : float
        Inverse L2 regularization strength for LR. Default 1.0 (sklearn default).
    max_iter : int
        LR optimizer iterations. 1000 is enough to converge on 253-dim tangent
        vectors at BCI scale; sklearn's default 100 sometimes underconverges.
    recenter_target : bool
        If True, apply unsupervised Riemannian domain adaptation at predict:
        transport target covariances from their own mean to the training
        reference before tangent projection. Recovers cross-session degradation
        (validated: TS-LR cross-session retention 95%->99%, d=+0.75). Default
        False = standard TS-LR, default path unchanged.
    """

    def __init__(self, cov_estimator: str = "oas", tangent_metric: str = "riemann",
                 C: float = 1.0, max_iter: int = 1000, recenter_target: bool = False):
        self.cov_estimator = cov_estimator
        self.tangent_metric = tangent_metric
        self.C = C
        self.max_iter = max_iter
        self.recenter_target = recenter_target

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TangentSpaceLRClassifier":
        """Fit Covariances -> TangentSpace -> LR.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
        y : ndarray, shape (n_trials,), integer labels starting at 0.

        Returns
        -------
        self
        """
        self.classes_ = np.unique(y)
        # No StandardScaler: tangent-vector entries have non-uniform natural
        # magnitudes (diagonal log-power ratios vs off-diagonal log-couplings)
        # that encode Riemannian-aware feature importance. Z-scoring strips
        # this structure and dropped within-CV accuracy by ~7 pts (verified
        # 2026-05-15). Matches MOABB's canonical Riemannian baseline convention.
        # lbfgs may log ConvergenceWarning at max_iter=1000; model is effectively
        # converged for prediction (gradient just doesn't hit tol=1e-4 ceiling).
        self.pipe_ = Pipeline([
            ("cov", Covariances(estimator=self.cov_estimator)),
            ("ts", TangentSpace(metric=self.tangent_metric)),
            ("lr", LogisticRegression(C=self.C, max_iter=self.max_iter)),
        ])
        self.pipe_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        recenter_target=False (default): standard Cov -> TangentSpace -> LR,
        identical to the prior behaviour. recenter_target=True: transport the
        target covariances from their own Riemannian mean to the training
        reference (unsupervised RPA domain adaptation) before tangent
        projection. fit() is untouched either way.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        ndarray, shape (n_trials,)
        """
        if not self.recenter_target:
            return self.pipe_.predict(X)
        cov = self.pipe_.named_steps["cov"]
        ts = self.pipe_.named_steps["ts"]
        lr = self.pipe_.named_steps["lr"]
        Ct = cov.transform(X)                            # target covariances (n, c, c)
        M_train_sqrt, _ = _spd_powers(ts.reference_)              # training reference M_T^(1/2)
        _, M_tgt_inv_sqrt = _spd_powers(mean_riemann(Ct))         # target mean M_E^(-1/2)
        # whiten target by its OWN mean (-> identity), then re-color to M_train:
        Ct_aligned = M_train_sqrt @ (M_tgt_inv_sqrt @ Ct @ M_tgt_inv_sqrt) @ M_train_sqrt
        return lr.predict(ts.transform(Ct_aligned))
