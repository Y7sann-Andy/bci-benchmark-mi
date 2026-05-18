from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score


def evaluate_cross_session(
    pipeline_factory: Callable[[], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train pipeline on one session, evaluate on another. No CV folds.

    Cross-session protocol: one-shot fit on the training session, one-shot
    predict on the test session. Factory pattern mirrors
    `cross_validate_within_session` for runner-call-site uniformity (both
    evaluators take the same first argument shape); a fresh pipeline
    matters less here than in K-fold since only one .fit per call.

    Parameters
    ----------
    pipeline_factory : callable () -> estimator
        Zero-arg callable returning a fresh pipeline exposing
        `.fit(X, y)` and `.predict(X) -> y_pred`.
    X_train : ndarray, shape (n_train, n_channels, n_times)
        Training session trials (e.g., BCICIV 2a session T).
    y_train : ndarray, shape (n_train,)
        Training session labels, integer-encoded starting at 0.
    X_test : ndarray, shape (n_test, n_channels, n_times)
        Test session trials (e.g., BCICIV 2a session E). Must NOT be
        pre-pooled with X_train by the caller — see leakage contract.
    y_test : ndarray, shape (n_test,)
        Test session labels. Used for scoring ONLY — never passed to
        the pipeline.
    metric : callable (y_true, y_pred) -> float
        Default accuracy.
    verbose : bool
        If True, prints train/test sizes and final score.

    Returns
    -------
    dict with keys:
        'score'  : float
        'y_true' : ndarray of shape (n_test,)
        'y_pred' : ndarray of shape (n_test,)

    Leakage contract
    ----------------
    1. Build pipeline via factory.
    2. Call .fit(X_train, y_train). The ONLY fit call.
    3. Call .predict(X_test). No refit, no partial_fit, no transform
       on test data before predict.
    4. Score once.

    Provided X_train and X_test were not pre-pooled by the caller,
    sklearn's fit/transform contract handles per-step leakage prevention
    (StandardScaler.fit / TangentSpace.fit / CSP.fit all restricted to
    X_train). Pre-pooling — concatenating X_train and X_test before
    passing them in to "normalize together" — is leakage and is the
    caller's responsibility to avoid.
    """
    pipeline = pipeline_factory()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = metric(y_test, y_pred)
    if verbose:
        print(
            f"Train size: {len(X_train)}, Test size: {len(X_test)}, Score: {score:.4f}"
        )
    return {
        'score': score,
        'y_true': y_test,
        'y_pred': y_pred,
    }
