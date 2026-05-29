"""Leave-one-subject-out (LOSO) cross-subject evaluation.

Capsulated from the Week-2 notebook LOSO experiment (BCI_learning_week2.ipynb,
cell 50), which produced the LOSO rows in results/exp_2a_baseline.csv. Same
methodology: pool all OTHER subjects' trials as the training set, fit once,
predict the held-out subject. This is the hardest, most deployment-relevant
protocol: a brand-new user the model has never seen.

Factory pattern mirrors evaluate_cross_session: the caller passes a zero-arg
callable returning a fresh estimator with .fit(X, y) / .predict(X).

For an RPA estimator (TangentSpaceLRClassifier(recenter_target=True)) the
recentering is applied to the held-out subject at predict time, i.e. unsupervised
domain adaptation to the new user (no target labels used; the new user's
unlabeled covariances are needed to estimate the recentering -- transductive).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score


def evaluate_loso(
    pipe_factory: Callable[[], object],
    data: dict[int, tuple[np.ndarray, np.ndarray]],
    held_out: int,
    all_subjects: list[int],
    metric: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
) -> dict:
    """One LOSO fold: train on all subjects except `held_out`, test on `held_out`.

    Parameters
    ----------
    pipe_factory : zero-arg callable returning a fresh estimator.
    data : {subject_id: (X, y)} with X (n_trials, n_channels, n_times).
    held_out : subject id used as the test set.
    all_subjects : every subject id available.
    metric : (y_true, y_pred) -> float, default accuracy.

    Returns
    -------
    dict with score, n_train, n_test.
    """
    train_subs = [s for s in all_subjects if s != held_out]
    X_train = np.concatenate([data[s][0] for s in train_subs], axis=0)
    y_train = np.concatenate([data[s][1] for s in train_subs], axis=0)
    X_test, y_test = data[held_out]

    clf = pipe_factory()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "score": float(metric(y_test, y_pred)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
