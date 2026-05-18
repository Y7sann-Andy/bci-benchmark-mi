"""Channel reduction sweep on BCICIV-2a, within-session CV.

For each method x subject x K (channel count), restrict X to a fixed
anatomically-motivated channel subset, then run within-session 5-fold CV.

Channel subsets are strictly nested (K=4 ⊂ K=8 ⊂ K=12 ⊂ K=16 ⊂ K=22)
so the accuracy-vs-K curve cleanly reflects channel count alone — no
confounding from per-subject channel choice strategy. Selection rule
follows standard sensorimotor topography (motor strip + immediately
surrounding electrodes), giving a deployable hardware recommendation
that maps directly to consumer MI headsets.

Writes to results/exp_channel_reduction.csv. Idempotent per the same
pattern as run_riemannian.py and run_cross_session.py.

WARNING: EEGNet in this sweep takes ~3-6h total across all subjects
and channel counts. Launch overnight via `--methods all`. Default
smoke-test runs only the shallow methods (finishes in minutes).

Usage
-----
    # Smoke test (default: shallow methods, finishes in minutes)
    python -m experiments.run_channel_reduction

    # Overnight: all methods including EEGNet (~3-6 hours)
    python -m experiments.run_channel_reduction --methods all

    # Specific subset (e.g., overnight EEGNet only)
    python -m experiments.run_channel_reduction --methods EEGNet
"""
from __future__ import annotations

import argparse
import time
import pandas as pd
from pathlib import Path
from typing import Callable

import numpy as np
import mne

mne.set_log_level("WARNING")  # silence per-covariance progress chatter

from data.bciciv2a import SUBJECTS, load_subject_session
from evaluation.crossval import cross_validate_within_session
from methods.csp_lda import CSPLDAClassifier
from methods.eegnet import EEGNetClassifier
from methods.riemannian import RiemannianMDMClassifier
from methods.tangent_lr import TangentSpaceLRClassifier
from preprocessing import normalize_per_session

# BCICIV 2a montage in loader output order (first 22 of 25, EOG dropped
# by data.bciciv2a.load_subject_session via sub['X'][:, :22]). Order is
# per the dataset description (Brunner et al. 2008, "BCI Competition
# 2008 — Graz data set A"). The .mat files do not carry channel labels;
# this ordering is the published specification, not in-file metadata.
BCICIV2A_CHANNELS_22: list[str] = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

# Anatomy-prioritized fixed subsets, strictly nested.
CHANNEL_SUBSETS: dict[int, list[str] | None] = {
    4:  ["C3", "Cz", "C4", "FCz"],
    8:  ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"],
    12: ["FC3", "FCz", "FC4", "C5", "C3", "Cz", "C4", "C6",
         "CP3", "CPz", "CP4", "Pz"],
    16: ["FC3", "FC1", "FCz", "FC2", "FC4",
         "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
         "CP3", "CPz", "CP4", "Pz"],
    22: None,  # sentinel: no slicing, use full montage
}

# All available methods, label -> factory.
METHODS: dict[str, Callable[[], object]] = {
    "CSP+LDA": lambda: CSPLDAClassifier(),
    "MDM":     lambda: RiemannianMDMClassifier(metric="riemann"),
    "TS-LR":   lambda: TangentSpaceLRClassifier(),
    "EEGNet":  lambda: EEGNetClassifier(scale=1, device="mps"),  # scale=1 (z-score); MPS ~10x speedup vs CPU default on Mac Air
}

# Smoke-test default — fast shallow methods only.
DEFAULT_METHODS = ["CSP+LDA", "MDM", "TS-LR"]

CSV_PATH = Path("results/exp_channel_reduction.csv")
FIELDNAMES = ["method", "subject", "n_channels", "score"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help=(
            f"Methods to run. Pass 'all' for {list(METHODS.keys())}, "
            f"or any subset. Default (smoke test): {DEFAULT_METHODS}."
        ),
    )
    return p.parse_args()


def resolve_methods(arg: list[str]) -> list[str]:
    if arg == ["all"]:
        return list(METHODS.keys())
    unknown = [m for m in arg if m not in METHODS]
    if unknown:
        raise ValueError(
            f"Unknown methods: {unknown}. Available: {list(METHODS.keys())}"
        )
    return arg


def select_channels(X: np.ndarray, names: list[str] | None) -> np.ndarray:
    """Slice X along the channel axis to channels named in `names`.

    Parameters
    ----------
    X : ndarray, shape (n_trials, 22, n_times)
        Assumed to be in BCICIV2A_CHANNELS_22 order (i.e., output of
        data.bciciv2a.load_subject_session).
    names : list[str] | None
        Channel names to retain. If None, X is returned unchanged.

    Returns
    -------
    X_sub : ndarray, shape (n_trials, len(names), n_times)
        Channels in the order specified by `names`.
    """
    if names is None:
        return X
    index = [BCICIV2A_CHANNELS_22.index(name) for name in names]
    return X[:, index, :]


def main() -> None:
    methods = resolve_methods(parse_args().methods)
    total = len(SUBJECTS) * len(CHANNEL_SUBSETS) * len(methods)
    print(f"=== Channel reduction sweep: {total} evaluations ===", flush=True)
    print(f"    {len(SUBJECTS)} subjects × {len(CHANNEL_SUBSETS)} K values × {len(methods)} methods", flush=True)
    print(f"    methods: {methods}", flush=True)
    print(f"    K values: {sorted(CHANNEL_SUBSETS.keys(), reverse=True)}", flush=True)
    print(flush=True)

    new_rows = []
    done = 0
    overall_start = time.time()
    for s_idx, subject in enumerate(SUBJECTS, 1):
        subj_start = time.time()
        print(f"===== Subject {subject} ({s_idx}/{len(SUBJECTS)}) =====", flush=True)
        X, y = load_subject_session(subject, "T")
        # Normalize on full 22-ch X before slicing — matches the within-CV
        # convention; keeps accuracy-vs-K cleanly attributable
        # to channel count alone (preprocessing stats don't change per K).
        X = normalize_per_session(X)
        for n_channels, names in CHANNEL_SUBSETS.items():
            X_sub = select_channels(X, names)
            for method in methods:
                t0 = time.time()
                pipe_factory = METHODS[method]
                result = cross_validate_within_session(
                    pipe_factory, X_sub, y, verbose=False
                )
                score = result['mean']
                elapsed = time.time() - t0
                done += 1
                rate = (time.time() - overall_start) / done
                eta_min = (total - done) * rate / 60
                print(
                    f"  [{done:3d}/{total}] {method:8s} K={n_channels:2d} "
                    f"score={score:.4f} (took {elapsed:5.1f}s, ETA ~{eta_min:5.1f}min)",
                    flush=True,
                )
                new_rows.append({
                    'method': method,
                    'subject': subject,
                    'n_channels': n_channels,
                    'score': score,
                })
        print(f"  -- subject {subject} done in {time.time() - subj_start:.1f}s --", flush=True)
    print(f"\n=== All {done} evaluations done in {(time.time() - overall_start)/60:.1f}min ===", flush=True)
    loaded_csv = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame(columns=FIELDNAMES)
    kept_rows = loaded_csv[~loaded_csv['method'].isin(methods)]
    all_rows = pd.concat([kept_rows, pd.DataFrame(new_rows)], ignore_index=True)
    all_rows = all_rows.sort_values(by=['method', 'subject', 'n_channels']).reset_index(drop=True)
    all_rows.to_csv(CSV_PATH, index=False, columns=FIELDNAMES)
    print(f"\nwrote {len(kept_rows)} kept + {len(new_rows)} new -> {CSV_PATH}")
    


if __name__ == "__main__":
    main()
