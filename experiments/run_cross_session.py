"""Cross-session evaluation for baseline methods on BCICIV-2a.

Train on session T, predict on session E. One score per (subject, method).
Writes to results/exp_baseline_v2.csv with protocol='cross-session'.

Idempotent: re-reads CSV, drops existing cross-session rows for the
methods this invocation actually runs, writes back non-affected rows
plus a fresh block. Smoke-test re-runs do not stomp rows from a prior
overnight run that included additional methods, and vice versa.

Usage
-----
    # Smoke test (default: shallow methods only, finishes in minutes)
    python -m experiments.run_cross_session

    # Specific subset
    python -m experiments.run_cross_session --methods CSP+LDA TS+LR

    # All methods including EEGNet (heavier — minutes to hours)
    python -m experiments.run_cross_session --methods all
"""
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
from typing import Callable

from data.bciciv2a import SUBJECTS, load_subject_session
from evaluation.cross_session import evaluate_cross_session
from methods.csp_lda import CSPLDAClassifier
from methods.eegnet import EEGNetClassifier
from methods.riemannian import RiemannianMDMClassifier
from methods.tangent_lr import TangentSpaceLRClassifier
from preprocessing import normalize_per_session

# All available methods, label -> factory. Labels must match existing
# within-CV rows in exp_baseline_v2.csv so cross-protocol comparisons
# join cleanly on the `method` column.
METHODS: dict[str, Callable[[], object]] = {
    "CSP+LDA": lambda: CSPLDAClassifier(),
    "MDM":     lambda: RiemannianMDMClassifier(metric="riemann"),
    "TS-LR":   lambda: TangentSpaceLRClassifier(),
    "TS-LR-RPA": lambda: TangentSpaceLRClassifier(recenter_target=True),  # unsupervised DA; cross-domain only (no-op within-CV/channel-reduction)
    "EEGNet":  lambda: EEGNetClassifier(scale=1, device="mps"),  # scale=1 (z-score); MPS ~10x speedup vs CPU default on Mac Air
}

# Smoke-test default — fast shallow methods only.
DEFAULT_METHODS = ["CSP+LDA", "MDM", "TS-LR"]

CSV_PATH = Path("results/exp_baseline_v2.csv")
FIELDNAMES = ["method", "subject", "protocol", "score"]
PROTOCOL = "cross-session"


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


def main() -> None:
    methods = resolve_methods(parse_args().methods)
    new_rows = []
    for subject in SUBJECTS:
        X_train, y_train = load_subject_session(subject, "T")
        X_test, y_test = load_subject_session(subject, "E")
        # A1 convention: each session z-scored using its own stats independently.
        # Matches deployment (new session normalized via its own calibration data)
        # and avoids T↔E leakage.
        X_train = normalize_per_session(X_train)
        X_test = normalize_per_session(X_test)
        for method in methods:
            pipe_factory = METHODS[method]
            result = evaluate_cross_session(
                pipe_factory, X_train, y_train, X_test, y_test
            )
            score = result['score']
            new_rows.append({
                'method': method,
                'subject': subject,
                'protocol': PROTOCOL,
                'score': score,
            })

    loaded_csv = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame(columns=FIELDNAMES)
    mask_keep = ~((loaded_csv['protocol'] == PROTOCOL) & (loaded_csv['method'].isin(methods)))
    kept_rows = loaded_csv[mask_keep]
    all_rows = pd.concat([kept_rows, pd.DataFrame(new_rows)], ignore_index=True)
    all_rows.to_csv(CSV_PATH, index=False, columns=FIELDNAMES)
    print(f"\nwrote {len(kept_rows)} kept + {len(new_rows)} new -> {CSV_PATH}")

if __name__ == "__main__":
    main()
