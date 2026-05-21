"""Lee2019 (n=54, 2-class) replication of the two capstone experiments.

Purpose: re-run cross-session and channel-reduction on a MUCH larger
dataset (54 subjects vs 2a's 9) so the qualitative findings can be tested
at real statistical power. Lee2019 is 2-class (chance 50%); this replicates
the *pattern* (which method degrades least cross-session; accuracy-vs-channel
trend), not 2a's absolute 4-class numbers.

Writes incrementally (one CSV append per subject) so an overnight kill mid-run
still leaves every completed subject's rows on disk:
    results/exp_lee2019_cross_session.csv     (method, subject, protocol, score)
    results/exp_lee2019_channel_reduction.csv (method, subject, n_channels, score)

Cross-session runs FIRST (lighter, it's the differentiator result); channel
reduction (heavier, EEGNet-bound) runs second.

Usage
-----
    # Smoke test: 2 subjects, shallow method, cross-session only
    python -m experiments.run_lee2019_overnight \
        --subjects 1 2 --methods CSP+LDA --experiments cross_session

    # Overnight: everything (~one full night, EEGNet channel-reduction is the long pole)
    python -m experiments.run_lee2019_overnight --methods all
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import mne

mne.set_log_level("WARNING")

from data import lee2019
from data.lee2019 import (
    SUBJECTS, load_subject_session, channel_names,
    select_channels, CHANNEL_SUBSETS,
)
from evaluation.crossval import cross_validate_within_session
from evaluation.cross_session import evaluate_cross_session
from methods.csp_lda import CSPLDAClassifier
from methods.eegnet import EEGNetClassifier
from methods.riemannian import RiemannianMDMClassifier
from methods.tangent_lr import TangentSpaceLRClassifier
from preprocessing import normalize_per_session

METHODS: dict[str, Callable[[], object]] = {
    "CSP+LDA": lambda: CSPLDAClassifier(),
    "MDM":     lambda: RiemannianMDMClassifier(metric="riemann"),
    "TS-LR":   lambda: TangentSpaceLRClassifier(),
    "TS-LR-RPA": lambda: TangentSpaceLRClassifier(recenter_target=True),  # cross-session only (within-CV = no-op)
    "EEGNet":  lambda: EEGNetClassifier(scale=1, device="mps"),
}
DEFAULT_METHODS = ["CSP+LDA", "MDM", "TS-LR"]   # smoke-test default (fast)

CS_CSV = Path("results/exp_lee2019_cross_session.csv")
CR_CSV = Path("results/exp_lee2019_channel_reduction.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                   help=f"'all' for {list(METHODS)}, or a subset. Default {DEFAULT_METHODS}")
    p.add_argument("--subjects", nargs="+", type=int, default=SUBJECTS,
                   help="Subject IDs (default 1..54)")
    p.add_argument("--experiments", nargs="+",
                   default=["cross_session", "channel_reduction"],
                   choices=["cross_session", "channel_reduction"])
    return p.parse_args()


def resolve_methods(arg: list[str]) -> list[str]:
    if arg == ["all"]:
        return list(METHODS)
    unknown = [m for m in arg if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {list(METHODS)}")
    return arg


def _append_rows(csv_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Append rows, writing header only if the file does not exist yet."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    pd.DataFrame(rows, columns=fieldnames).to_csv(
        csv_path, mode="a", header=write_header, index=False
    )


def run_cross_session(methods: list[str], subjects: list[int]) -> None:
    print(f"\n=== Cross-session: {len(subjects)} subjects x {len(methods)} methods ===", flush=True)
    t0 = time.time()
    for i, subject in enumerate(subjects, 1):
        X_tr, y_tr = load_subject_session(subject, "T")
        X_te, y_te = load_subject_session(subject, "E")
        X_tr, X_te = normalize_per_session(X_tr), normalize_per_session(X_te)
        rows = []
        for method in methods:
            r = evaluate_cross_session(METHODS[method], X_tr, y_tr, X_te, y_te)
            rows.append({"method": method, "subject": subject,
                         "protocol": "cross-session", "score": r["score"]})
            print(f"  [CS {i}/{len(subjects)}] s{subject:02d} {method:8s} "
                  f"score={r['score']:.4f} ({(time.time()-t0)/60:.1f}min)", flush=True)
        _append_rows(CS_CSV, rows, ["method", "subject", "protocol", "score"])
    print(f"=== cross-session done in {(time.time()-t0)/60:.1f}min -> {CS_CSV} ===", flush=True)


def run_channel_reduction(methods: list[str], subjects: list[int]) -> None:
    all_names = channel_names(subjects[0])
    ks = sorted(CHANNEL_SUBSETS, reverse=True)
    print(f"\n=== Channel reduction: {len(subjects)} subj x {len(ks)} K x {len(methods)} methods ===", flush=True)
    print(f"    montage has {len(all_names)} ch; K values {ks}", flush=True)
    t0 = time.time()
    for i, subject in enumerate(subjects, 1):
        X, y = load_subject_session(subject, "T")
        X = normalize_per_session(X)
        rows = []
        for k in ks:
            X_sub = select_channels(X, CHANNEL_SUBSETS[k], all_names)
            for method in methods:
                r = cross_validate_within_session(METHODS[method], X_sub, y, verbose=False)
                rows.append({"method": method, "subject": subject,
                             "n_channels": k, "score": r["mean"]})
                print(f"  [CR {i}/{len(subjects)}] s{subject:02d} K={k:2d} {method:8s} "
                      f"score={r['mean']:.4f} ({(time.time()-t0)/60:.1f}min)", flush=True)
        _append_rows(CR_CSV, rows, ["method", "subject", "n_channels", "score"])
    print(f"=== channel-reduction done in {(time.time()-t0)/60:.1f}min -> {CR_CSV} ===", flush=True)


def main() -> None:
    args = parse_args()
    methods = resolve_methods(args.methods)
    print(f"Lee2019 replication | methods={methods} | n_subjects={len(args.subjects)} "
          f"| experiments={args.experiments}", flush=True)
    if "cross_session" in args.experiments:
        run_cross_session(methods, args.subjects)
    if "channel_reduction" in args.experiments:
        run_channel_reduction(methods, args.subjects)


if __name__ == "__main__":
    main()
