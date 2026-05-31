"""Cross-subject LOSO: does RPA recover the new-user drop?

Core pair = TS-LR vs TS-LR-RPA (unsupervised re-centering of the held-out
subject). On 2a, CSP+LDA is a reproduction gate (LOSO already in
results/exp_2a_baseline.csv). EEGNet can be added but is heavy on Lee2019
(trains a CNN on ~50 pooled subjects x 54 folds) -- use --epochs to cap it.

Methodology matches BCI_learning_week2.ipynb cell 50: session T per subject,
per-subject z-score, pool all other subjects -> fit -> predict held-out subject.

Idempotent + crash-safe: re-running a method overwrites only its rows; the CSV
is rewritten after every fold so a kill mid-run keeps completed folds.

Usage
-----
    python -m experiments.run_cross_subject --dataset 2a
    python -m experiments.run_cross_subject --dataset 2a --methods EEGNet
    python -m experiments.run_cross_subject --dataset lee2019 --methods EEGNet --epochs 150
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import mne
import torch

mne.set_log_level("WARNING")

from evaluation.cross_subject import evaluate_loso
from methods.csp_lda import CSPLDAClassifier
from methods.riemannian import RiemannianMDMClassifier
from methods.tangent_lr import TangentSpaceLRClassifier
from methods.eegnet import EEGNetClassifier

REPO = Path(__file__).resolve().parents[1]   # absolute -> output always lands in repo/results

DEFAULT_METHODS = {
    "2a":      ["CSP+LDA", "MDM", "TS-LR", "TS-LR-RPA"],
    "lee2019": ["CSP+LDA", "MDM", "TS-LR", "TS-LR-RPA"],
}

# Saved CSP+LDA LOSO scores from results/exp_2a_baseline.csv (reproduction gate, 2a only).
CSP_LDA_LOSO_REF = {
    1: 0.5729166666666666, 2: 0.2916666666666667, 3: 0.5868055555555556,
    4: 0.3958333333333333, 5: 0.3125,             6: 0.2916666666666667,
    7: 0.4722222222222222, 8: 0.6215277777777778, 9: 0.5590277777777778,
}


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_methods(names: list[str], epochs: int | None, device: str) -> dict:
    eegnet_kw = {"scale": 1, "device": device}
    if epochs is not None:
        eegnet_kw["n_epochs"] = epochs
    registry = {
        "CSP+LDA":   lambda: CSPLDAClassifier(n_components=4),
        "MDM":       lambda: RiemannianMDMClassifier(metric="riemann"),
        "TS-LR":     lambda: TangentSpaceLRClassifier(),
        "TS-LR-RPA": lambda: TangentSpaceLRClassifier(recenter_target=True),
        "EEGNet":    lambda: EEGNetClassifier(**eegnet_kw),
    }
    return {n: registry[n] for n in names}


def normalize_per_subject(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score per subject, per channel (over trials and time). Matches notebook."""
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + eps
    return (X - mean) / std


def _load_dataset(name: str):
    if name == "2a":
        from data.bciciv2a import SUBJECTS, load_subject_session
    elif name == "lee2019":
        from data.lee2019 import SUBJECTS, load_subject_session
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return list(SUBJECTS), load_subject_session


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="2a", choices=["2a", "lee2019"])
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=None,
                   help="override EEGNet n_epochs (cap cost on lee2019; classical ignore it)")
    p.add_argument("--out", default=None,
                   help="output CSV filename under results/ (default exp_<ds>_loso_extended.csv); "
                        "use a separate name for provisional/trial runs to keep the canonical file clean")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _device()
    subjects, load_subject_session = _load_dataset(args.dataset)
    names = args.methods or DEFAULT_METHODS[args.dataset]
    methods = make_methods(names, args.epochs, device)
    out_csv = REPO / "results" / (args.out if args.out else f"exp_{args.dataset}_loso_extended.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cross-subject LOSO | {args.dataset} | {len(subjects)} subj | {names} "
          f"| device={device} | epochs={args.epochs} | -> {out_csv}", flush=True)

    # idempotent: keep existing rows for methods we are NOT re-running
    cols = ["method", "subject", "protocol", "score"]
    if out_csv.exists():
        kept = pd.read_csv(out_csv)
        kept = kept[~kept["method"].isin(names)]
    else:
        kept = pd.DataFrame(columns=cols)

    data = {}
    for s in subjects:
        X_s, y_s = load_subject_session(subject=s, session="T")
        data[s] = (normalize_per_subject(X_s), y_s)

    new_rows = []
    for name in names:
        factory = methods[name]
        for held_out in subjects:
            try:
                score = evaluate_loso(factory, data, held_out, subjects)["score"]
                tag = f"{score:.4f}"
            except Exception as e:               # crash-safe: skip a bad fold, keep going
                score = float("nan")
                tag = f"FAILED ({type(e).__name__}: {e})"
            new_rows.append({"method": name, "subject": held_out,
                             "protocol": "LOSO", "score": score})
            # rewrite after every fold so a kill keeps progress
            pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True).to_csv(out_csv, index=False)
            print(f"  {name:10s} held-out s{held_out:02d}  {tag}", flush=True)

    print(f"\nwrote {out_csv}", flush=True)

    df = pd.DataFrame(new_rows)
    if args.dataset == "2a" and "CSP+LDA" in names:
        csp = df[df["method"] == "CSP+LDA"].set_index("subject")["score"]
        ref = pd.Series(CSP_LDA_LOSO_REF)
        max_diff = float((csp - ref).abs().max())
        print(f"[reproduction gate] CSP+LDA LOSO vs saved CSV: max|diff|={max_diff:.2e} "
              f"-> {'PASS' if max_diff < 1e-3 else 'FAIL'}")
    means = df.groupby("method")["score"].mean()
    print(f"\nLOSO mean accuracy ({args.dataset}):")
    for m in names:
        print(f"  {m:10s} {means[m]:.4f}")
    if {"TS-LR", "TS-LR-RPA"}.issubset(means.index):
        print(f"\n  RPA gain on LOSO: {means['TS-LR-RPA'] - means['TS-LR']:+.4f}")


if __name__ == "__main__":
    main()
