"""Run the TS+LR ablation on all 9 BCICIV-2a subjects.

Two metrics (riemann / logeuclid) x 9 subjects, within-session 5-fold CV.
Appends TS-LR* rows to results/exp_baseline_v2.csv.

Idempotent: re-reads the CSV, drops any existing TS-LR* rows, and writes
back the non-TS-LR rows plus a fresh TS-LR block. Re-running never
duplicates rows.
"""
from __future__ import annotations

import csv
from pathlib import Path

from data.bciciv2a import SUBJECTS, load_subject_session
from evaluation.crossval import cross_validate_within_session
from methods.tangent_lr import TangentSpaceLRClassifier

VARIANTS = [("TS-LR", "riemann"), ("TS-LR-logeuclid", "logeuclid")]
CSV_PATH = Path("results/exp_baseline_v2.csv")
FIELDNAMES = ["method", "subject", "protocol", "score"]
PROTOCOL = "within-CV"


def main() -> None:
    # Compute the fresh TS-LR block: 2 metrics x 9 subjects, within-session CV.
    ts_lr_rows: list[dict[str, object]] = []
    for subject in SUBJECTS:
        X, y = load_subject_session(subject, "T")
        for method, metric in VARIANTS:
            result = cross_validate_within_session(
                lambda m=metric: TangentSpaceLRClassifier(tangent_metric=m), X, y
            )
            ts_lr_rows.append(
                {"method": method, "subject": subject,
                 "protocol": PROTOCOL, "score": result["mean"]}
            )
            print(f"{method:18s} subj {subject}  {result['mean']:.4f}")

    # Idempotent write: keep every non-TS-LR row, replace the TS-LR block wholesale.
    with CSV_PATH.open(newline="") as f:
        kept_rows = [row for row in csv.DictReader(f)
                     if not row["method"].startswith("TS-LR")]

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept_rows)
        writer.writerows(ts_lr_rows)

    print(f"\nwrote {len(kept_rows)} non-TS-LR + {len(ts_lr_rows)} TS-LR rows -> {CSV_PATH}")


if __name__ == "__main__":
    main()
