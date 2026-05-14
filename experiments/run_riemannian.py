"""Run the Riemannian MDM ablation on all 9 BCICIV-2a subjects.

Three metrics (riemann / logeuclid / euclid) x 9 subjects, within-session
5-fold CV. Reproduces the MDM block of results/exp_baseline_v2.csv.

Idempotent: re-reads the CSV, drops any existing MDM* rows, and writes back
the non-MDM rows plus a fresh MDM block. Re-running never duplicates rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.bciciv2a import SUBJECTS, load_subject_session
from evaluation.crossval import cross_validate_within_session
from methods.riemannian import RiemannianMDMClassifier

VARIANTS = [("MDM", "riemann"), ("MDM-logeuclid", "logeuclid"), ("MDM-euclid", "euclid")]
CSV_PATH = Path("results/exp_baseline_v2.csv")
FIELDNAMES = ["method", "subject", "protocol", "score"]
PROTOCOL = "within-CV"


def main() -> None:
    # Compute the fresh MDM block: 3 metrics x 9 subjects, within-session CV.
    mdm_rows: list[dict[str, object]] = []
    for subject in SUBJECTS:
        X, y = load_subject_session(subject, "T")
        for method, metric in VARIANTS:
            result = cross_validate_within_session(
                lambda m=metric: RiemannianMDMClassifier(metric=m), X, y
            )
            mdm_rows.append(
                {"method": method, "subject": subject,
                 "protocol": PROTOCOL, "score": result["mean"]}
            )
            print(f"{method:16s} subj {subject}  {result['mean']:.4f}")

    # Idempotent write: keep every non-MDM row, replace the MDM block wholesale.
    existing = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame(columns=FIELDNAMES)
    kept = existing[~existing["method"].str.startswith("MDM")]
    fresh = pd.DataFrame(mdm_rows)
    out = pd.concat([kept, fresh], ignore_index=True)
    out.to_csv(CSV_PATH, index=False, columns=FIELDNAMES)

    print(f"\nwrote {len(kept)} non-MDM + {len(fresh)} MDM rows -> {CSV_PATH}")


if __name__ == "__main__":
    main()
