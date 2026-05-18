"""Statistical tests for the cross-method benchmark results.

Tests:
1. EEGNet vs CSP+LDA at K=4 (paired Wilcoxon) — does EEGNet really win at low channels?
1b. EEGNet vs TS-LR at K=4 (paired Wilcoxon)
2. EEGNet within-CV vs cross-session (paired Wilcoxon) — does EEGNet really not degrade?
2b. TS-LR within-CV vs cross-session (for contrast — should show degradation)
3. Friedman test across 4 methods at each K — methods systematically differ?
4. Pre vs post alignment for shallow methods (paired t-test) — z-score harmless?
5. MPS vs CPU EEGNet within-CV (paired Wilcoxon) — devices equivalent?

n=9 subjects → underpowered for small effects. Report Cohen's d alongside p.

Private/temp script. Saves nothing — output to stdout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CR_CSV = Path("results/exp_channel_reduction.csv")
BASE_CSV = Path("results/exp_baseline_v2.csv")
PRE_ALIGN_CR = Path("/tmp/exp_channel_reduction.before_align.csv")
DEVICE_MPS_CSV = Path("/tmp/eegnet_mps_compare.csv")


def cohen_d_paired(diff):
    """Cohen's d for paired samples = mean_diff / std_diff."""
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def interpret_d(d):
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


def report_paired(name, x, y, test="wilcoxon"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    n = len(diff)
    print(f"--- {name} ---")
    print(f"  n={n}, mean diff = {diff.mean():+.4f} (std {diff.std(ddof=1):.4f})")

    if test == "wilcoxon":
        try:
            stat, p = stats.wilcoxon(x, y)
            test_name = "Wilcoxon signed-rank"
        except ValueError as e:
            print(f"  Wilcoxon failed ({e}); falling back to paired t-test")
            stat, p = stats.ttest_rel(x, y)
            test_name = "Paired t-test (fallback)"
    else:
        stat, p = stats.ttest_rel(x, y)
        test_name = "Paired t-test"

    d = cohen_d_paired(diff)
    sig = "YES" if (not np.isnan(p) and p < 0.05) else "NO"
    print(f"  {test_name}: stat = {stat:.3f}, p = {p:.4f}, sig (α=0.05): {sig}")
    print(f"  Cohen's d (paired) = {d:+.3f} ({interpret_d(d)})")
    print()


def main() -> None:
    print("=" * 64)
    print("Statistical Tests — Channel Reduction + Cross-Session")
    print("=" * 64)
    print("n=9 subjects. Many tests will be underpowered for small effects.")
    print("Report Cohen's d alongside p-values.\n")

    cr = pd.read_csv(CR_CSV)
    base = pd.read_csv(BASE_CSV)

    # === Test 1: EEGNet vs CSP+LDA at K=4 ===
    eeg_k4 = cr[(cr['method'] == 'EEGNet') & (cr['n_channels'] == 4)].sort_values('subject')['score'].values
    csp_k4 = cr[(cr['method'] == 'CSP+LDA') & (cr['n_channels'] == 4)].sort_values('subject')['score'].values
    tslr_k4 = cr[(cr['method'] == 'TS-LR') & (cr['n_channels'] == 4)].sort_values('subject')['score'].values
    mdm_k4 = cr[(cr['method'] == 'MDM') & (cr['n_channels'] == 4)].sort_values('subject')['score'].values
    report_paired("Test 1: EEGNet vs CSP+LDA at K=4", eeg_k4, csp_k4)
    report_paired("Test 1b: EEGNet vs TS-LR at K=4", eeg_k4, tslr_k4)
    report_paired("Test 1c: EEGNet vs MDM at K=4", eeg_k4, mdm_k4)

    # === Test 2: EEGNet within-CV vs cross-session ===
    eeg_wcv = base[(base['method'] == 'EEGNet') & (base['protocol'] == 'within-CV')].sort_values('subject')['score'].values
    eeg_cs = base[(base['method'] == 'EEGNet') & (base['protocol'] == 'cross-session')].sort_values('subject')['score'].values
    report_paired("Test 2: EEGNet cross-session vs within-CV (does it degrade?)", eeg_cs, eeg_wcv)

    tslr_wcv = base[(base['method'] == 'TS-LR') & (base['protocol'] == 'within-CV')].sort_values('subject')['score'].values
    tslr_cs = base[(base['method'] == 'TS-LR') & (base['protocol'] == 'cross-session')].sort_values('subject')['score'].values
    report_paired("Test 2b: TS-LR cross-session vs within-CV (for contrast)", tslr_cs, tslr_wcv)

    csp_wcv = base[(base['method'] == 'CSP+LDA') & (base['protocol'] == 'within-CV')].sort_values('subject')['score'].values
    csp_cs = base[(base['method'] == 'CSP+LDA') & (base['protocol'] == 'cross-session')].sort_values('subject')['score'].values
    report_paired("Test 2c: CSP+LDA cross-session vs within-CV (for contrast)", csp_cs, csp_wcv)

    # === Test 2d: does RPA domain adaptation recover TS-LR cross-session? ===
    tslr_rpa_cs = base[(base['method'] == 'TS-LR-RPA') & (base['protocol'] == 'cross-session')].sort_values('subject')['score'].values
    if len(tslr_rpa_cs) == len(tslr_cs) and len(tslr_rpa_cs) > 0:
        report_paired("Test 2d: TS-LR-RPA vs TS-LR cross-session (does RPA recover?)", tslr_rpa_cs, tslr_cs)
        print(f"  retention: TS-LR {100*tslr_cs.mean()/tslr_wcv.mean():.1f}% -> "
              f"TS-LR-RPA {100*tslr_rpa_cs.mean()/tslr_wcv.mean():.1f}% "
              f"(within-CV ceiling {tslr_wcv.mean():.4f})\n")

    # === Test 3: Friedman across methods at each K ===
    print("--- Test 3: Friedman test — 4 methods at each K ---")
    print("  H0: methods produce same scores; H1: at least one method differs.\n")
    for K in [4, 8, 12, 16, 22]:
        groups = []
        for m in ['CSP+LDA', 'MDM', 'TS-LR', 'EEGNet']:
            s = cr[(cr['method'] == m) & (cr['n_channels'] == K)].sort_values('subject')['score'].values
            groups.append(s)
        stat, p = stats.friedmanchisquare(*groups)
        sig = "YES" if p < 0.05 else "NO"
        print(f"  K={K:2d}: χ² = {stat:6.3f}, p = {p:.4f}, sig (α=0.05): {sig}")
    print()

    # === Test 4: z-score harmless for shallow methods ===
    print("--- Test 4: Pre vs post alignment for shallow methods ---")
    print("  H0: z-score has no effect on scores (delta=0); H1: delta != 0\n")
    if PRE_ALIGN_CR.exists():
        pre = pd.read_csv(PRE_ALIGN_CR)
        for m in ['CSP+LDA', 'MDM', 'TS-LR']:
            if m in pre['method'].unique():
                new = cr[cr['method'] == m].sort_values(['subject', 'n_channels'])['score'].values
                old = pre[pre['method'] == m].sort_values(['subject', 'n_channels'])['score'].values
                report_paired(f"Test 4: {m} aligned vs pre-aligned (n=45)", new, old, test="ttest")
    else:
        print(f"  Pre-alignment backup not found at {PRE_ALIGN_CR}; skipping.\n")

    # === Test 5: MPS vs CPU EEGNet ===
    if DEVICE_MPS_CSV.exists():
        mps_df = pd.read_csv(DEVICE_MPS_CSV).sort_values('subject')
        mps = mps_df['score'].values
        cpu = base[(base['method'] == 'EEGNet') & (base['protocol'] == 'within-CV')].sort_values('subject')['score'].values
        report_paired("Test 5: EEGNet MPS vs CPU within-CV (devices equivalent?)", mps, cpu)
    else:
        print(f"  Device comparison CSV not found at {DEVICE_MPS_CSV}; skipping.\n")

    # === Interpretation guide ===
    print("=" * 64)
    print("Interpretation guide")
    print("=" * 64)
    print("p < 0.05: reject H0 (a real difference, given the data)")
    print("p >= 0.05 with d > 0.5: likely real effect, underpowered (need more subjects)")
    print("p >= 0.05 with d < 0.2: probably no meaningful effect")
    print("Cohen's d: 0.2=small, 0.5=medium, 0.8=large (Cohen 1988 conventions)")


if __name__ == "__main__":
    main()
