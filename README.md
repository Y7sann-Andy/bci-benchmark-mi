# MI Decoding Benchmark: Channel Reduction and Cross-Session Generalization

**BCI Competition IV-2a (n=9) and Lee2019 / OpenBMI (n=54).**

## Overview

A systematic comparison of motor-imagery (MI) decoding methods across two evaluation protocols, on two public datasets. Both protocols target conditions that matter for real deployment: limited electrode count (cost, comfort, setup time) and stability across recording days.

Datasets:

- **BCI Competition IV 2a**: 9 subjects, 4-class (primary)
- **Lee2019 / OpenBMI**: 54 subjects, 2-class (higher statistical power)

Methods compared:

- CSP + LDA
- Riemannian MDM (PyRiemann)
- Tangent Space + Logistic Regression (TS-LR)
- EEGNet (CNN)
- **TS-LR-RPA**: TS-LR with unsupervised Riemannian recentering (cross-session domain adaptation)

(The extended baseline also includes LSTM, CSP+MLP, and MDM/TS-LR metric variants.)

## Evaluation protocols

### Within-session (channel reduction)
5-fold CV within each session, progressively dropping electrodes (2a: 22, 16, 12, 8, 4; Lee2019: 62, 16, 12, 8, 4) with the motor-cortex ring preserved.

### Cross-session
Train on one session, test on another (different day), full montage. RPA is evaluated here as an unsupervised adaptation step.

Both protocols run on both datasets; degradation (within-session vs cross-session) and a cross-dataset κ comparison are derived from them.

## Results

**[`notebooks/results_2a_vs_lee2019.ipynb`](notebooks/results_2a_vs_lee2019.ipynb)**: full results with figures (channel reduction, cross-session, degradation, RPA recovery, cross-dataset comparison).

Observations:

- TS-LR (Riemannian tangent space) ranks first on both datasets, within- and cross-session. Under channel reduction it is also top at most channel counts; the exception is 2a at K=4, where the methods are not separable (Friedman p=0.14).
- EEGNet is competitive on 2a (4-class) but ranks last on Lee2019. With ~100 calibration trials/session, the data-hungry CNN trains less effectively than the data-efficient classical and Riemannian methods.
- RPA recovers ~70% of the cross-session drop (TS-LR retention 91.5% to 97.4%): p=2.5e-5 at n=54, only a trend at n=9 (p=0.055).
- Edge-deployment check (`experiments/run_onnx_deploy_check.py`): the EEGNet ONNX export runs a single 2 s window in ~0.17 ms (~3.5x faster than PyTorch eager, 3.4-3.6x across runs; 2,548 params, 13.6 KB), about 1500x under a 250 ms online sliding-window budget. On 9 trained subject models over 2,592 real held-out windows the export holds 100% label agreement (worst max|delta logit| 1.14e-5). The real-time bottleneck is the front-end (causal filtering / covariance), not the network.

Stats: Friedman + Wilcoxon (Holm-corrected), Cohen's dz effect sizes, κ-normalization for the cross-paradigm comparison. All numbers are generated from `results/*.csv` (see `experiments/_make_capstone_comparison.py`), never hand-typed.

## Status

Core benchmark complete on both datasets (2a n=9, Lee2019 n=54). Capstone write-up in progress.

## Repository layout

```
preprocessing/     shared preprocessing pipeline (n_channels parameterized)
methods/           one file per method, uniform interface
evaluation/        crossval + cross-session evaluation utilities
experiments/       experiment runners (one per research question)
notebooks/         results notebook
results/           CSV outputs from experiment runners
data/              raw + cached data (gitignored)
```
