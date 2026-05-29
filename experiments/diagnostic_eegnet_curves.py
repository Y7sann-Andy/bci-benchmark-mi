"""EEGNet train-vs-cross-session learning-curve diagnostic (NOT part of the benchmark).

Purpose: produce a per-epoch train-vs-cross-session (E) learning curve for EEGNet,
aggregated across subjects as a mean +/- std envelope, so we can see whether
EEGNet overfits MORE on the scarce-calibration Lee2019 data than on 2a. This
is the mechanistic evidence behind the Finding-3 "deep nets are data-hungry"
claim, and a defense against "did EEGNet lose because you trained it badly?".

This is a DIAGNOSTIC, separate from the benchmark. It does not touch the
canonical results CSVs and uses a single full-montage training run per subject
(the light cross-session-style path), not the channel-reduction sweep.

Output: a mean+/-band figure (train vs cross-session E) per dataset, plus the raw per-epoch
arrays saved to npz so the figure is reproducible without retraining.

Usage
-----
    # 2a, all 9 subjects
    python -m experiments.diagnostic_eegnet_curves --dataset 2a

    # Lee2019, all 54 subjects (background this while building the PPT)
    python -m experiments.diagnostic_eegnet_curves --dataset lee2019

    # smoke test: 2 subjects, 30 epochs
    python -m experiments.diagnostic_eegnet_curves --dataset 2a --subjects 1 2 --epochs 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import mne
mne.set_log_level("WARNING")

from methods.eegnet import _EEGNet
from preprocessing import normalize_per_session

OUT_DIR = Path("results/diagnostics")

# Per-epoch history keys. "eval_*" is the cross-session E set, monitored for the
# learning curve only (never used for early stopping or model selection).
HISTORY_KEYS = ("epoch", "train_loss", "train_acc", "eval_loss", "eval_acc")


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_dataset(name: str):
    """Return (module, subjects) for the chosen dataset."""
    if name == "2a":
        from data import bciciv2a as mod
    elif name == "lee2019":
        from data import lee2019 as mod
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return mod, list(mod.SUBJECTS)


def _to_tensor(X: np.ndarray, y: np.ndarray, device: str):
    """(trials, ch, samples) -> model-ready tensors. Data is already z-scored
    (normalize_per_session), so no microvolt rescaling here (scale=1)."""
    Xt = torch.from_numpy(X).float().unsqueeze(1).to(device)   # (N, 1, ch, samples)
    yt = torch.from_numpy(y).long().to(device)
    return Xt, yt


@torch.no_grad()
def _evaluate(model: nn.Module, Xt: torch.Tensor, yt: torch.Tensor,
              criterion: nn.Module) -> tuple[float, float]:
    """Mean loss and accuracy of `model` on a tensor set. Sets eval() then
    restores train() so it is safe to call inside the training loop."""
    was_training = model.training
    model.eval()
    logits = model(Xt)
    loss = criterion(logits, yt).item()
    acc = (logits.argmax(dim=1) == yt).float().mean().item()
    if was_training:
        model.train()
    return loss, acc


def train_with_history(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    *, n_epochs: int, batch_size: int, lr: float, dropout: float,
    device: str, seed: int = 42,
) -> dict[str, list]:
    """Train EEGNet on one subject, logging train and cross-session curves per epoch.

    X_tr/y_tr : the training session ('T'); the model is trained on this.
    X_te/y_te : the evaluation session ('E'); monitored each epoch to trace
                cross-session generalization. Plot-only: never used for early
                stopping or model selection, so it does not leak into the
                benchmark's reported cross-session score.

    Returns a dict with keys HISTORY_KEYS, each a list of length n_epochs.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_channels, n_samples = X_tr.shape[1], X_tr.shape[2]
    n_classes = int(np.unique(y_tr).size)

    # Full training-session tensors and the evaluation-session tensors.
    Xtr_t, ytr_t = _to_tensor(X_tr, y_tr, device)
    Xte_t, yte_t = _to_tensor(X_te, y_te, device)

    model = _EEGNet(n_channels, n_samples, n_classes, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list] = {k: [] for k in HISTORY_KEYS}

    # drop_last=True: BatchNorm needs >1 sample per batch in train mode, so a
    # size-1 tail batch would crash; dropping it is standard practice with BN.
    data_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True, drop_last=True,
    )

    model.train()
    for epoch in range(n_epochs):
        for batch_X, batch_y in data_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        eval_loss, eval_acc = _evaluate(model, Xte_t, yte_t, criterion)
        train_loss, train_acc = _evaluate(model, Xtr_t, ytr_t, criterion)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["eval_loss"].append(eval_loss)
        history["eval_acc"].append(eval_acc)

    return history


def _aggregate(histories: list[dict[str, list]]) -> dict[str, np.ndarray]:
    """Stack per-subject curves into mean +/- std envelopes.

    Returns epoch axis plus mean/std arrays for train and cross-session loss/acc.
    """
    if not histories or not histories[0]["epoch"]:
        raise RuntimeError("No per-epoch history recorded; nothing to aggregate.")
    epochs = np.asarray(histories[0]["epoch"])
    out = {"epoch": epochs, "n_subjects": len(histories)}
    for key in ("train_loss", "train_acc", "eval_loss", "eval_acc"):
        stacked = np.stack([np.asarray(h[key]) for h in histories])  # (n_subj, n_epoch)
        out[f"{key}_mean"] = stacked.mean(axis=0)
        out[f"{key}_std"] = stacked.std(axis=0)
    return out


def _plot(agg: dict[str, np.ndarray], dataset: str, out_png: Path) -> None:
    """Train vs cross-session (E) accuracy with shaded std bands. The gap between
    the two curves, and how it widens with epochs, is the overfit signal; compare
    across datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = agg["epoch"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, color, label in (("train", "C0", "train acc"),
                              ("eval", "C1", "cross-session (E) acc")):
        mean = agg[f"{key}_acc_mean"]
        std = agg[f"{key}_acc_std"]
        ax.plot(ep, mean, color=color, label=label)
        ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(f"EEGNet learning curve ({dataset}, n={agg['n_subjects']}, mean +/- std)")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["2a", "lee2019"])
    p.add_argument("--subjects", nargs="+", type=int, default=None,
                   help="subject IDs (default: all for the dataset)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _pick_device()
    mod, all_subjects = _load_dataset(args.dataset)
    subjects = args.subjects if args.subjects is not None else all_subjects
    print(f"EEGNet curve diagnostic | {args.dataset} | {len(subjects)} subjects "
          f"| {args.epochs} epochs | device={device}", flush=True)

    histories = []
    for i, s in enumerate(subjects, 1):
        X_tr, y_tr = mod.load_subject_session(s, "T")
        X_te, y_te = mod.load_subject_session(s, "E")
        X_tr, X_te = normalize_per_session(X_tr), normalize_per_session(X_te)
        h = train_with_history(
            X_tr, y_tr, X_te, y_te,
            n_epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, dropout=args.dropout, device=device,
        )
        histories.append(h)
        last = (h["eval_acc"][-1] if h["eval_acc"] else float("nan"))
        print(f"  [{i}/{len(subjects)}] s{s:02d} done; final E acc={last:.4f}", flush=True)

    agg = _aggregate(histories)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / f"eegnet_curve_{args.dataset}.npz"
    np.savez(npz_path, **agg)
    print(f"wrote {npz_path}")
    _plot(agg, args.dataset, OUT_DIR / f"eegnet_curve_{args.dataset}.png")


if __name__ == "__main__":
    main()
