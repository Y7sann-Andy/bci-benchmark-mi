"""
Shared preprocessing pipeline for BCI Competition IV 2a (BNCI2014_001).

Three functions, each taking the previous stage's output:

    raw     = load_and_preprocess(subject, session, ...)
    epochs  = make_epochs(raw, ...)
    X, y    = get_training_data(epochs, ...)

Design choices:
  - n_channels parameter on load_and_preprocess lets Experiment 1
    (channel reduction) be a one-liner loop in the experiment runner.
  - The MOTOR_PRIORITY list below defines the selection order:
    innermost (C3/C4) first, outermost (POz) last.
  - ICA is off by default. BCICIV 2a cue-locked epochs don't contain
    typical blink/ECG artifacts within the MI window (1-2s post-cue),
    so ICA on raw is not worth the complexity here.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import mne
from moabb.datasets import BNCI2014_001


# Priority ring for channel reduction. First N entries = which channels to keep
# when the caller passes n_channels=N. Order is motor-cortex outward:
#   n=2  → C3, C4 (bilateral M1 hand area)
#   n=4  → + Cz, FCz (SMA / foot / pre-motor midline)
#   n=8  → + FC3, FC4, CP3, CP4 (Hjorth-Laplacian neighbors)
#   n=16 → + C1, C2, C5, C6, FC1, FC2, CP1, CP2
#   n=22 → all BCICIV 2a channels (everything)
MOTOR_PRIORITY: list[str] = [
    "C3", "C4",
    "Cz", "FCz",
    "FC3", "FC4", "CP3", "CP4",
    "C1", "C2", "C5", "C6",
    "FC1", "FC2", "CP1", "CP2",
    "CPz", "Fz", "P1", "P2", "Pz", "POz",
]

# Default band-pass for motor imagery: 8-30 Hz covers mu + beta.
# 7 Hz lower edge (not 8) gives a softer filter skirt without touching alpha.
DEFAULT_L_FREQ = 7.0
DEFAULT_H_FREQ = 30.0


def load_and_preprocess(
    subject: int,
    session: str = "0train",
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
    n_channels: int | None = None,
    verbose: bool = False,
) -> mne.io.Raw:
    """Load one subject + one session of BCICIV 2a, concatenate runs, filter.

    Parameters
    ----------
    subject : int
        Subject ID (1..9).
    session : str
        'session_T' for training session, 'session_E' for evaluation session.
        (MOABB key names — BNCI2014_001 has 2 sessions per subject.)
    l_freq, h_freq : float
        Band-pass edges in Hz.
    n_channels : int or None
        If given, keep only the first n_channels from MOTOR_PRIORITY.
        If None, keep all 22 EEG channels.

    Returns
    -------
    raw : mne.io.Raw
        Filtered, channel-subsetted, reference-set Raw object ready for epoching.
    """
    dataset = BNCI2014_001()
    # MOABB returns {subject: {session: {run: raw}}}
    all_data = dataset.get_data(subjects=[subject])
    session_runs = all_data[subject][session]

    raws = [run_raw for _, run_raw in session_runs.items()]
    raw = mne.concatenate_raws(raws, verbose=verbose)

    # BCICIV 2a provides 22 EEG + 3 EOG. Drop EOG for MI decoding.
    raw.pick(picks="eeg")

    if n_channels is not None:
        keep = MOTOR_PRIORITY[:n_channels]
        missing = [ch for ch in keep if ch not in raw.ch_names]
        if missing:
            raise ValueError(
                f"Channels not found in raw: {missing}. "
                f"Available: {raw.ch_names}"
            )
        raw.pick(picks=keep)

    # Common average reference (CAR), applied as a projection.
    raw.set_eeg_reference("average", projection=True, verbose=verbose)

    # Band-pass filter. skip_by_annotation='edge' avoids filtering across the
    # run boundaries created by concatenate_raws.
    raw.filter(
        l_freq, h_freq,
        fir_design="firwin",
        skip_by_annotation="edge",
        verbose=verbose,
    )
    return raw


def make_epochs(
    raw: mne.io.Raw,
    tmin: float = -1.0,
    tmax: float = 4.0,
    event_labels: Sequence[str] | None = None,
    verbose: bool = False,
) -> mne.Epochs:
    """Extract events and build epochs from a preprocessed Raw.

    Parameters
    ----------
    raw : mne.io.Raw
        Output of load_and_preprocess.
    tmin, tmax : float
        Epoch window relative to cue onset (seconds).
    event_labels : sequence of str or None
        If given, keep only epochs matching these labels.
        For BCICIV 2a the 4 labels are 'left_hand', 'right_hand', 'feet', 'tongue'.
        If None, keep all 4 classes.
    """
    events, event_id = mne.events_from_annotations(raw, verbose=verbose)

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=(None, 0),
        preload=True,
        picks="eeg",
        verbose=verbose,
    )

    if event_labels is not None:
        epochs = epochs[list(event_labels)]
    return epochs


def get_training_data(
    epochs: mne.Epochs,
    crop_tmin: float = 1.0,
    crop_tmax: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop epochs to the motor-imagery window and return (X, y) for sklearn.

    Parameters
    ----------
    epochs : mne.Epochs
    crop_tmin, crop_tmax : float
        Window within the epoch to use for classification.
        Default 1.0-2.0s post-cue = where MI-related ERD is strongest
        (confirmed by sliding-window analysis on this dataset).

    Returns
    -------
    X : np.ndarray of shape (n_trials, n_channels, n_times)
    y : np.ndarray of shape (n_trials,) with integer class labels starting at 0.
    """
    epochs_cropped = epochs.copy().crop(tmin=crop_tmin, tmax=crop_tmax)
    X = epochs_cropped.get_data(copy=False)

    # Event codes in MNE start at 1. Subtract the minimum to get 0-indexed labels
    # (what sklearn / pytorch expect).
    raw_labels = epochs_cropped.events[:, -1]
    _, y = np.unique(raw_labels, return_inverse=True)
    return X, y


if __name__ == "__main__":
    # Smoke test: full pipeline on subject 1, session 0train.
    for sub in [1,2]:
        print(f"Loading subject={sub}, session=0train ...")
        raw = load_and_preprocess(subject=sub, session="0train", verbose=False)
        print(f"  Raw: {len(raw.ch_names)} channels, {raw.n_times} samples, "
            f"{raw.info['sfreq']} Hz")

        print("Building epochs ...")
        epochs = make_epochs(raw)
        print(f"  Epochs: {len(epochs)} trials, event_id={epochs.event_id}")

        print("Extracting training data ...")
        X, y = get_training_data(epochs)
        print(f"  X shape: {X.shape}   y shape: {y.shape}   classes: {np.unique(y)}")

        print("\nChannel-reduction sanity check (n_channels=4):")
        raw4 = load_and_preprocess(subject=sub, session="0train", n_channels=4)
        print(f"  Kept channels: {raw4.ch_names}")
