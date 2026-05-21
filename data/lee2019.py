"""Lee2019 (OpenBMI) motor-imagery loader — mirrors data/bciciv2a.py.

Lee2019_MI via MOABB: 54 subjects, 2 sessions, **2-class** MI (left/right
hand), 62 EEG channels, native 1000 Hz. We band-pass, resample to 250 Hz,
and crop to the MI window so the output matches the (n_trials, n_ch, n_times)
contract the experiment runners already expect — and so EEGNet input size /
compute stay comparable to the BCICIV-2a runs.

IMPORTANT vs 2a: this is a **2-class** dataset (chance = 50%), not 4-class.
Lee2019 results are a robustness *replication of the qualitative pattern*
(which method degrades least cross-session; the accuracy-vs-channel-count
trend) on n=54 — NOT an absolute-number match to the 2a (4-class) figures.

Convention (matches data/bciciv2a.py so runners can swap loaders):
    X.shape = (n_trials, n_ch, 500)   # float, resampled to 250 Hz, 2 s window
    y.shape = (n_trials,)             # ∈ {0, 1}, from sorted event order
    SUBJECTS = 1..54
    load_subject_session(subject, session) -> (X, y), session in {"T", "E"}
"""
from __future__ import annotations

import numpy as np
import mne
from moabb.datasets import Lee2019_MI

mne.set_log_level("ERROR")

SUBJECTS = list(range(1, 55))          # 1..54
FS = 250                                # resample target (native 1000 Hz)
N_SAMPLES = 500                         # 2 s × 250 Hz
DEFAULT_BAND = (8.0, 30.0)             # μ + β, same as 2a loader

# MI crop window relative to imagery-cue onset (s). 2 s window to yield 500
# samples @250 Hz, matching the 2a [0.5, 2.5]s post-cue convention. If the
# smoke test shows ~chance accuracy, this window (or the event onset
# assumption) is wrong and must be retuned — see smoke test at bottom.
CROP_TMIN, CROP_TMAX = 0.5, 2.5
_EPOCH_TMAX = CROP_TMAX + 1.0          # epoch a bit wider, then crop

# Motor-priority ring for channel reduction (same anatomy logic as 2a's
# preprocessing.MOTOR_PRIORITY). All names present in Lee2019's 62-ch montage
# (validated at runtime in select_channels()). NB: Lee2019 has NO 'FCz' — the
# midline sensorimotor slot uses 'CPz' (the nearest available midline site).
MOTOR_PRIORITY: list[str] = [
    "C3", "C4",
    "Cz", "CPz",
    "FC3", "FC4", "CP3", "CP4",
    "C1", "C2", "C5", "C6",
    "FC1", "FC2", "CP1", "CP2",
]

# Nested subsets, strictly nested like 2a. None sentinel = full montage (62).
CHANNEL_SUBSETS: dict[int, list[str] | None] = {
    4:  MOTOR_PRIORITY[:4],
    8:  MOTOR_PRIORITY[:8],
    12: MOTOR_PRIORITY[:12],
    16: MOTOR_PRIORITY[:16],
    62: None,   # sentinel: no slicing, full montage
}

_DATASET = Lee2019_MI()


def _session_keys(subj_data: dict) -> list[str]:
    """Sorted session keys (Lee2019 has 2 per subject)."""
    return sorted(subj_data.keys())


def _concat_session_raw(subj_data: dict, session: str) -> mne.io.BaseRaw:
    """Concatenate the runs of one session into a single native-rate Raw.

    Keeps ALL channels (EEG + EMG + STIM): events live in the 'STI 014' stim
    channel and must be read upstream BEFORE the EEG pick. Session keys are the
    0-indexed strings '0'/'1' returned by _get_single_subject_data; 'T' selects
    the first, 'E' the second. No filter/resample here — the caller does that
    after pulling events out, so the stim impulses are never resampled.
    """
    keys = _session_keys(subj_data)
    if len(keys) < 2:
        raise RuntimeError(f"expected 2 sessions, got {keys}")
    sess_key = keys[0] if str(session) in ("T", "0", "train", keys[0]) else keys[1]
    runs = subj_data[sess_key]
    raws = [r.copy() for _, r in sorted(runs.items())]
    return mne.concatenate_raws(raws, verbose=False)


def load_subject_session(
    subject: int,
    session: str = "T",
    bandpass: tuple[float, float] | None = DEFAULT_BAND,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one (subject, session) of Lee2019 MI as trial-wise tensors.

    Parameters
    ----------
    subject : int, 1..54
    session : str
        "T" -> first session, "E" -> second session (Lee2019 has 2).
    bandpass : (low, high) or None
        Band-pass applied before resampling. Default (8, 30) Hz.

    Returns
    -------
    X : ndarray, (n_trials, n_ch, 500), float, 250 Hz, [0.5, 2.5]s MI window.
    y : ndarray, (n_trials,), int in {0, 1}.
    """
    # MOABB 1.5.0 bug: the public get_data() filters sessions with a 1-indexed
    # selected_sessions=(1,2) against the 0-indexed keys '0'/'1' it actually
    # produces, silently dropping session '0' (you get only one session). The
    # constructor validates sessions in [1,2], so there is no public way to make
    # the filter match — call the unfiltered loader directly to get BOTH sessions.
    subj = _DATASET._get_single_subject_data(subject)
    raw = _concat_session_raw(subj, session)
    # Lee2019 events live in a 'STI 014' stim channel (not annotations), so read
    # them at native rate BEFORE picking EEG / resampling. shortest_event=1 keeps
    # the single-sample cue impulses; consecutive handles back-to-back ids.
    events = mne.find_events(raw, stim_channel="STI 014",
                             shortest_event=1, consecutive=True, verbose=False)
    raw.pick("eeg")
    if bandpass is not None:
        raw.filter(bandpass[0], bandpass[1], fir_design="firwin",
                   skip_by_annotation="edge", verbose=False)
    epochs = mne.Epochs(
        raw, events, event_id=None,
        tmin=0.0, tmax=_EPOCH_TMAX, baseline=None,
        preload=True, picks="eeg", verbose=False,
    )
    if round(epochs.info["sfreq"]) != FS:   # resample epochs (events already applied)
        epochs.resample(FS, verbose=False)
    epochs = epochs.crop(CROP_TMIN, CROP_TMAX)
    X = epochs.get_data(copy=False).astype(float)
    if X.shape[-1] != N_SAMPLES:        # guard: window must yield 500 samples
        X = X[..., :N_SAMPLES]
    _, y = np.unique(epochs.events[:, -1], return_inverse=True)
    return X, y


def channel_names(subject: int = 1) -> list[str]:
    """EEG channel names in montage order (for channel-reduction indexing)."""
    subj = _DATASET._get_single_subject_data(subject)   # bypass get_data session-filter bug
    keys = _session_keys(subj)
    raw = list(subj[keys[0]].values())[0].copy().pick("eeg")
    return raw.ch_names


def select_channels(
    X: np.ndarray, names: list[str] | None, all_names: list[str]
) -> np.ndarray:
    """Slice X (n_trials, n_ch, n_times) to `names`, indexed via all_names order."""
    if names is None:
        return X
    missing = [n for n in names if n not in all_names]
    if missing:
        raise ValueError(f"channels not in Lee2019 montage: {missing}")
    idx = [all_names.index(n) for n in names]
    return X[:, idx, :]


if __name__ == "__main__":
    # Smoke test on subject 1 once downloaded. Accuracy sanity (>0.5 = window OK)
    # is checked by the runner, not here; here we just verify shapes & labels.
    for sess in ("T", "E"):
        X, y = load_subject_session(1, sess)
        print(f"subject 1 session {sess}: X={X.shape}  y={y.shape}  "
              f"classes={np.unique(y)}  counts={np.bincount(y)}")
    print("channels:", len(channel_names(1)))
    print(channel_names(1))
