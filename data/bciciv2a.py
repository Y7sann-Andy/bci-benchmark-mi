"""BCI Competition IV Dataset 2a loader.

Loads pre-downloaded .mat files from ~/mne_data/MNE-bnci-data/database/001-2014/.
Path layout was repaired manually on 2026-04-29 (MOABB sanitization bug —
see BCICIV2a_path_notes.md in the learning repo for the original mv command).

Pipeline applied per session (in order):
    raw .mat run → drop EOG (last 3 of 25 ch) → bandpass on continuous run
    (default 8-30 Hz μ/β band) → epoch [+0.5s, +2.5s] post-cue
    → label shift {1..4} → {0..3} for PyTorch CrossEntropyLoss compatibility.

Trial structure (BCICIV 2a, 8s/trial):
    [0.0s, 2.0s)  fixation cross
    [2.0s, 3.25s) cue arrow (left/right/feet/tongue)
    [3.0s, 6.0s)  motor imagery period
    [6.0s, 8.0s)  rest

`sub['trial']` gives the sample index of fixation onset (t=0). The MI epoch
window [+0.5s, +2.5s] post-cue translates to samples [ts+625, ts+1125)
(post-trial-start [+2.5s, +4.5s)) — covers the cue → early-MI transition,
which is where μ/β ERD is strongest.

Convention:
    X.shape = (n_trials, 22, 500)   # 22 EEG, fs=250 Hz, 2s window
    X units = microvolts (μV); raw .mat is already μV-scaled, no conversion done.
    y.shape = (n_trials,)           # ∈ {0, 1, 2, 3} — 0=left, 1=right, 2=feet, 3=tongue
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import mne


DATA_ROOT = Path.home() / "mne_data" / "MNE-bnci-data" / "database" / "001-2014"

SUBJECTS = list(range(1, 10))   # 1..9
N_CHANNELS = 22                 # EOG dropped (last 3 of original 25)
N_SAMPLES = 500                 # 2s × 250 Hz, [+0.5s, +2.5s) post-cue
FS = 250                        # sampling rate


def load_subject_session(                                                        
    subject: int,                                                                                                
    session: str = "T",                                                                                          
    bandpass: tuple[float, float] | None = (8.0, 30.0),
) -> tuple[np.ndarray, np.ndarray]:                                                                              
    """Load one (subject, session) of BCICIV 2a as trial-wise tensors.

    Parameters
    ----------
    subject : int, 1..9
    session : str
        'T' (training, session A) or 'E' (evaluation, session B).
    bandpass : (low_freq, high_freq) or None
        Bandpass filter applied to each continuous run BEFORE epoching — filtering
        2-second epochs separately would introduce edge artifacts that contaminate
        the discriminative signal. Default (8, 30) Hz — the MI μ/β band, standard
        for motor imagery decoding (Lotte et al. 2018). Pass None to skip filtering
        (broadband output, retains only the 0.5-100 Hz hardware acquisition filter).

    Returns
    -------
    X : ndarray, shape (n_trials, 22, 500), float
        EEG epochs in microvolts, EOG channels dropped, cue-aligned 0.5–2.5s window,
        optionally bandpass-filtered.
    y : ndarray, shape (n_trials,), int
        Class labels in {0, 1, 2, 3}. Source MAT labels are 1..4; the loader
        subtracts 1 so PyTorch CrossEntropyLoss can consume directly.

    Notes
    -----
    Each session has 9 runs in the MAT file. The first ~3 are calibration
    (eyes-open / eyes-closed / artifact) and have empty `y` — skip them.
    The remaining 6 are MI runs, 48 trials each → 288 trials per session.
    """                                                                          
    path = DATA_ROOT / f"A{subject:02d}{session}.mat"                                                            
    mat = sio.loadmat(str(path))                                                                                 
                                                                                                                 
    X_list, y_list = [], []                                                                                      
                                                                                                                 
    for run in mat['data'][0]:                                                                                   
        sub = run[0, 0]                                                                                          
        if 'X' not in sub.dtype.names or sub['y'].size == 0:                                                     
            continue                                                                                             
                                                                                 
        X_run = sub['X'][:, :22]                # (n_samples, 22), drop EOG early

        if bandpass is not None:
            X_run = mne.filter.filter_data(X_run.T.astype(float), sfreq=FS,
                                            l_freq=bandpass[0], h_freq=bandpass[1], verbose=False).T                           
                                                                                                                 
        y_run = sub['y'].flatten()                                                                               
        trial_starts = sub['trial'].flatten()                                                                    
                                                                                                                 
        for ts, label in zip(trial_starts, y_run):                                                               
            epoch = X_run[ts + 625 : ts + 1125, :].T   # (22, 500)                                                
            X_list.append(epoch)                                                                                 
            y_list.append(label - 1)                                                                             
                                                                 
    X = np.stack(X_list)                                                                                         
    y = np.array(y_list)                                                                                         
    return X, y 