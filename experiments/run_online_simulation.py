"""Causal online MI decoding simulation (multi-run).

Trains TS+LR (Covariances → TangentSpace → LR) on subject 1 session T MI epochs,
then slides W=2s windows across ALL session E runs with S=0.25s stride and
K=3-frame smoothing. Plots run 0 timeline (truth vs raw vs smoothed) to results/.

TS+LR chosen over CSP+LDA because within-CV on this dataset showed TS-LR=0.691
vs CSP+LDA=0.623. Riemannian tangent-space features
let LR exploit covariance geometry that CSP's log-variance feature compresses away.

Key discipline:
  - Single-pass causal 4th-order Butterworth IIR (8-30 Hz), NOT filtfilt.
  - Training epochs extracted from causally-filtered continuous data, so
    train and inference share the same filter (train-test consistency).
  - Stateful zi initialization suppresses session-start transient.
"""
from __future__ import annotations
from collections import Counter, deque
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter, lfilter_zi
import matplotlib.pyplot as plt

from methods.tangent_lr import TangentSpaceLRClassifier


# --- Constants ---
DATA_ROOT = Path.home() / "mne_data" / "MNE-bnci-data" / "database" / "001-2014"
RESULTS_DIR = Path("results")
FS, N_CHANNELS = 250, 22
L_FREQ, H_FREQ, ORDER = 8.0, 30.0, 4
EPOCH_START = int(2.5 * FS)   # +0.5s post-cue (cue at +2s post-fixation)
EPOCH_END   = int(4.5 * FS)   # +2.5s post-cue
W = EPOCH_END - EPOCH_START   # 500 samples = 2s — must match training window
S = int(0.25 * FS)            # 62 samples ≈ 0.25s stride
K = 3
IDLE_THRESHOLD = 0.60    # Strategy 3: min LDA posterior to emit a class; below → idle (-1).
                          # LDA's predict_proba is overconfident — 0.45 barely rejected anything.
                          # 0.60 is the empirical sweet spot for this dataset (see sweep results).
SUBJECT = 1
CLASS_NAMES = ['left', 'right', 'feet', 'tongue']


# --- Loaders & filters ---
def load_session_runs(subject: int, session: str) -> list[dict]:
    """Return list of {'signal', 'trial_starts', 'labels'} for MI runs only.

    Skips calibration runs (which have empty 'y' field).
    """
    path = DATA_ROOT / f"A{subject:02d}{session}.mat"
    mat = sio.loadmat(str(path))
    runs = []
    for entry in mat['data'][0]:
        sub = entry[0, 0]
        if 'X' not in sub.dtype.names or sub['y'].size == 0:
            continue
        runs.append({
            'signal': sub['X'][:, :N_CHANNELS].T.astype(float),     # (22, n_samples), uV
            'trial_starts': sub['trial'].flatten().astype(int),
            'labels': sub['y'].flatten().astype(int) - 1,           # → {0..3}
        })
    return runs


def causal_bandpass(signal: np.ndarray) -> np.ndarray:
    """Causal 4th-order Butterworth bandpass 8-30 Hz, single-pass with zi init.

    zi seeded from steady-state for a constant input matching the first sample,
    so the session-start transient is suppressed.
    """
    b, a = butter(ORDER, [L_FREQ, H_FREQ], btype='band', fs=FS)
    # lfilter_zi returns (filter_state_size,) = (8,) for 4th-order bandpass.
    # Tile across channels: zi shape = (n_channels, filter_state_size) = (22, 8),
    # each channel's state seeded by its own first sample (suppresses transient).
    zi_template = lfilter_zi(b, a)                       # (8,)
    zi = zi_template[None, :] * signal[:, [0]]           # (1, 8) * (22, 1) → (22, 8)
    filtered, _ = lfilter(b, a, signal, axis=1, zi=zi)
    return filtered


def extract_epochs(filtered: np.ndarray, trial_starts: np.ndarray,
                   labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slice (W,) epochs from filtered continuous signal at [+0.5s, +2.5s] post-cue."""
    epochs = np.stack([filtered[:, ts + EPOCH_START : ts + EPOCH_END]
                       for ts in trial_starts])
    return epochs, labels


def build_label_timeline(n_samples: int, trial_starts: np.ndarray,
                         labels: np.ndarray) -> np.ndarray:
    """Per-sample ground truth: class label during MI window, -1 elsewhere (rest)."""
    timeline = np.full(n_samples, -1, dtype=int)
    for ts, lbl in zip(trial_starts, labels):
        timeline[ts + EPOCH_START : ts + EPOCH_END] = lbl
    return timeline


# --- Inference (YOUR PART) ---
def simulate_one_run(test_run: dict, clf,
                     threshold: float = IDLE_THRESHOLD,
                     strict_dwell: bool = False):
    """Slide W-window across one test run; return per-emit arrays + metrics.

    Parameters
    ----------
    threshold : float
        Strategy 3 — min LDA posterior to emit a class. Below → idle (-1).
        Use 0.0 to disable (no idle gate; always emit argmax).
    strict_dwell : bool
        Strategy 4 — if True, require ALL K consecutive predictions to agree
        AND be non-idle before emitting. Otherwise → idle (-1).
        If False, use soft K-frame majority vote (variance reduction only).

    Returns
    -------
    times, raw_preds, smooth_preds, truth : np.ndarray   (each length n_emits)
    metrics : dict with mi_correct, mi_wrong, mi_idle, mi_total, rest_total, rest_trigger
    """
    test_filtered = causal_bandpass(test_run['signal'])
    truth_timeline = build_label_timeline(
        test_filtered.shape[1], test_run['trial_starts'], test_run['labels'])

    times_list, raw_list, smooth_list, truth_list = [], [], [], []
    pred_history = deque(maxlen=K)
    mi_correct = mi_wrong = mi_idle = mi_total = 0
    rest_total = rest_trigger = 0

    for t in range(W, test_filtered.shape[1] + 1, S):
        window = test_filtered[:, t-W:t][np.newaxis]
        # TS+LR pipeline accepts raw (n_trials, n_channels, n_times) — no manual CSP step
        proba = clf.pipe_.predict_proba(window)[0]
        max_conf = float(np.max(proba))
        # Strategy 3: confidence gate
        raw_pred = int(np.argmax(proba)) if max_conf >= threshold else -1

        pred_history.append(raw_pred)
        if len(pred_history) < K:
            smoothed_pred = raw_pred
        else:
            most_common, count = Counter(pred_history).most_common(1)[0]
            if strict_dwell:
                # Strategy 4: emit only if all K agree AND not idle, else abstain
                smoothed_pred = most_common if (count == K and most_common != -1) else -1
            else:
                # Soft majority vote — variance reduction, no abstention discipline
                smoothed_pred = most_common

        ground_truth = truth_timeline[t - W//2]
        if ground_truth >= 0:
            mi_total += 1
            if smoothed_pred == ground_truth:
                mi_correct += 1
            elif smoothed_pred == -1:
                mi_idle += 1
            else:
                mi_wrong += 1
        else:
            rest_total += 1
            if smoothed_pred != -1:
                rest_trigger += 1

        times_list.append(t / FS)
        raw_list.append(raw_pred)
        smooth_list.append(smoothed_pred)
        truth_list.append(ground_truth)
    # ===== END INFERENCE LOOP =====

    return (np.array(times_list), np.array(raw_list),
            np.array(smooth_list), np.array(truth_list),
            {'mi_correct': mi_correct, 'mi_wrong': mi_wrong, 'mi_idle': mi_idle,
             'mi_total': mi_total, 'rest_total': rest_total, 'rest_trigger': rest_trigger})


# --- Plotting ---
def plot_run_timeline(times: np.ndarray, raw: np.ndarray, smooth: np.ndarray,
                       truth: np.ndarray, run_idx: int) -> Path:
    """Two stacked colored strips: ground truth (top, white during rest) vs
    smoothed prediction (bottom, always colored). Same palette for both —
    visual color-match means correct prediction.
    """
    from matplotlib.patches import Patch

    fig, (ax_truth, ax_pred) = plt.subplots(
        2, 1, figsize=(14, 3.2), sharex=True, gridspec_kw={'hspace': 0.15})

    class_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # left/right/feet/tongue

    # Top strip: ground truth — colored band during MI, white during rest
    in_mi = truth >= 0
    if in_mi.any():
        edges = np.diff(in_mi.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends   = np.where(edges == -1)[0] + 1
        if in_mi[0]:  starts = np.r_[0, starts]
        if in_mi[-1]: ends   = np.r_[ends, len(in_mi)]
        for s, e in zip(starts, ends):
            ax_truth.axvspan(times[s], times[e-1], color=class_colors[int(truth[s])])
    ax_truth.set_ylim(0, 1); ax_truth.set_yticks([])
    ax_truth.set_ylabel('truth', rotation=0, ha='right', va='center', labelpad=20)
    ax_truth.set_title(f'Online MI decoding — subject {SUBJECT}, session E run {run_idx}  '
                       f'(W={W/FS}s, S={S/FS:.2f}s, K={K})')

    # Bottom strip: smoothed prediction — white where classifier abstained (idle, -1)
    change_idx = np.where(np.diff(smooth) != 0)[0] + 1
    seg_starts = np.r_[0, change_idx]
    seg_ends   = np.r_[change_idx, len(smooth)]
    for s, e in zip(seg_starts, seg_ends):
        if int(smooth[s]) == -1:
            continue                       # leave white (abstention)
        ax_pred.axvspan(times[s], times[min(e, len(times)) - 1],
                        color=class_colors[int(smooth[s])])
    ax_pred.set_ylim(0, 1); ax_pred.set_yticks([])
    ax_pred.set_ylabel('smoothed\npred', rotation=0, ha='right', va='center', labelpad=20)
    ax_pred.set_xlabel('time (s)')

    # Legend
    legend = [Patch(facecolor=class_colors[i], label=CLASS_NAMES[i]) for i in range(4)]
    legend.append(Patch(facecolor='white', edgecolor='gray',
                         label='no MI (truth) / abstain (pred)'))
    fig.legend(handles=legend, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f'online_simulation_run{run_idx}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


# --- Orchestration ---
def main() -> None:
    # 1. Train on all session T runs
    print(f"[Train] subject {SUBJECT}, session T")
    train_X_list, train_y_list = [], []
    for run in load_session_runs(SUBJECT, "T"):
        filtered = causal_bandpass(run['signal'])
        X, y = extract_epochs(filtered, run['trial_starts'], run['labels'])
        train_X_list.append(X); train_y_list.append(y)
    X_train = np.concatenate(train_X_list)
    y_train = np.concatenate(train_y_list)
    print(f"  X_train: {X_train.shape}  classes: {np.unique(y_train)}")
    clf = TangentSpaceLRClassifier().fit(X_train, y_train)
    print(f"  TS+LR fitted")

    # 2. Sweep across 4 strategy configurations on all session E runs
    print(f"\n[Sweep] subject {SUBJECT}, session E — comparing 4 idle-detection configurations")
    test_runs = load_session_runs(SUBJECT, "E")

    configs = [
        {'name': 'no gate + soft K=3 vote (baseline)',  'threshold': 0.00, 'strict_dwell': False},
        {'name': 'no gate + strict K=3 dwell (S4 only)','threshold': 0.00, 'strict_dwell': True},
        {'name': f'τ={IDLE_THRESHOLD} + soft K=3 vote (S3 only)',
                                                          'threshold': IDLE_THRESHOLD, 'strict_dwell': False},
        {'name': f'τ={IDLE_THRESHOLD} + strict K=3 dwell (S3+S4)',
                                                          'threshold': IDLE_THRESHOLD, 'strict_dwell': True},
    ]

    # Metric definitions (correct async-BCI nomenclature):
    #   det_TPR   = (correct + wrong) / mi_total = 1 - reject
    #               "of MI windows, did the system emit any non-idle command" (binary detection)
    #   precision = correct / (correct + wrong)
    #               "of emitted commands during MI, fraction with the right class"
    #   cmd_acc   = correct / mi_total = det_TPR * precision
    #               "end-to-end: of MI windows, fraction where user got the right command"
    #   FPR/min   = rest_trigger / rest_minutes
    #               "false trigger rate during rest periods"
    header = (f"  {'config':<48} {'det_TPR':>8}  {'precision':>9}  "
              f"{'cmd_acc':>8}  {'reject':>7}  {'FPR/min':>9}")
    print(f"\n{header}\n  {'-'*48} {'-'*8}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*9}")

    plot_data_for_best = None

    for cfg in configs:
        tc = tw = ti = tm = trest = trtrig = 0
        first_run_data = None
        for i, test_run in enumerate(test_runs):
            times, raw, smooth, truth, m = simulate_one_run(
                test_run, clf,
                threshold=cfg['threshold'], strict_dwell=cfg['strict_dwell'])
            tc     += m['mi_correct']
            tw     += m['mi_wrong']
            ti     += m['mi_idle']
            tm     += m['mi_total']
            trest  += m['rest_total']
            trtrig += m['rest_trigger']
            if i == 0:
                first_run_data = (times, raw, smooth, truth)

        det_tpr   = (tc + tw) / tm
        precision = tc / (tc + tw) if (tc + tw) > 0 else 0.0
        cmd_acc   = tc / tm
        reject    = ti / tm
        rest_sec  = trest * S / FS
        fpr_min   = trtrig / (rest_sec / 60) if rest_sec > 0 else 0
        print(f"  {cfg['name']:<48} {det_tpr:>8.3f}  {precision:>9.3f}  "
              f"{cmd_acc:>8.3f}  {reject:>7.3f}  {fpr_min:>9.1f}")

        # Save plot for the headline combined config (Strategy 3 + Strategy 4)
        if cfg['threshold'] == IDLE_THRESHOLD and cfg['strict_dwell']:
            plot_data_for_best = first_run_data

    if plot_data_for_best is not None:
        out = plot_run_timeline(*plot_data_for_best, run_idx=0)
        print(f"\n  plot saved (S3+S4 combined config) → {out}")

    print(f"\n  → det_TPR × precision = cmd_acc (end-to-end correct-command rate).")
    print(f"    S3+S4 combined trades det_TPR (more abstention) for higher precision.")
    print(f"    For deployment safety, prioritize precision + low FPR/min over cmd_acc.")


if __name__ == "__main__":
    main()
