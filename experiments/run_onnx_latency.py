"""A5: EEGNet -> ONNX export + single-window inference latency.

Deployment demo for the #1 JD gap (端侧部署). Exports the benchmark's EEGNet
(2a config: 22 ch, 2 s window @ 250 Hz, 4 classes) to ONNX, verifies numerical
parity with the PyTorch model, then measures single-window (batch=1) inference
latency for both PyTorch and ONNX Runtime -- the realistic online-BCI regime
(one 2 s window classified at a time, cf. run_online_simulation.py).

Latency depends on architecture + input size, not weights, so a freshly
initialized EEGNet gives a representative number without a trained checkpoint.

Run:
  python -m experiments.run_onnx_latency
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
import onnxruntime as ort

from methods.eegnet import _EEGNet

# --- Config: 2a single-window deployment scenario (matches run_online_simulation) ---
N_CHANNELS, N_SAMPLES, N_CLASSES = 22, 500, 4   # 22 ch, 2 s @ 250 Hz, 4-class
FS = 250
RESULTS_DIR = Path("results")
ONNX_PATH = RESULTS_DIR / "eegnet_2a.onnx"
SEED = 42


def build_eegnet() -> _EEGNet:
    """Fresh EEGNet in eval mode (eval => BatchNorm uses running stats / Dropout off
    -- the deployment-correct mode, and required for batch=1 inference)."""
    torch.manual_seed(SEED)
    model = _EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
    model.eval()
    return model


def export_onnx(model: _EEGNet, dummy: torch.Tensor) -> None:
    """Export EEGNet to ONNX at a fixed single-window (batch=1) shape."""
    RESULTS_DIR.mkdir(exist_ok=True)
    # No opset_version pin: torch 2.11's exporter default avoids a Pad-op
    # downconversion that the version converter can't handle (harmless, but noisy).
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        input_names=["eeg"], output_names=["logits"],
    )


def check_parity(model: _EEGNet, sess: ort.InferenceSession,
                 dummy: torch.Tensor) -> float:
    """Max abs diff between PyTorch and ONNX Runtime outputs (expect ~1e-5).

    A large value means the export is broken -- always verify before trusting
    latency numbers from the exported graph.
    """
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    onnx_out = sess.run(None, {"eeg": dummy.numpy()})[0]
    return float(np.max(np.abs(torch_out - onnx_out)))


def measure_latency(run_once, n_warmup: int, n_timed: int) -> dict:
    """Measure single-call latency of `run_once` (a zero-arg callable doing ONE inference).

    Methodology:
      1. `n_warmup` UNTIMED calls first, to discard cold-start / lazy-init overhead.
      2. Time `n_timed` calls INDIVIDUALLY with time.perf_counter().
      3. Return statistics in MILLISECONDS: "mean", "p50", "p95", "p99".
         Tail percentiles matter for real-time BCI -- a good mean with a bad p99
         still blows the deadline occasionally.
    """
    durations = []
    for _ in range(n_warmup):
        run_once() 
    for _ in range(n_timed):
        start = time.perf_counter()
        run_once()
        end = time.perf_counter()
        duration_ms = (end - start) * 1000
        durations.append(duration_ms)
    mean = np.mean(durations)
    p50 = np.percentile(durations, 50)
    p95 = np.percentile(durations, 95)
    p99 = np.percentile(durations, 99)
    return {"mean": mean, "p50": p50, "p95": p95, "p99": p99}


def main() -> None:
    model = build_eegnet()
    dummy = torch.randn(1, 1, N_CHANNELS, N_SAMPLES)   # batch=1 single window

    export_onnx(model, dummy)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    parity = check_parity(model, sess, dummy)
    n_params = sum(p.numel() for p in model.parameters())
    onnx_kb = ONNX_PATH.stat().st_size / 1024

    # single-inference closures (batch=1)
    x_np = dummy.numpy()

    def run_torch():
        with torch.no_grad():
            model(dummy)

    def run_onnx():
        sess.run(None, {"eeg": x_np})

    torch_stats = measure_latency(run_torch, n_warmup=20, n_timed=1000)
    onnx_stats = measure_latency(run_onnx, n_warmup=20, n_timed=1000)

    print(f"=== A5: EEGNet single-window latency "
          f"({N_CHANNELS}ch x {N_SAMPLES} @ {FS}Hz, {N_CLASSES}-class) ===")
    print(f"params: {n_params:,}   ONNX size: {onnx_kb:.1f} KB   "
          f"parity (max|delta|): {parity:.2e}")
    print(f"{'backend':<10} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}  (ms)")
    for name, s in [("PyTorch", torch_stats), ("ONNX", onnx_stats)]:
        print(f"{name:<10} {s['mean']:>8.3f} {s['p50']:>8.3f} "
              f"{s['p95']:>8.3f} {s['p99']:>8.3f}")


if __name__ == "__main__":
    main()
