"""A5 (formal): EEGNet to ONNX deployment-readiness check.

Trains a real EEGNet on BCICIV-2a, exports it to ONNX, then on HELD-OUT real
windows (session E) reports two things:
  1. single-window inference latency, PyTorch vs ONNX Runtime (mean + p50/p95/p99)
  2. deployment parity: how faithfully the ONNX export reproduces the PyTorch model
     on real signals (see compute_parity).

Why this supersedes the dummy-input latency demo: latency was verified to be
weight/data-independent (so a random window is fine for timing), but numerical
parity is NOT -- it depends on input magnitude and model conditioning, so parity
must be measured on a TRAINED model + REAL windows (the actual deployment scenario).

Run:
  python -m experiments.run_onnx_deploy_check --subject 1 --epochs 100 --seeds 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from methods.eegnet import EEGNetClassifier
from data.bciciv2a import load_subject_session

REPO = Path(__file__).resolve().parents[1]
SCALE = 1e6   # matches EEGNetClassifier (V -> uV); the model is trained on X * SCALE
ONNX_PATH = REPO / "results" / "eegnet_2a_deploy.onnx"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)),
                   help="subjects to check (default all 9); each is a distinct trained model")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="fixed training seed (reproducible)")
    p.add_argument("--device", default=None, help="cpu / mps / cuda (default: auto)")
    return p.parse_args()


def pick_device(arg: str | None) -> str:
    if arg:
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def measure_latency(run_once, n_warmup: int = 20, n_timed: int = 1000) -> dict:
    """Single-call latency in ms: n_warmup untimed calls (discard cold-start), then
    time n_timed calls individually. Returns mean / p50 / p95 / p99 (tail matters
    for real-time BCI: a good mean with a bad p99 still blows the deadline)."""
    for _ in range(n_warmup):
        run_once()
    d = []
    for _ in range(n_timed):
        s = time.perf_counter()
        run_once()
        d.append((time.perf_counter() - s) * 1000)
    return {"mean": float(np.mean(d)), "p50": float(np.percentile(d, 50)),
            "p95": float(np.percentile(d, 95)), "p99": float(np.percentile(d, 99))}


def compute_parity(torch_logits: np.ndarray, onnx_logits: np.ndarray) -> dict:
    """TODO(human): define what "deployment parity" means for the exported model.

    Both inputs have shape (n_windows, n_classes): the PyTorch model's logits and the
    ONNX Runtime logits on the SAME real held-out windows.

    Return a dict that at least contains:
      - "max_abs_diff":   worst-case |torch - onnx| over all elements
                          (the numerical-safety number -- one bad window is what bites you)
      - "label_agreement": fraction of windows where argmax(torch) == argmax(onnx)
                          (the deployment-relevant parity: does the export make the SAME decision?)
    Add anything else you think an interviewer would want to see (e.g. "mean_abs_diff").
    """
    max_abs_diff = np.abs(torch_logits - onnx_logits).max()
    label_agreement = (torch_logits.argmax(axis=1) == onnx_logits.argmax(axis=1)).mean()
    return {"max_abs_diff": max_abs_diff, "label_agreement": label_agreement}


def torch_logits_over(model: torch.nn.Module, windows_scaled: np.ndarray) -> np.ndarray:
    """Run each already-scaled window through the PyTorch model at batch=1, return logits."""
    out = []
    with torch.no_grad():
        for w in windows_scaled:
            t = torch.from_numpy(w)[None, None]          # (1, 1, n_ch, n_samp)
            out.append(model(t).numpy()[0])
    return np.array(out)


def onnx_logits_over(sess: ort.InferenceSession, windows_scaled: np.ndarray) -> np.ndarray:
    """Same windows through ONNX Runtime at batch=1, return logits."""
    out = []
    for w in windows_scaled:
        x = w[None, None].astype(np.float32)
        out.append(sess.run(None, {"eeg": x})[0][0])
    return np.array(out)


def _fwd(model: torch.nn.Module, x: torch.Tensor) -> None:
    with torch.no_grad():
        model(x)


def run_one_seed(subject: int, epochs: int, device: str, seed: int) -> dict:
    Xtr, ytr = load_subject_session(subject=subject, session="T")
    Xte, _ = load_subject_session(subject=subject, session="E")    # held-out real windows

    clf = EEGNetClassifier(n_epochs=epochs, device=device, random_state=seed)
    clf.fit(Xtr, ytr)
    model = clf.model.to("cpu").eval()      # export + latency on CPU regardless of train device

    n_ch, n_samp = Xtr.shape[1], Xtr.shape[2]
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.randn(1, 1, n_ch, n_samp), str(ONNX_PATH),
                      input_names=["eeg"], output_names=["logits"])
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    # logits on ALL held-out real windows (scaled like the wrapper does), batch=1
    Xte_s = (Xte * SCALE).astype(np.float32)
    parity = compute_parity(torch_logits_over(model, Xte_s), onnx_logits_over(sess, Xte_s))

    # latency on one representative real window (verified weight/data-independent)
    win = torch.from_numpy(Xte_s[0])[None, None]
    win_np = win.numpy()
    latency = {
        "PyTorch": measure_latency(lambda: _fwd(model, win)),
        "ONNX": measure_latency(lambda: sess.run(None, {"eeg": win_np})),
    }
    return {"parity": parity, "latency": latency,
            "n_params": sum(p.numel() for p in model.parameters()),
            "onnx_kb": ONNX_PATH.stat().st_size / 1024, "n_windows": len(Xte_s)}


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"ONNX deploy check | 2a subjects {args.subjects} | {args.epochs} epochs | "
          f"seed={args.seed} | train device={device} | export+latency on CPU", flush=True)

    runs = []
    for s in args.subjects:
        r = run_one_seed(s, args.epochs, device, args.seed)
        runs.append(r)
        print(f"  S{s}: max|delta|={r['parity']['max_abs_diff']:.2e}  "
              f"label agreement={r['parity']['label_agreement'] * 100:.2f}%  "
              f"(n={r['n_windows']})", flush=True)
    r0 = runs[0]

    print(f"\nparams: {r0['n_params']:,}   ONNX size: {r0['onnx_kb']:.1f} KB   "
          f"windows/subject: {r0['n_windows']}   subjects: {len(runs)}")
    print("\nsingle-window latency (ms, CPU, batch=1; weight/data-independent):")
    print(f"  {'backend':<8} {'mean':>7} {'p50':>7} {'p95':>7} {'p99':>7}")
    for name, st in r0["latency"].items():
        print(f"  {name:<8} {st['mean']:>7.3f} {st['p50']:>7.3f} {st['p95']:>7.3f} {st['p99']:>7.3f}")
    print(f"  speedup (PyTorch/ONNX mean): "
          f"{r0['latency']['PyTorch']['mean'] / r0['latency']['ONNX']['mean']:.2f}x")

    print(f"\ndeployment parity (worst-case across {len(runs)} subjects x real held-out windows):")
    print(f"  worst-case max|delta logit|: {max(r['parity']['max_abs_diff'] for r in runs):.2e}")
    print(f"  worst-case label agreement : {min(r['parity']['label_agreement'] for r in runs) * 100:.2f}%")

    # Persist a summary so the results notebook can read these numbers instead of
    # hand-typing them (keeps the "every number comes from a script-generated file"
    # invariant that the rest of results/ already follows).
    summary = {
        "n_subjects": len(runs),
        "n_windows_per_subject": r0["n_windows"],
        "n_windows_total": int(sum(r["n_windows"] for r in runs)),
        "n_params": r0["n_params"],
        "onnx_kb": r0["onnx_kb"],
        "latency_ms": {name: st for name, st in r0["latency"].items()},
        "speedup": r0["latency"]["PyTorch"]["mean"] / r0["latency"]["ONNX"]["mean"],
        "worst_max_abs_diff": float(max(r["parity"]["max_abs_diff"] for r in runs)),
        "worst_label_agreement": float(min(r["parity"]["label_agreement"] for r in runs)),
        "epochs": args.epochs, "seed": args.seed, "train_device": device,
    }
    out = REPO / "results" / "onnx_deploy_check.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary -> {out}")


if __name__ == "__main__":
    main()
