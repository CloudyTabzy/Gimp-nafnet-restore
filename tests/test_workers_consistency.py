"""Cross-worker consistency test.

Runs the same input through both the Rust and Python workers and
compares the outputs. They should be near-identical: the model is
the same ONNX, the input is the same PNG, the only difference is
floating-point ordering of identical arithmetic. We allow a small
absolute tolerance for these.

This is a contract test: it catches cases where the GIMP-side
plug-in could legitimately be confused (e.g., one worker is clipping
output to [0, 1] but the other isn't).
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
RUST_WORKER = REPO / "nafnet-worker-rs" / "target" / "release" / "nafnet-worker.exe"
PYTHON_WORKER = REPO / "nafnet-restoration-py" / "nafnet_worker.py"
MODEL = REPO / "NAFNet-REDS-width64_v1.onnx"


def make_input(width: int, height: int, path: Path) -> None:
    """Smooth gradient + a sharp test pattern.

    The sharp pattern is the discriminator: if the two workers
    restore it differently, the difference will show up as a large
    error at the pattern edges. The smooth gradient is the
    baseline; any per-row / per-column reordering in the worker
    would also surface there.
    """
    rng = np.random.default_rng(42)
    base = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            base[y, x, 0] = x / width
            base[y, x, 1] = y / height
            base[y, x, 2] = (x + y) / (2 * (width + height))
    img = base + rng.random(base.shape, dtype=np.float32) * 0.20
    # Add a sharp red square that both workers must preserve
    img[40:80, 40:80, 0] = 0.9
    img[40:80, 40:80, 1] = 0.05
    img[40:80, 40:80, 2] = 0.05
    img = np.clip(img, 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def run(rust: bool, in_path: Path, out_path: Path) -> np.ndarray:
    if rust:
        cmd = [str(RUST_WORKER), "--image", str(in_path),
               "--output", str(out_path), "--model", str(MODEL)]
    else:
        cmd = [sys.executable, str(PYTHON_WORKER), "--image", str(in_path),
               "--output", str(out_path), "--model", str(MODEL)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise SystemExit(f"worker {'rust' if rust else 'python'} failed rc={result.returncode}")
    return np.array(Image.open(out_path).convert("RGB"), dtype=np.float32) / 255.0


def main() -> int:
    if not RUST_WORKER.exists():
        print("rust worker binary not built; run `cargo build --release` in nafnet-worker-rs/")
        return 0
    if not MODEL.exists():
        raise SystemExit(f"model not found: {MODEL}")

    with tempfile.TemporaryDirectory(prefix="nafnet-cross-") as tmp:
        tmp = Path(tmp)
        in_path = tmp / "in.png"
        make_input(256, 256, in_path)

        print("running rust worker...")
        out_rust = run(True, in_path, tmp / "out_rust.png")
        print(f"  range [{out_rust.min():.3f}, {out_rust.max():.3f}] mean {out_rust.mean():.3f}")

        print("running python worker...")
        out_py = run(False, in_path, tmp / "out_py.png")
        print(f"  range [{out_py.min():.3f}, {out_py.max():.3f}] mean {out_py.mean():.3f}")

        diff = np.abs(out_rust - out_py)
        max_diff = diff.max()
        mean_diff = diff.mean()
        n_different = int((diff > 0.01).sum())  # >1% difference
        n_total = diff.size

        print(f"\nmax abs diff:  {max_diff:.5f}")
        print(f"mean abs diff: {mean_diff:.5f}")
        print(f"pixels >1% diff: {n_different} / {n_total} ({100.0 * n_different / n_total:.3f}%)")

        if max_diff > 0.05:
            print("FAIL: workers diverge by more than 5% on a single pixel")
            return 1
        if mean_diff > 0.005:
            print("FAIL: mean abs diff > 0.005")
            return 1
        print("PASS: workers are consistent within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
