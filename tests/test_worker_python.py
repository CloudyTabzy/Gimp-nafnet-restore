"""Test the Python NAFNet worker end-to-end.

Mirrors the Rust worker test (test_worker_rust.py) but exercises
the Python fallback path. Both should produce equivalent output
for the same input (modulo floating-point ordering).
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(r"C:\Dev\GIMP_Native_Plugin\Gimp-restoration-plugin")
WORKER = REPO / "nafnet-restoration-py" / "nafnet_worker.py"
MODEL = REPO / "NAFNet-REDS-width64_v1.onnx"


def make_noisy_image(width: int, height: int, path: Path) -> None:
    rng = np.random.default_rng(42)
    base = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            base[y, x, 0] = x / width
            base[y, x, 1] = y / height
            base[y, x, 2] = (x + y) / (2 * (width + height))
    img = base + rng.random(base.shape, dtype=np.float32) * 0.30
    img = np.clip(img, 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def run_worker(in_path: Path, out_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(WORKER), "--image", str(in_path),
         "--output", str(out_path), "--model", str(MODEL)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise SystemExit(f"python worker failed rc={result.returncode}")
    # The Python worker should also emit LAMA_MARKER lines.
    for line in result.stdout.splitlines():
        if "LAMA_MARKER" in line or "error" in line.lower():
            print(f"  {line.strip()}")


def assert_image(path: Path, expected_size: tuple[int, int]) -> None:
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w, c = img.shape
    assert (h, w) == expected_size, f"size mismatch: {img.shape} vs {expected_size}"
    assert 0.0 <= img.min() and img.max() <= 1.0, f"out of range: [{img.min()}, {img.max()}]"
    print(f"  output {w}x{h}x{c}: range [{img.min():.3f}, {img.max():.3f}] mean {img.mean():.3f}")


def main() -> int:
    if not WORKER.exists():
        raise SystemExit(f"worker script not found: {WORKER}")
    if not MODEL.exists():
        raise SystemExit(f"model not found: {MODEL}")

    with tempfile.TemporaryDirectory(prefix="nafnet-py-test-") as tmp:
        tmp = Path(tmp)
        print("test: 256x256, python worker (no tiling, fallback path)")
        in_path = tmp / "in.png"
        out_path = tmp / "out.png"
        make_noisy_image(256, 256, in_path)
        run_worker(in_path, out_path)
        assert_image(out_path, (256, 256))
    print("\nall python worker tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
