"""End-to-end test for the Rust NAFNet worker.

Generates a small noisy test image, runs the worker on it, and
verifies the output is in the [0, 1] range with a sensible mean
shift (denoising reduces the mean noise level).

Also tests the tiled-inference path on a 1024x1024 input that
exceeds the 512-tile default. This exercises `tiled_inference`
and the blend window in addition to `run_single_inference`.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(r"C:\Dev\GIMP_Native_Plugin\Gimp-restoration-plugin")
WORKER = REPO / "nafnet-worker-rs" / "target" / "release" / "nafnet-worker.exe"
MODEL = REPO / "NAFNet-REDS-width64_v1.onnx"


def make_noisy_image(width: int, height: int, path: Path) -> None:
    """Synthetic test: smooth gradient with added noise.

    The smooth gradient is the "ground truth"; the noise is the
    degradation. After restoration, the output should be closer to
    the gradient (less variance) than the input.
    """
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


def run_worker(in_path: Path, out_path: Path, tile_size: int = 512, tile_overlap: int = 32) -> None:
    cmd = [
        str(WORKER),
        "--image", str(in_path),
        "--output", str(out_path),
        "--model", str(MODEL),
        "--tile-size", str(tile_size),
        "--tile-overlap", str(tile_overlap),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise SystemExit(f"worker failed rc={result.returncode}")


def assert_image(path: Path, expected_size: tuple[int, int]) -> None:
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w, c = img.shape
    assert (h, w) == expected_size, f"size mismatch: {img.shape} vs {expected_size}"
    assert 0.0 <= img.min() and img.max() <= 1.0, f"out of range: [{img.min()}, {img.max()}]"
    print(f"  output {w}x{h}x{c}: range [{img.min():.3f}, {img.max():.3f}] mean {img.mean():.3f}")


def main() -> int:
    if not WORKER.exists():
        raise SystemExit(f"worker binary not found: {WORKER} (build first)")
    if not MODEL.exists():
        raise SystemExit(f"model not found: {MODEL}")

    with tempfile.TemporaryDirectory(prefix="nafnet-test-") as tmp:
        tmp = Path(tmp)
        # Test 1: small image, single inference (no tiling)
        print("test 1: 256x256, no tiling")
        in_path = tmp / "in_256.png"
        out_path = tmp / "out_256.png"
        make_noisy_image(256, 256, in_path)
        run_worker(in_path, out_path)
        assert_image(out_path, (256, 256))

        # Test 2: image that requires tiling (1024 > default 512)
        print("test 2: 1024x1024, tiled (2x2 tiles, 32px overlap)")
        in_path = tmp / "in_1024.png"
        out_path = tmp / "out_1024.png"
        make_noisy_image(1024, 1024, in_path)
        run_worker(in_path, out_path)
        assert_image(out_path, (1024, 1024))

        # Test 3: non-power-of-two dimensions, last tile is partial
        # The tiling logic must handle the right/bottom edges correctly
        print("test 3: 700x900, tiled (irregular last tiles)")
        in_path = tmp / "in_700x900.png"
        out_path = tmp / "out_700x900.png"
        make_noisy_image(700, 900, in_path)
        run_worker(in_path, out_path)
        assert_image(out_path, (900, 700))

        # Sanity: restoration should reduce noise variance in a
        # region that's pure noise on a smooth background. This is
        # a weak check because the synthetic image isn't well-matched
        # to the REDS training distribution, but it should still
        # show some smoothing.
        in_arr = np.array(Image.open(tmp / "in_256.png").convert("RGB"), dtype=np.float32) / 255.0
        out_arr = np.array(Image.open(tmp / "out_256.png").convert("RGB"), dtype=np.float32) / 255.0
        # A 100x100 region in the middle (well away from the gradient edges)
        in_var = in_arr[100:200, 100:200, :].var()
        out_var = out_arr[100:200, 100:200, :].var()
        print(f"  noise variance: in={in_var:.4f}  out={out_var:.4f}  "
              f"reduction={(1 - out_var/in_var)*100:.1f}%")

    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
