"""Smoke test for the deployed GIMP-side plug-in.

The GIMP-side plug-in does:
  1. Check if the drawable has alpha.
  2. If yes, extract the alpha to a side-channel PNG and pass to
     the worker via --alpha; the worker combines model RGB output
     with the original alpha.
  3. If no, skip the alpha round-trip; worker outputs plain RGB.

This test exercises both paths by spawning the Python worker
directly with the same args the GIMP-side would use, on a tiny
synthetic image. If both pass, the GIMP-side should not raise
"Invalid type" on RGB or RGBA inputs.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Cross-platform path to the deployed plug-in directory. On
# Windows, the per-user GIMP plug-in directory is under %APPDATA%.
# On other platforms, fall back to ~/.config (XDG-style default).
if os.name == "nt":
    _config_root = Path(os.environ["APPDATA"]) / "GIMP" / "3.2" / "plug-ins"
else:
    _config_root = Path.home() / ".config" / "GIMP" / "3.2" / "plug-ins"

INSTALL_DIR = _config_root / "nafnet-restore"
WORKER = INSTALL_DIR / "nafnet_worker.py"
MODEL = INSTALL_DIR / "nafnet-REDS-width64_v1.onnx"
cfg = json.loads((INSTALL_DIR / "nafnet_config.json").read_text())
WORKER_PY = Path(cfg["worker_python"])


def make_image(mode: str, size: int, path: Path) -> None:
    """Make a small synthetic image in the given PIL mode."""
    rng = np.random.default_rng(0)
    h = w = size
    if mode in ("RGB", "RGBA"):
        arr = np.zeros((h, w, 3 if mode == "RGB" else 4), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                arr[y, x, 0] = x
                arr[y, x, 1] = y
                arr[y, x, 2] = (x + y) // 2
        if mode == "RGBA":
            arr[..., 3] = 255
        arr = (arr * 0.5).astype(np.uint8)
    elif mode == "L":
        arr = np.tile(np.arange(h, dtype=np.uint8).reshape(-1, 1), (1, w)) // 2
    else:
        raise ValueError(f"unsupported mode: {mode}")
    Image.fromarray(arr, mode=mode).save(path)


def run_worker(in_path: Path, out_path: Path, alpha_path: Path | None) -> Path:
    """Spawn the Python worker with the same args the GIMP-side uses."""
    cmd = [
        str(WORKER_PY), str(WORKER),
        "--image", str(in_path),
        "--output", str(out_path),
        "--model", str(MODEL),
    ]
    if alpha_path is not None:
        cmd += ["--alpha", str(alpha_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise SystemExit(f"worker failed rc={result.returncode}: {result.stderr or result.stdout}")
    return out_path


def case(mode: str, has_alpha: bool) -> None:
    """Run one full case mimicking the GIMP-side plug-in's flow."""
    print(f"\n=== case: input mode={mode}, has_alpha={has_alpha} ===")
    with tempfile.TemporaryDirectory(prefix="nafnet-case-") as t:
        t = Path(t)
        in_p = t / "in.png"
        out_p = t / "out.png"
        alpha_p = t / "alpha.png" if has_alpha else None

        make_image(mode, 128, in_p)

        if has_alpha:
            # Mimic the GIMP-side alpha extraction
            img = Image.open(in_p)
            assert img.mode == mode
            alpha_arr = np.asarray(img.split()[-1])
            Image.fromarray(alpha_arr, mode="L").save(alpha_p)
            print(f"  saved alpha ({alpha_arr.shape}) to {alpha_p}")

        run_worker(in_p, out_p, alpha_p)

        out_img = Image.open(out_p)
        print(f"  output: {out_img.mode} {out_img.size}")
        if has_alpha:
            assert out_img.mode == "RGBA", f"expected RGBA, got {out_img.mode}"
            out_arr = np.asarray(out_img)
            print(f"  output range: [{out_arr.min()}, {out_arr.max()}]")
            # Verify the alpha was preserved byte-for-byte
            original_alpha = np.asarray(Image.open(in_p).split()[-1])
            np.testing.assert_array_equal(
                out_arr[..., 3], original_alpha,
                err_msg="alpha not preserved",
            )
            print("  alpha preserved byte-for-byte: OK")
        else:
            assert out_img.mode == "RGB", f"expected RGB, got {out_img.mode}"
            out_arr = np.asarray(out_img)
            print(f"  output range: [{out_arr.min()}, {out_arr.max()}]")
            print("  RGB-only output (no alpha to preserve): OK")


if __name__ == "__main__":
    if not WORKER.exists() or not WORKER_PY.exists() or not MODEL.exists():
        raise SystemExit("worker / model / python missing; run install.bat first")

    # The two paths the GIMP-side plug-in takes:
    #   - drawable has alpha  -> extract alpha, pass --alpha
    #   - drawable is RGB      -> skip alpha extraction
    case("RGB", has_alpha=False)
    case("RGBA", has_alpha=True)
    # Bonus: grayscale
    case("L", has_alpha=False)

    print("\nall cases passed")
