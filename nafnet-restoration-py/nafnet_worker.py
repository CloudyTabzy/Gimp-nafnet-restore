"""NAFNet image restoration sidecar worker (Python).

CLI mirrors the Rust `nafnet-worker` one-for-one so the GIMP plug-in
can swap between Rust and Python transparently. Same input/output
contract: load PNG -> run NAFNet on the RGB channels -> save PNG.
No tiling in this worker; the Rust worker handles that for
high-resolution images. The Python worker is the fallback when
the Rust binary is not installed (per `lama_config.json`-style
detection, mirrored here as `nafnet_config.json`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def _load_session(model_path: Path) -> tuple[ort.InferenceSession, str]:
    """Build the ORT inference session.

    The provider is whichever ORT can find first; we don't try to
    force CUDA/DirectML here. For a one-shot image restoration on
    CPU, the default is fine and the simplest install path.
    """
    providers = ort.get_available_providers()
    primary = "CPUExecutionProvider" if "CPUExecutionProvider" in providers else providers[0]
    sess = ort.InferenceSession(
        str(model_path),
        providers=[primary],
    )
    return sess, primary


def _run_nafnet(sess: ort.InferenceSession, rgb_chw: np.ndarray) -> np.ndarray:
    """Run NAFNet on a (1, 3, H, W) f32 input, return (1, 3, H, W) f32.

    Output values are clipped to [0, 1] — NAFNet's output range is
    [0, 1] for well-formed input but the network can produce values
    slightly outside this range on edge cases.
    """
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: rgb_chw})
    out = outputs[0]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def restore(image_path: Path, model_path: Path, output_path: Path) -> None:
    """Load -> run -> save. Pure RGB; alpha is dropped (NAFNet is 3ch)."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    rgb_chw = arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)

    t0 = time.monotonic()
    sess, provider = _load_session(model_path)
    t_load = time.monotonic() - t0
    print(f"[LAMA_MARKER] phase session_built provider={provider} ({t_load * 1000:.0f}ms)")

    print(f"[LAMA_MARKER] phase loaded {arr.shape[1]}x{arr.shape[0]}")
    t0 = time.monotonic()
    out_chw = _run_nafnet(sess, rgb_chw)
    t_infer = time.monotonic() - t0
    print(f"[LAMA_MARKER] phase inference_done ({t_infer * 1000:.0f}ms)")

    out_hwc = (out_chw[0].transpose(1, 2, 0) * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(out_hwc, mode="RGB").save(output_path)
    print(f"[LAMA_MARKER] phase saved")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True, type=Path, help="Input PNG path (RGB or RGBA; alpha is dropped).")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path. Same spatial size as input.")
    parser.add_argument("--model", required=True, type=Path, help="Path to the NAFNet ONNX model.")
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"error: input image not found: {args.image}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        restore(args.image, args.model, args.output)
    except Exception as exc:  # noqa: BLE001 - the GIMP parent reads stderr on failure
        print(f"nafnet_worker: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
