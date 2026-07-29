"""Sanity-test NAFNet inference: run on synthetic inputs, save
PNGs to the project root. Run from anywhere; the model is
located relative to this file's directory (the model is not
committed -- see ``install.bat`` and ``.gitignore``; download
via HuggingFace).
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "NAFNet-REDS-width64_v1.onnx"
sess = ort.InferenceSession(str(MODEL), providers=['CPUExecutionProvider'])
inp_name = sess.get_inputs()[0].name

# Try a photo-like synthetic image: smooth gradient with sharp edges
H = W = 256
img = np.zeros((1, 3, H, W), dtype=np.float32)
for y in range(H):
    for x in range(W):
        img[0, 0, y, x] = x / W          # R gradient
        img[0, 1, y, x] = y / H          # G gradient
        img[0, 2, y, x] = (x + y) / (W + H)  # B diagonal
img += 0.05  # lift
img = np.clip(img, 0, 1)
# Add a bit of mild noise
rng = np.random.default_rng(42)
img += rng.random(img.shape, dtype=np.float32) * 0.05
img = np.clip(img, 0, 1)

print('Input range:', img.min(), '..', img.max(), 'mean', img.mean())

# Try with 8x-aligned padding
def pad8(x):
    h, w = x.shape[-2:]
    nh = (h + 7) // 8 * 8
    nw = (w + 7) // 8 * 8
    if nh == h and nw == w:
        return x, (0, 0, 0, 0)
    pad = np.zeros((1, 3, nh, nw), dtype=np.float32)
    pad[..., :h, :w] = x
    return pad, (0, h, 0, w)  # top, bottom, left, right

padded, unpad = pad8(img)
print(f'Padded: {padded.shape}')
out = sess.run(None, {inp_name: padded})[0]
print(f'Output full: range [{out.min():.2f}, {out.max():.2f}] mean {out.mean():.2f}')
# Unpad
top, bot, left, right = unpad
out_crop = out[..., top:bot if bot else out.shape[-2], left:right if right else out.shape[-1]]
print(f'Output crop: range [{out_crop.min():.2f}, {out_crop.max():.2f}] mean {out_crop.mean():.2f}')

# Try with input preprocessed to [-1, 1] (NAFNet standard)
img_norm = (img * 2) - 1
padded_n, _ = pad8(img_norm)
out_n = sess.run(None, {inp_name: padded_n})[0]
out_n_crop = out_n[..., top:bot if bot else out_n.shape[-2], left:right if right else out_n.shape[-1]]
print(f'Input [-1,1], output range [{out_n_crop.min():.2f}, {out_n_crop.max():.2f}] mean {out_n_crop.mean():.2f}')

# Save outputs as PNG to visually inspect
out_png = np.clip(out_crop, 0, 1)[0].transpose(1, 2, 0)  # CHW -> HWC, RGB
out_png = (out_png * 255).astype(np.uint8)
Image.fromarray(out_png).save(str(REPO / "out1.png"))
print('saved out1.png (clipped [0, 1] from raw output)')

# Same for normalized input
out_n_png = np.clip((out_n_crop + 1) / 2, 0, 1)[0].transpose(1, 2, 0)
out_n_png = (out_n_png * 255).astype(np.uint8)
Image.fromarray(out_n_png).save(str(REPO / "out2.png"))
print('saved out2.png (from [-1, 1] input, shifted back)')
