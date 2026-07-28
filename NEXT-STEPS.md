# NAFNet Restoration Plug-in — Analysis & Next Steps

A new project folder for a GIMP 3.2+ plug-in that wraps
`NAFNet-REDS-width64_v1.onnx` (275 MB). The model is a 1:1 image
restoration network (denoising/deblurring, **not** super-resolution)
trained on the REDS (REalistic and Diverse Scenes) dataset.

## What we have

```
C:\Dev\GIMP_Native_Plugin\Gimp-restoration-plugin\
├── NAFNet-REDS-width64_v1.onnx   275 MB
└── tools/
    ├── inspect_onnx.py           dump model inputs/outputs/metadata
    └── test_inference.py         sanity test: run on a synthetic image
```

## Model I/O (verified)

| | Shape | Dtype | Range |
|---|---|---|---|
| Input | `(1, 3, H, W)` | float32 | `[0, 1]` |
| Output | `(1, 3, H, W)` | float32 | `[0, 1]` (pixel values) |

- Dynamic `H, W` (any size)
- 1:1 spatial (input dims == output dims → not a super-resolution model)
- 1446 nodes, opset 14, PyTorch 2.0.1 export
- ORT CPU inference: **9 s for 1024×1024** (single-threaded, no `with_parallel_execution`)
- 256×256 inference runs in <1 s

The model accepts any size; no padding to multiples-of-8 required (tested
without padding — works correctly). May need padding to multiples-of-8
for non-power-of-2 sizes, worth verifying with a 1023×1023 test.

## What the model does

NAFNet (Non-linear Activation Free Network) is a 2022 image
restoration architecture by Megvii. The REDS (REalistic and Diverse
Scenes) variant is trained on the REDS dataset, which is primarily
used for video **deblurring** and **super-resolution** tasks in the
NTIRE competitions.

Given that this model is 1:1 (not upsampling), it's most likely a
**deblurring** model — the input is a blurry image, the output is
the sharp image. Test with a blurry photo to confirm.

The model is **not** suitable for:
- Super-resolution (1:1, no upscaling)
- Inpainting (no mask input; would treat the unmasked region as
  valid image data)
- Style transfer

It **is** suitable for:
- Motion blur removal
- Defocus blur correction
- Light denoising (REDS-trained models handle this as a side effect)

To verify, run on a known-blurry image and compare against the
input. The output should look like the unblurred version.

---

## Architecture decision: sidecar vs native

The existing `Gimp-lama-inpainting` plug-in uses a sidecar
architecture (GIMP-side Python in GIMP's MINGW Python + worker
subprocess in system MSVC Python). The same architecture works
here and is the right call:

- The restoration filter is structurally simpler than LaMa (no
  selection, no mask, no bbox/pad/resize pipeline). The GIMP-side
  glue is ~100 lines instead of LaMa's 28 KB.
- The worker (Python or Rust) just does: load image → run
  inference → save image. No pre/post-processing complexity.
- Re-using the existing sidecar pattern means we inherit
  install.bat, the per-user `.interp` mapping, the model
  download-from-HuggingFace flow, and the test harness.

**Don't go native for this one** — the savings (no sidecar) are
~50 KB of Python on the GIMP side. The rewrite cost is weeks. Keep
the architecture consistent across plug-ins until both plug-ins
are stable.

When to consider going native: when there are 3+ plug-ins in
the suite and the sidecar overhead becomes a real cost. Then
rewrite the whole suite together. (See
`GOAL-Native-GIMP-Plugin-via-Zig.md` for that path.)

## Architecture decision: separate project vs add to LaMa repo

**Recommendation: separate project.** Reasons:

- The LaMa repo is a one-model project with `Lama` everywhere in
  the code. Adding NAFNet to it would mean either renaming it
  `Gimp-image-plugins` (loss of focus) or hiding NAFNet in a
  subfolder (confusing structure).
- The NAFNet user is a different audience: someone wanting
  deblurring isn't necessarily the same person wanting inpainting.
- Separate repos can ship independently (release v1.0 of
  restoration while LaMa is at v0.9).
- The plug-in can still share code via git submodules or a future
  shared library.

The directory `C:\Dev\GIMP_Native_Plugin\Gimp-restoration-plugin\`
exists with the ONNX and the inspection tools. Build out from here.

---

## Project layout (proposed)

```
C:\Dev\GIMP_Native_Plugin\Gimp-restoration-plugin\
├── NAFNet-REDS-width64_v1.onnx   the model (not committed; downloaded by install)
├── AGENTS.md                      project rules for AI agents
├── README.md                      user-facing install + usage
├── install.bat                    Windows per-user installer
├── uninstall.bat                  clean removal
├── Tools/                          dev/CI scripts
│   ├── inspect_onnx.py            dump model I/O
│   ├── test_inference.py          run on synthetic input, verify output range
│   └── make_sample_png.py         generate a noisy test image
├── tests/
│   ├── test_pipeline.py           end-to-end Python worker test
│   ├── test_nan_inf.py            detect numerical issues
│   └── outputs/                   (gitignored)
├── nafnet-restoration-py/         GIMP plug-in
│   ├── nafnet-inpaint.py          the GIMP-side plug-in (not the best name, see below)
│   ├── nafnet_worker.py           Python ORT sidecar worker
│   ├── nafnet_inpaint.py          GIMP-independent inference core
│   ├── nafnet_config.json         generated by install.bat
│   ├── nafnet-model.onnx          symlink/copy of the model
│   ├── install.bat                in-tree, called by repo install.bat
│   └── Docs/
│       └── NOTES.md               consolidated lessons
└── nafnet-worker-rs/              (optional) Rust sidecar for faster cold-start
    ├── Cargo.toml
    ├── src/main.rs
    └── ...
```

The plug-in file name: `nafnet-inpaint.py` is misleading since
NAFNet doesn't inpaint. Better names:
- `nafnet-deblur.py` (most accurate if we confirm it's a deblurring model)
- `nafnet-restore.py` (generic, accurate)
- `nafnet.py` (short, paired with `nafnet_worker.py`)

Recommendation: `nafnet-restore.py` until we confirm the specific
task; rename to `nafnet-deblur.py` once verified.

---

## Implementation plan

### Day 1: verify the model does what we think

Before writing any plug-in code, run the model on real test images
to confirm the restoration behavior. Use the existing `tools/`:

```bat
mkdir test-inputs
:: copy a known-blurry photo to test-inputs/blurry1.jpg
python tools\test_inference.py ^
    --input test-inputs\blurry1.jpg ^
    --output test-outputs\sharp1.png
```

Visually inspect the output. Does it look like the unblurred
version? Is the result different from the input? Save a comparison
PNG side-by-side.

If the output looks like the input (no change), the model might
be a feature extractor (intermediate representation), not a
restoration model. We'd need to reconsider.

If the output is sharper than the input, the model works. Proceed.

### Day 2: Python worker (sidecar)

`nafnet_worker.py` — a thin CLI that:
1. Loads `--image` (PNG)
2. Runs the model with ONNX Runtime CPU
3. Saves `--output` (PNG)

No selection/mask/bbox — just a straight pass-through. ~50 lines
of code. Reuses the ORT setup from the LaMa worker
(`lama_inpaint.py` in the LaMa repo).

```python
# pseudocode
import argparse, onnxruntime as ort, numpy as np, PIL.Image
sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
img = np.array(Image.open(args.image).convert('RGB'), dtype=np.float32) / 255.0
# (1, 3, H, W) layout
img_chw = img.transpose(2, 0, 1)[None]
out = sess.run(None, {sess.get_inputs()[0].name: img_chw})[0]
# (1, 3, H, W) -> (H, W, 3) for saving
out_hwc = (np.clip(out[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
Image.fromarray(out_hwc).save(args.output)
```

### Day 3: GIMP-side plug-in

`nafnet-restore.py` — the GIMP-side glue. Even simpler than
`lama-inpaint.py` because there's no selection/mask:

```python
# pseudocode
class NafnetRestore(Gimp.PlugIn):
    def do_query_procedures(self):
        return ["plug-in-nafnet-restore"]
    
    def do_create_procedure(self, name):
        # set up image procedure, menu, sensitivity
    
    def run(self, procedure, run_mode, image, drawables, config, run_data):
        # 1. validate drawables
        # 2. write drawable buffer to temp PNG
        # 3. spawn worker subprocess
        # 4. load result PNG into shadow buffer
        # 5. merge_shadow, update, flush
```

No selection processing. No bbox logic. No mask export. Just
"save the drawable, run the worker, load the result."

### Day 4: install + tests + README

- `install.bat` modeled on the LaMa install (one `install.bat` click
  → per-user interpreter mapping, model download from HuggingFace,
  plug-in files copied)
- `tests/test_pipeline.py` end-to-end test (Python worker roundtrip)
- `README.md` with: install, usage, model source, attribution
- `AGENTS.md` with the project rules

### Optional: Rust worker (if 9s 1024² is too slow)

The LaMa Rust worker pattern can be replicated for NAFNet. Static
linking ORT, parallel preprocessing, etc. Expected speedup: 2-3x
cold-start (similar to LaMa). For a restoration filter where
9s/inference is already at the edge of acceptable, this might
matter.

---

## Memory and tile considerations

For 1024×1024 input, NAFNet at width64 needs roughly:
- Input: 12 MB
- Output: 12 MB
- Intermediate activations (1446 nodes, transformer-style with
  global attention): peak ~1-2 GB

For 4K (3840×2160), the input/output alone is 99 MB each, and
intermediate activations could exceed 8 GB. Tiling is required.

**Tiling strategy** (when needed):

- Split image into 512×512 tiles with 32-pixel overlap
- Run NAFNet on each tile
- Blend in the overlap region (linear ramp or similar)
- Stitch back together

Tile size 512×512 is a good default — keeps inference per-tile at
<1s, total 4K = ~144 tiles = ~30-60s. Acceptable for a "render
this and go make coffee" filter.

For a first version, start with no tiling. Limit to images <2K
(3840×2160 → warn or refuse). Add tiling in a v2 once basic
behavior is verified.

---

## Concrete next step (the one to do first)

Verify the model does what we think. The output range is
[-0.05, 1.04] on a smooth-gradient test, which is promising, but
that's a synthetic test. We need to know:

1. Does it actually deblur (or whatever task) on real images?
2. Is the output quality good enough to ship?
3. Does the plug-in UX make sense (whole-image apply, or with
   selection = only restore the selected region)?

Suggested first action: pick a known-blurry image (search
"REDS validation" or use one of the REDS dataset samples) and
run `tools/test_inference.py` on it. Visually inspect the output.
Decide:
- If it works: proceed to Day 2 (Python worker)
- If the output is bad: figure out if the model needs different
  preprocessing (e.g., gamma correction, color space conversion)
- If the output is meaningless: this isn't a usable restoration
  model; reconsider the project

That's it. The model is verified to be loadable and runnable.
Whether it's useful depends on running it on real inputs.

---

## Open questions for the user

1. **What does REDS-width64 actually do?** Confirm by running
   on a known-blurry image. My best guess is deblurring, but it
   could be denoising or both.
2. **Should the plug-in restore the whole image, or only the
   selected region?** Default to whole-image (simpler), but a
   "selection = only the masked area" mode is also a valid UX
   (different from LaMa inpainting: no inpainting, just
   restoration of the selected pixels).
3. **Do you want the plug-in to ship separately, or as a second
   filter in the existing Gimp-lama-inpainting repo?** The
   analysis above recommends separate. Confirm.
4. **Rust worker for the restoration filter?** Adds 1-2 days of
   work. The LaMa Rust worker can serve as a template. Worth
   doing if the 9s/1024² CPU inference is too slow for the use
   case.
5. **Tiling for 4K images?** Defer to v2 unless someone hits the
   limit immediately.
