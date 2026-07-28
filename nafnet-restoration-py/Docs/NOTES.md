# NAFNet Restoration Plug-in - Lessons & Future Directions

> Consolidated from the development session. Keeps what is durable
> and useful; drops the implementation plan, the chronological
> attempts log, and anything superseded by the final `.interp`
> shebang solution. Mirrors the structure of the sibling LaMa
> plug-in's `Docs/NOTES.md`.

**Date:** 2026-07
**Status:** Reference notes; not a build plan

---

## 1. The wall: MINGW Clang vs MSVC wheels

GIMP 3.2 ships a MINGW Clang-built Python 3.14 with `gi` and GEGL,
but **every** `cp314` wheel on PyPI (`numpy`, `onnxruntime`,
`opencv-python`, `imath`) is MSVC-built. The two ABIs do not
match, and `numpy`'s import code refuses to load MSVC `.pyd` files
in a MINGW Python with a hard error - not a warning.

Three ways past it:

1. **MSYS2 + build MINGW numpy/onnxruntime from source** - 1+ GB
   toolchain, hours of build time. Out of scope.
2. **Portable MINGW Python + MINGW-built wheel bundle** - same
   problem at a different scale.
3. **Sidecar architecture** - keep the GIMP side pure-stdlib + `gi`
   + GEGL; do the ML work in a separate configured system Python
   process where MSVC wheels load normally. **Chosen.**

The sidecar is the only realistic path. The whole GIMP side stays
clean (`import gi`, `Gimp`, `GimpUi`, `Gegl`, `GLib` only); the
worker process gets `numpy`, `onnxruntime`, `Pillow`. They talk via
temp PNGs.

---

## 2. The sidecar pattern

```
GIMP process  (MINGW Python 3.14, gi + GEGL only)
  ╰─ #!nafnet-gimp-python  (shebang ╰ user-level .interp mapping)
       ╰─ nafnet-restore.py
            ├─ exports drawable (+ selection ROI for region mode) to temp PNGs via GEGL
            ├─ spawns worker subprocess (Rust or Python)
            ╰─ loads result PNG into drawable's shadow buffer,
               merge_shadow(True)
                    │
                    ▼
           worker process  (Rust or Python, with Pillow + onnxruntime)
               onnxruntime / ort CPU provider
               ↳ model pipeline:
                  load PNG ╰ HWC f32 [0, 1] ╰ (Rust: tile + blend
                  overlap) ╰ NAFNet ╰ HWC f32 ╰ save PNG
```

**Invariants that must hold for every future filter:**

- The GIMP-side plug-in imports **only** `gi`, `Gimp`, `GimpUi`,
  `Gegl`, `GLib`. No `numpy`, no `PIL`, no `onnxruntime`, no `cv2`. This
  is the entire point of the architecture.
- Worker communication is file-based (temp PNGs). No shared
  Python interpreter, no shared C ABI, no sockets.
- One inference per call. The worker writes a single output PNG;
  the GIMP side pastes it into the drawable's shadow buffer.
- No GIMP installation is modified. Per-user interpreter alias
  (`.interp` files in `%APPDATA%\GIMP\3.2\interpreters\`) + shebang on
  the plug-in's first line is the canonical handoff, not patching
  `pygimp.interp`, not modifying `PATH`, not re-exec'ing across CRTs.

---

## 3. Getting GIMP to actually use its own Python

On stock Windows GIMP 3.2, third-party `.py` plug-ins can initially
be launched by the first system `python.exe`/`pythonw.exe` on
`PATH` (typically system Python 3.13, which has no `gi`). We saw this
break empirically before we fixed it.

The fix is **not** to patch `<GIMP install>\lib\gimp\3.0\interpreters\
pygimp.interp` - that's a system file and modifying it breaks the
next GIMP update. The fix is a per-user interpreter alias:

1. The plug-in's first line is `#!nafnet-gimp-python` (a project-specific
   alias, never `python` or `python3`).
2. The installer writes two `.interp` files into
   `%APPDATA%\GIMP\3.2\interpreters\`:
   - `nafnet-gimp-python.interp` → maps to `bin\python.exe` (console)
   - `nafnet-gimp-python_win.interp` → maps to `bin\pythonw.exe` (GUI)
3. GIMP's shebang resolution runs before `.py` extension resolution
   (`gimpinterpreterdb.c:887-899`), so this mapping is scoped to
   this single plug-in. Every other `.py` plug-in still uses the
   system Python.
4. Use **forward slashes** in `.interp` file paths. GIMP's parser
   requires them on Windows.
5. Removal is two file deletions. No GIMP install touched.

**Important:** the alias must be a unique name, not `python` or
`python3`. Using a common name would globally override `.py`
resolution for all plug-ins.

---

## 4. Image-scoped procedure gotcha (don't waste a day on this)

`Gimp.ImageProcedure` with `<Image>/Filters/...` menu path, an
image-type filter like `"RGB*, GRAY*"`, and `DRAWABLE` sensitivity
mask is **image-scoped**. GIMP is allowed - and does, by default -
to **omit the menu entry when no image/drawable is open**.

Symptoms when you forget: the plug-in is registered (visible in
`gimp --verbose` as `Querying plug-in: ...`), the PDB shows the
right `MENU_PATHS` and `SENSITIVITY`, but the filter is missing from
the menu.

Test rule: always test menu presence **with an image loaded**.
Headless `gimp --verbose` with no image will not show the entry -
this is correct GIMP behavior, not a bug.

Other things that hide an image-scoped menu entry:
- No image open
- No drawable/layer active
- The image type is not in `set_image_types(...)` (e.g. indexed color)

---

## 5. `subprocess.Popen` pipe leak locks the worker EXE on Windows

**Symptom:** after running the plug-in once, you cannot replace
`nafnet-worker_rust.exe` (or `nafnet_worker.py`) in the plug-in folder
with "the folder or a file in it is open in another program". Task
Manager shows the EXE still running with non-zero CPU even after GIMP
is closed. Stays locked until PC restart.

**Cause:** the inference function used `subprocess.Popen` with
`stdout=PIPE, stderr=PIPE` and read lines in a loop. When the
inference done marker was detected, an early `return` exited the
loop without closing the pipes or waiting for the child. Windows
holds a file lock on the EXE as long as any handle is open, and
those pipe handles are owned by the GIMP process (which may run
for hours).

**Fix:** use `try/finally` to unconditionally close all handles and
`process.wait()` to reap the child:

```python
try:
    for line in iter(process.stdout.readline, ""):
        if line.startswith("RESULT:"):
            result_data = json.loads(line[7:])
            break  # break, not return
finally:
    process.stdout.close()
    process.stderr.close()
    process.wait()  # releases the file lock immediately
```

If you don't need streaming, just use `subprocess.run()` - it
always waits and closes. Only use `Popen` when you actually need
incremental output.

---

## 6. Execution provider cascade (what to try, in order)

For a CPU-only Rust worker:

| Provider | Tries? | Notes |
|---|---|---|
| `CPUExecutionProvider` | Yes, default | Always works. ~9 s inference for 1024². |
| `CUDAExecutionProvider` | Optional | Needs matching CUDA 12 + cuDNN 9 on PATH. NVIDIA only. |
| `DmlExecutionProvider` | Known issues | ORT prebuilt DirectML binary can crash at session init with `AbiCustomRegistry.cpp(519)`, `E_INVALIDARG` in some environments due to a prebuilt-binary incompatibility with certain D3D12 runtime versions. |
| `WebGPU / Dawn` | Experimental | ORT upstream marks it experimental; can't be combined with other GPU EPs in the prebuilt binary. |

**Three things that hurt ORT CPU inference specifically:**

1. **`with_intra_threads(N)`** — counterintuitively *slows down*
   inference (1.75 s → 4 s). ORT's CPU EP is already internally
   parallelized. Forcing more threads creates pool overhead and
   contention. **Don't override ORT's default thread count unless
   you've measured a specific win.**
2. **`with_memory_pattern(false)`** — required for DirectML,
   harmless for CPU. Keep it.
3. **`with_parallel_execution(false)`** — required for DirectML,
   harmless for CPU. Keep it.

**What actually helps for ORT CPU on this model:** (this model = NAFNet)

- `with_dimension_override("batch", 1)` - required because the
  NAFNet graph has a dynamic batch dim. The attention blocks'
  global pooling produces batch-dependent 4D tensors that DirectML
  can't pre-compile kernels for. Fixing batch=1 is always correct
  (one image per call).
- `with_optimization_level(GraphOptimizationLevel::Level1)` -
  basic constant folding. Don't set to All - it adds compile time
  without measurable runtime benefit on a 275 MB model already at
  the inference-time ceiling.
- Pre-computed resize weights hoisted out of the channel loop.
- `rayon` parallelism for per-row preprocessing (HWC → CHW) and
  postprocessing (CHW → HWC). Marginal gain for single-tile
  inference but a real win on 4K images with 16+ tiles.

---

## 7. NAFNet-specific: 1:1 spatial and the tiled-inference design

NAFNet-REDS is **1:1 spatial**: input H,W == output H,W. Unlike
LaMa which crops + resizes to 512² + inverts, NAFNet processes the
full input at its native resolution. This is a huge simplification
of the GIMP-side glue (no bbox/pad/resize/paste pipeline) but a
huge complication at the worker for high-resolution images.

**Tiling design (Rust worker):**

- Tile size 512 px, overlap 32 px. The dghs-imgutils defaults.
- 2D tent window for blending in the overlap region. Top and
  bottom ramps are clamped to `min(overlap, tile_h/2)` so the
  center stays at weight 1; left and right ramps similarly. When
  tiles overlap, each pixel's effective weight is the sum of
  ramps from each covering tile, and the result is divided by
  that sum so the contribution is uniform across the seam.
- Stride = `tile_size - overlap` = 480 px. For a 1024×1024 image
  this is 2×2 tiles (one tile is the "corner", three others
  share overlap with it).
- Right/bottom edges handle partial tiles correctly: the tile
  is cropped to the actual image boundary before inference, and
  the blend window is trimmed to match. No "padding" tile
  size; the inference call's input is always `(1, 3, h, w)` for
  whatever h, w the partial tile happens to be.

**Why tile size 512?** It matches the dghs-imgutils default. Larger
tiles (768, 1024) give marginally better quality (the model sees
more spatial context) at much higher memory cost. 512 is a
reasonable default for a 4 GB VRAM RTX 4050. The tile size is
exposed as `--tile-size` so users with more memory can override
at the command line.

---

## 8. Cross-CRT re-exec - DON'T do this

We tried a stdlib-only re-exec bootstrap that detected when the
plug-in was launched with the wrong Python and relaunched under
GIMP's Python. It "worked" in the sense that `import gi` succeeded
in the child. But:

1. **The re-exec loses GIMP's wire-protocol descriptor table.** On
   Windows, FD numbers are owned by the CRT instance. MSVC → MINGW
   re-exec means the child gets an FD that points to a different
   descriptor table, and the wire protocol immediately returns EOF.
2. **Importing `gi` in the child is not proof the wire protocol
   survived.** The wire protocol is the problem, not the import.
3. **The reliable symptom is `LibGimpBase-WARNING: gimp_wire_read():
   unexpected EOF` with no Python traceback.** End-to-end test
   (real `gimp --verbose`) is the only honest check.

Don't try to re-exec across CRTs. Use the per-user `.interp` alias
instead - it's the supported way.

---

## 9. The seven big mistakes we made (don't repeat)

1. **Patching `pygimp.interp`** to force GIMP to use its bundled
   Python. The user pushed back on this and they were right - there
   is a supported way (`*.interp` files in the user interpreter
   dir) that doesn't touch GIMP's install.
2. **Bundling wheels in `vendor/`** with `sys.path` prepending. The
   C-ABI mismatch is at the binary level; no amount of file
   shuffling makes MSVC `.pyd` files loadable in a MINGW Python.
3. **Trusting `pip install` to fix GIMP-internal dependencies**
   (e.g. PyGObject). GIMP's runtime has no dev headers; the
   supported path is to use GIMP's bundled Python, not to rebuild
   it.
4. **Re-exec'ing across CRTs** to "get into" GIMP's Python. The
   wire-protocol descriptor table does not survive.
5. **Treating "import gi succeeded" as proof the bootstrap
   worked.** It isn't - the wire protocol is the real test, and
   the symptom is `gimp_wire_read(): unexpected EOF` with no
   Python traceback.
6. **Assuming the model is "fast enough on CPU"** without measuring.
   NAFNet at 1024² is 9 s on CPU; 4K images with 16+ tiles will
   be 30-60 s. Tell the user, don't hide the latency.
7. **Trusting `cargo install` to silently work** in install scripts.
   The Rust build can take minutes; install scripts should not
   silently fail and leave the user without a working plug-in.
   The installer should print clear status (built / already
   present / source not found / cargo missing / build failed)
   and never block on a Rust build.

---

## 10. Where to go from here

### Short term (low risk, high value)

- **Alpha-channel preservation in region mode.** v1 simply drops
  the alpha channel inside the bbox; the context ring is
  restored as a side effect. v2 should copy the original drawable's
  alpha bytes back into the bbox after applying the model output.
  ~10 lines in the GIMP-side glue.
- **Test the menu visibility properly** - add a CI step that
  launches GIMP headless with a test image and asserts the two
  menu entries resolve. The image-scoped procedure gotcha is real
  and easy to miss in manual testing.

### Medium term (moderate risk, big upside)

- **Persistent worker** - instead of spawning a new worker per
  call (with model reload each time), keep one worker running and
  communicate via stdin/stdout JSON. The model loading (~3-5 s)
  disappears from the per-call budget. Uses the same temp PNG
  handoff or upgrades to length-prefixed binary IPC.
- **Tile-batched inference** - currently the Rust worker processes
  tiles serially. The NAFNet graph supports dynamic batch dim, so
  we could batch 4-8 tiles per inference call and use the model
  more efficiently. ~20% expected speedup, ~100 lines of changes.
- **Tiled Python worker** - the Python worker is a fallback; it
  currently has no tiling (only handles single-tile inputs).
  Port the tile-and-blend loop from the Rust worker (~50 lines)
  to Python if 4K inputs become a real use case in Python mode.

### Long term (architectural changes, only if needed)

- **Native GIMP plug-in via Zig** - rewrite the plug-in in C/C++
  compiled with Zig targeting MINGW, statically link ORT CPU, no
  Python anywhere. See `C:\Dev\GIMP_Native_Plugin\GOAL-Native-GIMP-Plugin-via-Zig.md`
  for the feasibility analysis. Cost: real C build pipeline, more
  brittle deploy, lost cross-platform Python tooling. Only worth
  doing if the sidecar architecture proves fragile on a real
  GIMP 3.4 / GIMP 4 transition.
- **GPU plug-in** - if the user can ever get DirectML or CUDA
  working on their hardware, the same sidecar with
  `--features directml` or `--features cuda` already exists in the
  Rust worker and just needs to be enabled.
- **Multiple-model plug-in** - register one entry per
  Gimp.Procedure (e.g. "Restore Image", "Restore Selection",
  "Restore Photo"), each dispatching to a different ONNX model.
  Same architecture; just more plug-in registrations.

---

## 11. Quick reference - the right shape for a new filter

If you're adding a new ONNX-backed filter to this project (or a
similar one), the shape is:

1. `filter-foo-py/foo-foo.py` - GIMP-side plug-in, shebang on
   first line, only `gi`/`Gimp`/`GimpUi`/`Gegl`/`GLib` imports.
2. `filter-foo-py/foo_worker.py` - system-Python CLI sidecar,
   `numpy` + `onnxruntime` + `PIL`, no GIMP, no `gi`. Takes
   PNGs in, returns PNGs out.
3. `filter-foo-py/install.bat` - detects GIMP Python, writes
   `.interp` files into the user interpreter dir, copies plug-in
   files to `%APPDATA%\GIMP\3.2\plug-ins\foo\`. Never touches
   `<GIMP install>`.
4. `tests/test_pipeline.py` - covers the core pipeline plus both
   workers. `max RGB error=0` is the contract for matching-output
   tests; tiling tests just check output is in range.
5. Optional: `filter-foo-rs/` - Rust sidecar for faster
   startup. CPU-only by default; GPU providers are opt-in
   Cargo features.

Don't add a Rust sidecar unless you need the first-call
speedup. The Python worker is always the default and is fully
functional.

---

## 12. The NAFNet "two procedures" UX pattern

The two procedures in this plug-in (`plug-in-nafnet-restore` and
`plug-in-nafnet-restore-region`) are the right pattern for
**1:1 spatial image-restoration models**. Different from LaMa:

- LaMa is **masked**: needs a selection to know which pixels to
  inpaint. Without a selection, nothing happens.
- NAFNet is **whole-image by default**: the natural operation is
  "process the whole photo". A selection is a *restriction*, not
  a *requirement*.

The GIMP menu order (`Filters > Enhance >`) matters here: put
whole-image first (the more common operation), then
selection-only (for targeted restoration). The leading underscore
on `_Restore Image (NAFNet)...` and `_Restore Selection (NAFNet)...`
gives Alt+R as the keyboard accelerator, which the user can
override in GIMP's keyboard shortcuts editor.

For the selection-only mode, the **64 px context padding** is
intentional: NAFNet needs surrounding pixels to know what the
"sharp" reference is. The 64 px is clipped to image bounds, so
edge selections still work. The restored region is the original
selection bbox; the context pixels are processed but reported
as "outside the updated bbox" in the v1 implementation, so the
GIMP display refresh is bounded to what the user expects.

In v2, crop the result PNG to the original selection bbox before
pasting, so the context ring's restored pixels are *not* visible
to the user. This is a known v1 limitation; the user might
notice a slight "halo" effect if they select a thin strip and the
context ring is much larger.

---

## 13. See also

- `AGENTS.md` at the project root - project conventions and rules.
- `README.md` - user-facing install and usage docs.
- `nafnet-worker-rs/OPTIMIZATION.md` - Rust worker post-mortem.
- `C:\Dev\GIMP_Native_Plugin\Gimp-lama-inpainting\Docs\NOTES.md` -
  the sibling plug-in's lessons. Most of the MINGW/MSVC, sidecar,
  and `.interp` lessons are shared between the two plug-ins.
- GIMP 3.x plug-in pitfalls (workspace root) - broader GIMP
  development traps.
