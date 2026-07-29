# NAFNet Restoration Plug-in

> A GIMP 3.2+ filter plug-in for NAFNet image restoration. The
> sidecar architecture mirrors the
> [CloudyTabzy/Gimp-lama-inpainting](https://github.com/CloudyTabzy/Gimp-lama-inpainting)
> plug-in: the GIMP side is pure stdlib + `gi`/`Gimp`/`GimpUi`/`Gegl`,
> the ML worker (Rust by default, Python fallback) runs in a
> separate process that talks ORT CPU.

This file is for AI agents and human contributors who need to get
up to speed on the project quickly. Read this before making any
non-trivial change.

---

## What this project is

A GIMP filter that runs the NAFNet (Non-linear Activation Free
Network) image restoration model on the active drawable. The
model is 1:1 spatial and 3-channel RGB; the plug-in exposes two
filters (whole-image and selected-region) backed by the same
sidecar worker.

The project is laid out as two sub-projects in one repo, mirroring
the LaMa plug-in structure:

```
Gimp-restoration-plugin/
├── nafnet-restoration-py/      # the GIMP plug-in (Python + GEGL glue)
├── nafnet-worker-rs/           # Rust sidecar (tiled inference, default)
└── tests/                      # cross-project tests
```

The Rust worker is the default when its binary is present; the
Python worker is the fallback. The Python side of the plug-in only
imports `gi`, `Gimp`, `GimpUi`, `Gegl`, `GLib` — no `numpy`, no
`onnxruntime`, no `PIL`. ML inference is always out of process.

---

## Build, test, install

```bash
# Syntax check (no GIMP required)
python -m py_compile nafnet-restoration-py/nafnet-restore.py
python -m py_compile nafnet-restoration-py/nafnet_worker.py

# Cross-project tests (Python worker required, Pillow + numpy + onnxruntime)
python tests/test_worker_python.py
python tests/test_workers_consistency.py

# Rust worker (optional, opt-in)
cd nafnet-worker-rs
cargo build --release
cargo test --release
cd ..
python tests/test_worker_rust.py

# Install (Windows)
cd nafnet-restoration-py
install.bat
```

The installer writes per-user `.interp` files into
`%APPDATA%\GIMP\3.2\interpreters\` and copies the plug-in files into
`%APPDATA%\GIMP\3.2\plug-ins\nafnet-restore\`. It also attempts to
build the Rust worker via `cargo install` and copy it next to the
Python worker; if `cargo` isn't available the Python worker is the
default.

---

## Architecture

```
GIMP process  (MINGW Python 3.14, gi + GEGL only)
  └─ #!nafnet-gimp-python  (shebang → user-level .interp mapping)
       └─ nafnet-restore.py
            ├─ exports drawable + (for region mode) selection ROI to temp PNG
            ├─ spawns worker subprocess (Rust or Python)
            └─ loads result PNG into drawable shadow buffer, merge_shadow
                    │
                    ▼
           worker process  (Rust or Python, with Pillow + onnxruntime)
               onnxruntime / ort CPU provider
               ?? model pipeline: load PNG -> HWC f32 [0, 1] -> (Rust:
                  tile + blend overlap; Python: single forward) -> NAFNet
                  -> HWC f32 -> save PNG
```

Two GIMP-side procedures:

- `plug-in-nafnet-restore` — passes the full drawable PNG to the
  worker. Worker processes the whole image. Tiled internally for
  large inputs.
- `plug-in-nafnet-restore-region` — gets the selection bbox, expands
  by 64 px context (clipped to image bounds), crops the drawable
  to that ROI, runs the worker, and pastes the result back. Only
  the original selection bbox is reported as updated; the model
  produces context pixels which the GEGL paste silently overwrites.

---

## Where to read for context

Before making any meaningful change, read in this order:

1. **This file** (you're here). Project overview, rules, where to look.
2. **`README.md`** — user-facing install + usage.
3. **`nafnet-restoration-py/Docs/NOTES.md`** — consolidated lessons
   for the GIMP-side integration. Mirrors the LaMa project's
   `NOTES.md` (MINGW/MSVC ABI wall, sidecar pattern, `.interp`
   shebang, image-scoped procedure gotcha, subprocess pipe leak on
   Windows).
4. **`nafnet-worker-rs/OPTIMIZATION.md`** (TBD) — Rust worker
   post-mortem. Currently the only optimization note is "tile with
   2D tent blend at the overlap".

For the bigger cross-project context (external reference checkouts,
pitfalls docs that don't belong in the repo, session history), look
at the parent workspace directory (the directory that contains
this repo, e.g. one level up from the project root).

---

## Rules for AI agents and contributors

1. **Never modify `nafnet-restoration-py/Docs/NOTES.md`** in a way
   that loses lessons. It is a chronological log of gotchas and
   architectural decisions; the value is in not repeating them.
   Adding new sections is fine.

2. **Never import `numpy`, `onnxruntime`, `PIL`, or `cv2` in the
   GIMP-side plug-in** (`nafnet-restore.py`). That is the entire
   point of the sidecar architecture. The plug-in runs in GIMP's
   MINGW Clang Python 3.14, which cannot load MSVC-built wheels
   (numpy, onnxruntime, PIL on PyPI are all MSVC-built). ML is
   always out of process.

3. **Never re-exec across CRTs.** GIMP's wire protocol (CRT-level
   file descriptors) does not survive an MSVC↔MINGW handoff. The
   user-level `.interp` alias is the supported way to start a
   plug-in in a different Python.

4. **Never patch GIMP's installation files** (under `<GIMP
   install>\lib\gimp\3.0\interpreters\`). Per-user `.interp` files
   in `%APPDATA%\GIMP\3.2\interpreters\` are the supported way.

5. **Never commit the model file** (`NAFNet-REDS-width64_v1.onnx`,
   275 MB). It's in `.gitignore`. `install.bat` downloads it
   transparently from HuggingFace. The same rule applies to
   `nafnet_config.json` (user-specific, auto-rewritten by
   install.bat) and `Cargo.lock` is intentionally NOT committed
   (binary crate, we want reproducible builds — see comment in
   `.gitignore`).

6. **Run the cross-project tests before committing.**
   `tests/test_worker_rust.py`, `tests/test_worker_python.py`, and
   `tests/test_workers_consistency.py` together exercise both
   workers and assert they produce equivalent output. If you
   change the inference algorithm (input preprocessing, output
   postprocessing, tile/blend logic, output range), these tests
   may need updating.

7. **If you change the inference pipeline, change a contract.**
   The GIMP-side glue, both workers, and the ONNX model all
   implement the same pipeline contract. Any of them changing
   means the test suite's output-equality assertion may fail and
   need updating.

8. **Document the output convention.** NAFNet outputs pixel
   values in `[0, 1]` for well-formed input. Both workers
   clip the output to `[0, 1]` defensively. If you change this
   (e.g., a new model that outputs in a different range), update
   both workers and the GIMP-side glue.

9. **At session end, review what stale/abandoned artifacts remain**
   and flag them. The Python worker's lack of tiling is a known
   v1 limitation; if 4K tiling becomes a real need, port the
   tile-and-blend loop from the Rust worker to the Python one
   (it's ~50 lines).

10. **If you don't know whether something is the right approach,**
    check `nafnet-restoration-py/Docs/NOTES.md` first. The
    `Gimp-lama-inpainting/Docs/NOTES.md` in the sibling project
    is also directly relevant — most of the GIMP-side lessons
    are shared between the two plug-ins.

---

## Code style and conventions

### Python (nafnet-restoration-py/, tests/)

- The plug-in's first line is `#!nafnet-gimp-python`. The
  installer writes two `.interp` files mapping this alias to
  GIMP's bundled Python 3.14 (`bin\python.exe` for console,
  `bin\pythonw.exe` for GUI).
- The plug-in class is `NafnetRestore(Gimp.PlugIn)`.
- All progress calls go through `_safe_progress`.
- All subprocess calls use `subprocess.Popen` (not `run`) with
  a daemon reader thread, so the GIMP UI stays responsive. See
  `_run_worker_with_progress`.
- The worker spawns with the same `CREATE_NO_WINDOW` +
  `STARTUPINFO(STARTF_USESHOWWINDOW | SW_HIDE)` dance the LaMa
  plug-in uses. Only the worker console is hidden; the GIMP
  console (from `gimp --verbose`) remains visible.

### Rust (nafnet-worker-rs/)

- Edition 2024, MSRV 1.94.
- CLI via `clap` derive; markers via `eprintln!` with explicit
  `flush()`.
- The session is built with `with_dimension_override("batch", 1)`
  and `with_optimization_level(GraphOptimizationLevel::Level1)`.
  Do not add `with_intra_threads` — see the LaMa project's
  `OPTIMIZATION.md` for the empirical reason.
- Use `axis_iter_mut(Axis(0)).par_bridge()` for per-row
  parallelism. Rayon owns the thread pool for this layer.
- Tiling (the only complexity vs. the Python worker): the
  `tiled_inference` function in `src/main.rs` walks a stride of
  `tile_size - tile_overlap` over the image, runs NAFNet on
  each tile, and blends predictions in the overlap region with
  a 2D tent window (`make_blend_window`).
- The worker accepts an optional `--alpha` argument. When
  provided, the output is RGBA with the original alpha channel
  preserved byte-for-byte. The GIMP-side glue extracts the alpha
  separately (via `gegl:component-extract component=alpha`) and
  passes it as a side channel. Tests in
  `tests/test_pipeline.py::TestRustWorkerWholeImage::test_alpha_preservation`
  verify the contract.

### Documentation

- `nafnet-restoration-py/Docs/NOTES.md` is for *human* readers,
  not for the code to generate. Keep it current when you change
  behavior.
- `nafnet-worker-rs/OPTIMIZATION.md` is the place to record
  optimization attempts (successes and failures). The LaMa
  project's version is the template.

---

## Layout (what's where)

```
Gimp-restoration-plugin/
├── AGENTS.md                 ← this file
├── README.md                 ← user-facing install/usage
├── NEXT-STEPS.md             ← in-flight planning notes
├── .gitignore
│
├── NAFNet-REDS-width64_v1.onnx        ← the model (not committed; downloaded by install)
│
├── nafnet-restoration-py/     ← the GIMP plug-in (Python + GEGL glue)
│   ├── nafnet-restore.py      ← GIMP-side entry, registered as `plug-in-nafnet-restore`
│   │                            and `plug-in-nafnet-restore-region`
│   ├── nafnet_worker.py       ← Python ORT sidecar (default worker if no Rust binary)
│   ├── install.bat            ← Windows per-user installer
│   ├── gimp-verbose.bat       ← launches GIMP with --verbose
│   ├── nafnet_config.json     ← generated by install.bat (user-specific)
│   └── Docs/
│       └── NOTES.md           ← consolidated lessons
│
├── nafnet-worker-rs/         ← the Rust sidecar (default worker)
│   ├── Cargo.toml             ← MSRV 1.94, ORT 2.0.0-rc.12
│   ├── src/main.rs            ← CLI + tiled inference + blend window
│   ├── tests/integration.rs   ← end-to-end test
│   └── scripts/run-tests.ps1  ← `cargo test --release` wrapper
│
├── tests/                     ← cross-project tests
│   ├── test_worker_rust.py
│   ├── test_worker_python.py
│   └── test_workers_consistency.py
│
└── tools/                     ← dev/CI scripts
    ├── inspect_onnx.py
    └── test_inference.py
```

External checkouts at the workspace root are not in this repo.
