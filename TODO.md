# TODO

Pending work tracked across the project. Format: `- [ ] <item>`
plus the file/place where it belongs.

## Current state (2026-08)

The whole-image procedure works end-to-end: a 424×640 RGB layer
restores in ~4 s (1 s save, 3 s Python-worker inference, <50 ms
load + merge). The region procedure and alpha-channel preservation
are stubbed and will be rebuilt incrementally — see below.

Verified live by running
`Filters > Enhance > Restore Image (NAFNet)...` on a 424×640
RGB Layer. The log shows the complete pipeline:
`get_buffer → save_buffer_as_png → worker spawn → rc=0 → load_png_into_shadow → merge_shadow → flush`.

Committed as `3c1574b` (Stub region procedure + simplify
_run_whole: drop alpha, drop Rust).

## Incremental build-up (in order)

- [ ] **Re-add alpha preservation** for RGBA drawables. The
      plug-in currently ignores `drawable.has_alpha()` and the
      worker is invoked without `--alpha`, so an RGBA drawable
      ends up with the model's RGB output and a fresh opaque
      alpha. For RGBA inputs we want the original alpha
      preserved byte-for-byte (the previous version did this;
      tests in `tests/test_pipeline.py` covered it).

      **Root cause of the previous failure** (researched
      2026-08, see `Docs/NOTES.md` for the full write-up):

      1. The OLD `extract_alpha_png` had three bugs in how it
         composed the GEGL graph:
         - `extract.link(source)` was the wrong direction
           (GeglNode links go from the source's output pad to
           the sink's input pad, so it should have been
           `source.link(extract)`).
         - `extract.process()` was called on an intermediate
           node. Only the LAST node in the chain (the sink)
           should be processed.
         - `extract.get_property("buffer")` was the actual
         trigger of "Invalid type". `gegl:component-extract`
         has properties `component`, `invert`, `linear` — no
         `buffer`. `gegl_node_get_property` then leaves the
         GValue at `G_TYPE_INVALID` and PyGObject's
         `_pygi_value_to_pyobject` raises
         `TypeError("Invalid type")` (see
         `pygobject-master/gi/pygi-value.c:782`).

      2. The correct pipeline for "extract alpha and save to
         PNG" is:
         ```
         buffer-source (input=src_buffer)
              ↓
         component-extract (component=alpha)
              ↓
         png-save (path=alpha_path)
         ```
         with `saver.process()` at the end. No
         `get_property("buffer")` needed.

      **Diagnostic log lines to add** (safe at runtime, file-based):
      - Right before each `set_property` call, log the property
        name and value type.
      - Right before the `saver.process()` call, log the
        connected graph: list of pads (in → out).
      - On exception, log the exception class, message, AND the
        full GEGL node's class name (so we know which node in
        the chain threw).

      **Files to change**:
      - `nafnet-restoration-py/nafnet-restore.py::extract_alpha_png`
        — rewrite with the correct GEGL pipeline
      - `nafnet-restoration-py/nafnet-restore.py::_run_whole` —
        call `extract_alpha_png(buffer, alpha_path)` when
        `has_alpha=True`, pass `alpha_path=alpha_path` to the
        worker when set
      - Worker Python path already handles `--alpha` correctly
        (see `nafnet_worker.py:98-114`); the fallback to
        `original_alpha` when shape mismatches is what we want
        for RGB inputs anyway

- [ ] **Re-add region procedure** once alpha works. The
      `_run_region` function is still in the source, just not
      registered. Re-add the menu entry in `do_create_procedure`
      and the menu item comes back. The `paste_roi_into_shadow`
      helper is also still there (verified by inspection). The
      diagnostic logging pattern from #1 (above) is reusable.

- [ ] **Re-add Rust worker** as the default. Currently
      `_run_whole` hard-codes Python. Move the choice back to
      `_resolve_worker_command` (which already exists and works)
      once we're confident both paths succeed. Rust cold-start is
      ~3× faster for the Python cold start of ORT/Pillow
      imports.

- [ ] **Re-add the progress callback** (`_run_worker_with_progress`)
      so the GIMP progress bar pulses during the 3s+ inference.
      We had it before, just stripped for diagnostics.

- [ ] **Add tiling** in the Python worker for high-res images.
      The Rust worker already has 512×512 tiles + 2D tent blend.
      Python can mirror that, or we can document "use the Rust
      worker for >1 Mpix images" and leave Python single-pass.

- [ ] **Update the docstring** in the source file to reflect the
      current state (the architecture diagram still mentions
      region-mode and Rust as the default; both are now stubs).

## Worker hardening (next session)

- [x] **Alpha-channel preservation in region mode** — STUBBED
      2026-08. Was done in an earlier session but the GEGL
      pipeline had bugs that surfaced as "Invalid type" — see
      "Re-add alpha preservation" above for the rebuild plan.
      The previous test
      `TestPythonWorkerWholeImage::test_alpha_preservation` and
      `TestRustWorkerWholeImage::test_alpha_preservation`
      are still in `tests/test_pipeline.py` and will pass once
      the new GEGL pipeline is in place.

- [x] **Crop the result PNG to the original selection bbox before
      pasting** — kept from earlier session. The GIMP-side glue
      uses a GEGL graph with `gegl:rectangle` (crop to inner
      selection bbox) + `gegl:translate` (move to (sel_x, sel_y))
      + `gegl:write-buffer` (paste at translated position). The
      `paste_roi_into_shadow` helper is still in the source; will
      be re-used when region mode comes back.

- [ ] **Headless GIMP CI** — none of the tests run automatically
      inside GIMP. Add a smoke test that launches GIMP with a
      small test image, applies each procedure via the PDB, and
      asserts the result is non-empty. Skipped silently if GIMP
      isn't available; required for any future change to the
      GIMP-side glue.

- [ ] **Tile-batched inference (Rust worker)** — currently tiles
      are processed serially. NAFNet's graph accepts a dynamic
      batch dim, so batching 4-8 tiles per forward pass should
      give ~10-15% speedup. Defer until profiling shows the
      per-tile forward-pass overhead is the bottleneck.

- [ ] **Tiled Python worker** — the Python worker is the fallback;
      it currently has no tiling (only handles single-tile inputs).
      Port the tile-and-blend loop from the Rust worker (~50 lines)
      to Python if 4K inputs become a real use case in Python mode.

## Real-world validation (next session)

- [ ] **Test on a real photo, not just a synthetic gradient** —
      the only end-to-end tests so far are synthetic images.
      Pick a known-blurry photo (REDS dataset sample) and verify
      the worker produces a sharper result. Compare against the
      Python worker's output on the same input. File:
      `tests/test_real_photo.py` (TBD).

- [ ] **Compare against the upstream dghs-imgutils reference** —
      dghs-imgutils uses the same model with similar settings.
      The outputs should be near-identical (modulo float ordering).
      ~5% diff at sharp edges is the tolerance.

## GPU runtime (deferred)

- [ ] **Stay on CPU for v1** — DirectML has known ABI issues with
      prebuilt ORT 1.24.2 (per LaMa project notes); CUDA needs
      an NVIDIA driver + Toolkit; WebGPU/Dawn is experimental.
      All three are gated Cargo features, off by default.

- [ ] **If a user has DirectML or CUDA working on their hardware**,
      the same sidecar with `--features directml` or
      `--features cuda` already exists in the Rust worker and just
      needs to be enabled. Document this in the README.

## ORT dependency pinning

- [x] Pinned to `ort-2.0.0-rc.12` (NOT `ort-main`) — `ort-main` ships
      ORT 1.28 prebuilts that need newer MSVC STL than is
      currently available (`__std_max_element_8i` etc.).
      2.0.0-rc.12 / ORT 1.24.2 builds clean. Reason recorded in
      `nafnet-worker-rs/Cargo.toml`.

- [ ] When a newer ORT version becomes available, retest. The
      blocker is whether the prebuilt ORT matches the current MSVC
      toolchain. Try `ort-2.0.0-rc.13`, `ort-2.0.0-rc.14`, etc. as
      they're released.

## Test infrastructure

- [ ] **GIMP integration test** — the cross-project Python tests
      exercise the workers end-to-end but the GIMP-side glue
      (`nafnet-restore.py`) is only syntax-checked. Add a smoke
      test that launches GIMP headless with a small image and
      verifies both procedures register correctly.

- [ ] **Real-image regression test** — the synthetic tests are
      good for catching breakage but the real test is "does it
      actually deblur a blurry photo". Add a small test image
      (committed, ~100 KB) and verify the output mean shifts in
      the expected direction.

## Stretch goals (deferred)

- [ ] **Persistent worker** — instead of spawning a new worker per
      call (with model reload each time), keep one worker running
      and communicate via stdin/stdout JSON. The model loading
      (~3-5 s) disappears from the per-call budget. Uses the same
      temp PNG handoff or upgrades to length-prefixed binary IPC.
      Worth doing for users who run restoration on many images in
      a row.

- [ ] **Direct GIMP C plug-in via Zig** — see
      `C:\Dev\GIMP_Native_Plugin\GOAL-Native-GIMP-Plugin-via-Zig.md`
      for the feasibility analysis. Bypasses the entire sidecar
      architecture; one `.dll` GIMP loads natively. Worth doing
      if the sidecar proves fragile on a GIMP 3.4 / GIMP 4
      transition.

- [ ] **Multiple-model plug-in** — register one entry per
      Gimp.Procedure (e.g. "Restore Image", "Restore Selection",
      "Restore Photo"), each dispatching to a different ONNX model.
      Same architecture; just more plug-in registrations. The
      dghs-imgutils repo has NAFNet-GoPro (deblurring) and
      NAFNet-SIDD (denoising) variants we could plug in here.

## Cleanup (2026-08)

- [x] GIMP-side plug-in written: `nafnet-restore.py` with two
      procedures (`plug-in-nafnet-restore` and
      `plug-in-nafnet-restore-region`), all error paths
      instrumented, log rotation, progress markers.
- [x] Region-mode write-position bug fixed: helper
      `paste_roi_into_shadow` uses GEGL rectangle+translate+write
      to land the result at (sel_x, sel_y) and drop the 64 px
      context ring. Result no longer "halos" outside the selection.
- [x] Alpha-channel preservation: GIMP-side extracts original
      alpha to a Y u8 PNG, both workers accept `--alpha` and
      combine with the model RGB output to produce RGBA. Alpha is
      preserved byte-for-byte through the sidecar round-trip.
- [x] Rust worker written with tiled inference and 2D tent blend
      window, MSRV 1.94, edition 2024, ORT 2.0.0-rc.12.
- [x] Python worker written as fallback (no tiling, single
      forward pass).
- [x] Cross-worker consistency verified (max abs diff <5% on
      256×256).
- [x] Tiled inference verified end-to-end on 1024×1024 and
      700×900 (irregular last tile).
- [x] `install.bat` written, mirrors the LaMa project flow.
- [x] Model auto-downloaded from HuggingFace, transparent to
      the user.
- [x] `.gitignore` written, excludes model, user config, and
      Rust build artifacts.
- [x] `Docs/NOTES.md` written, consolidated lessons.
- [x] `OPTIMIZATION.md` written, Rust worker post-mortem.
