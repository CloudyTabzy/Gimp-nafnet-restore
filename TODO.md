# TODO

Pending work tracked across the project. Format: `- [ ] <item>`
plus the file/place where it belongs.

## Worker hardening (next session)

- [x] **Alpha-channel preservation in region mode** — done. The
      GIMP-side glue extracts the original alpha to a separate Y u8
      PNG, the worker combines it with the model output to produce
      RGBA, and the GIMP-side writes the combined RGBA back to the
      shadow buffer. Both workers (Python and Rust) accept an
      `--alpha` argument. Tests in `tests/test_pipeline.py`
      (`TestPythonWorkerWholeImage::test_alpha_preservation`,
      `TestRustWorkerWholeImage::test_alpha_preservation`) verify
      the alpha is preserved byte-for-byte through the sidecar
      round-trip.

- [x] **Crop the result PNG to the original selection bbox before
      pasting** — done. The GIMP-side glue now uses a GEGL graph
      with `gegl:rectangle` (crop to inner selection bbox) +
      `gegl:translate` (move to (sel_x, sel_y)) + `gegl:write-buffer`
      (paste at translated position). The context ring is no longer
      visible to the user and the result lands at the right
      pixel position. Helper: `paste_roi_into_shadow` in
      `nafnet-restore.py`. Fixed the position bug (the result was
      previously written at (0, 0) instead of (sel_x, sel_y)).

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
