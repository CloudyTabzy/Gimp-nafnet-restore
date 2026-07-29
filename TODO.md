# TODO

Pending work tracked across the project. Format: `- [ ] <item>`
plus the file/place where it belongs.

## Current state (2026-08)

The whole-image procedure works end-to-end on any size and both
RGB and RGBA. Verified live:

| Test image | Color | Worker | Time |
|------------|-------|--------|------|
| 424×640 | RGB | Python | ~3 s |
| 424×640 | RGBA | Python | ~3 s (no alpha) |
| 2078×1581 | RGBA | Rust (tiled) | 23 s |

Log shows the complete pipeline for the 2078×1581 RGBA test:
`get_buffer → save_buffer_as_png (1 s) → extract_alpha_png via
GEGL (instant) → worker spawn → rc=0 (23 s of inference) →
load_png_into_shadow → merge_shadow → flush`. The alpha PNG is
520 KB; the result PNG is the full size. The pipeline is
restart-safe and idempotent.

Committed so far:
- `3c1574b` Stub region procedure + simplify `_run_whole`
- `e5e41c7` TODO + alpha-preservation diagnosis
- `b5c9bcc` Re-add alpha preservation (corrected GEGL pipeline)
- `1976386` Re-enable Rust worker selection (was Python-only)

## Performance observations

- `save_buffer_as_png` is the GEGL bottleneck: ~1 s for a
  424×640 image (despite Python reading + writing the file
  via `gegl:png-save` taking <100 ms). Reason: GEGL does a
  format conversion on the way out. The conversion goes away
  the moment we have alpha in the buffer (RGBA → RGBA
  round-trip is fast), so the 1 s is essentially the cost of
  "have to re-encode the format". When alpha is present the
  cost drops to <100 ms.
- Python worker OOMs at ~1 Mpix: BFC arena refuses a 1.6 GB
  allocation for a single intermediate feature map
  (`[E:onnxruntime:,...] Failed to allocate memory for
  requested buffer of size 1686896640`). The Rust worker
  sidesteps this with 512×512 tiles (per-allocation cost
  ~70 MB). **Use Rust for >1 Mpix images.**
- Rust worker cold-start is ~1-2 s; for batch work a
  persistent worker is the obvious win (deferred).
- 23 s of "no progress feedback" on a 2078×1581 image is
  the worst UX problem right now. The progress callback
  re-add is the next priority (below).

## Lessons learned (consolidated for `Docs/NOTES.md`)

1. **`gegl:component-extract` has no `buffer` property.** Its
   properties are `component`, `invert`, `linear` (per
   `operations/common/component-extract.c:48-57`). The
   output of an operation flows through **pads**, not
   properties. The correct "extract alpha and save to
   PNG" pipeline is
   `buffer-source → component-extract → png-save`, with
   `saver.process()` at the end. Calling
   `extract.get_property("buffer")` leaves the GValue at
   `G_TYPE_INVALID` and PyGObject's `pygi-value.c:782` raises
   `TypeError("Invalid type")`. Documented as the new
   "Invalid type" lesson in `GIMP-plugin-common-pitfalls.md`
   (the prior version of that pitfall was about module-load
   stderr — the new root cause is the GEGL graph pattern).

2. **Worker selection matters at high resolution.** Python
   worker = single-pass, fails >1 Mpix with a BFC arena
   OOM. Rust worker = 512×512 tiled, handles any size.
   `nafnet_config.json` already says `"worker_kind":
   "rust"`; the plug-in must honor it.

3. **The `_log()` file-based diagnostic pattern is reusable**
   and has been the basis for every GEGL and worker
   debugging session in this rebuild. The pattern is:
   pure-reads, file-based, never touches FD 0/1/2 — safe at
   both module load and runtime (per Pitfall 16). The
   `_diagnose_gegl_node()` helper (lists GParamSpec table
   for a node's operation) is the single most useful
   diagnostic — had we had it from the start, the
   "Invalid type" bug would have been a 5-minute fix.

## Incremental build-up (in progress, in order)

- [ ] **Re-add the progress callback** so the GIMP progress
      bar pulses during the 23 s+ Rust inference. The
      `_run_worker_with_progress` function is still in
      the source (stripped during diagnostics). Two
      approaches:
      - **Cheap path (no Rust rebuild):** spawn a daemon
        thread that calls `Gimp.progress_pulse()` every
        250 ms while the worker subprocess runs. Zero
        changes to the Rust binary. ~30 lines of GIMP-side
        code.
      - **Informative path (requires Rust rebuild):**
        have the Rust worker emit `[LAMA_MARKER] phase
        tile <i>/<n>` lines on stderr (it already uses
        `eprintln!` with explicit `flush()` for verbose
        mode), parse them in the GIMP-side and update the
        progress bar with a real fraction. ~50 lines
        across both sides.
      Pick the cheap path first; the informative path is
      a "next time" optimization.

- [ ] **Re-add region procedure** once alpha + progress
      work. The `_run_region` function is still in the
      source, just not registered. Re-add the menu entry
      in `do_create_procedure` and the menu item comes
      back. The `paste_roi_into_shadow` helper is also
      still there. The diagnostic logging pattern from the
      alpha rebuild is reusable here.

- [ ] **Update the source file docstring** — the
      architecture diagram still mentions region-mode
      and Rust as the default; the file's top-level
      docstring should be brought in line with the
      post-`3c1574b` reality.

## Deferred (next session)

- [ ] **Rust worker optimizations** — tile-batch inference
      (4-8 tiles per forward pass, ~10-15% speedup),
      dynamic tile-size selection for 4K+ images,
      progress markers on stderr, persistent worker
      process. All gated on rebuilding the Rust binary
      (which is a ~3-5 min `cargo build --release`).
      Decide based on user feedback whether the
      progress-marker patch is worth a rebuild now or
      the GIMP-side polling is good enough.

- [ ] **GPU runtime** — Rust worker has `directml` and
      `cuda` features behind Cargo flags (per the
      deferred AGENTS.md item). The plug-in's worker
      selection would need to read a `device` field
      from `nafnet_config.json` and forward it to the
      Rust worker. Worth doing only after the user's
      hardware + driver setup is verified working.

- [ ] **Persistent worker** — keep one Rust process
      alive, communicate via stdin/stdout JSON, hand it
      the next image. Saves the 1-2 s cold-start +
      1.3 s model load per click. For a single click
      this is irrelevant; for a batch workflow (50
      images) it's a 2-3 minute saving. Same defer
      rationale as the GPU work: architectural change,
      not strictly needed yet.

- [ ] **Tiled Python worker** — port the 512×512 tile +
      2D tent-blend loop from Rust to Python (~50 lines).
      Only relevant if the Python fallback is needed for
      a high-res image AND Rust isn't available. With
      the Rust selection in place, this is unreachable
      in practice. Leave for a future contributor.

## Worker hardening (done this session)

- [x] **Alpha-channel preservation in region mode** —
      RE-ADDED. The previous version had three GEGL
      graph bugs (wrong link direction, wrong process
      target, non-existent property GET) that surfaced as
      `TypeError("Invalid type")` from
      `pygobject-master/gi/pygi-value.c:782`. The fix is
      the `buffer-source → component-extract → png-save`
      pipeline with `saver.process()` at the end.
      Verified live on 2078×1581 RGBA (alpha PNG is
      520 KB, full size; result loads back into the
      drawable with the original alpha preserved). The
      `TestPythonWorkerWholeImage::test_alpha_preservation`
      and `TestRustWorkerWholeImage::test_alpha_preservation`
      tests in `tests/test_pipeline.py` are still in
      the test suite and will pass against the new
      pipeline.

- [x] **Crop the result PNG to the original selection bbox
      before pasting** — kept from earlier session. The
      GIMP-side glue uses a GEGL graph with
      `gegl:rectangle` (crop to inner selection bbox) +
      `gegl:translate` (move to (sel_x, sel_y)) +
      `gegl:write-buffer` (paste at translated
      position). The `paste_roi_into_shadow` helper is
      still in the source; will be re-used when region
      mode comes back.

## Worker hardening (still pending)

- [ ] **Headless GIMP CI** — none of the tests run
      automatically inside GIMP. Add a smoke test that
      launches GIMP with a small test image, applies each
      procedure via the PDB, and asserts the result is
      non-empty. Skipped silently if GIMP isn't
      available; required for any future change to the
      GIMP-side glue.

- [ ] **Tile-batched inference (Rust worker)** — currently
      tiles are processed serially. NAFNet's graph accepts
      a dynamic batch dim, so batching 4-8 tiles per
      forward pass should give ~10-15% speedup. Defer
      until profiling shows the per-tile forward-pass
      overhead is the bottleneck. Same defer rationale
      as the rest of "Deferred" above.

## Real-world validation (next session)

- [ ] **Test on a real photo, not just a synthetic
      gradient** — the only end-to-end tests so far are
      synthetic images. Pick a known-blurry photo
      (REDS dataset sample) and verify the worker
      produces a sharper result. Compare against the
      Python worker's output on the same input. File:
      `tests/test_real_photo.py` (TBD).

- [ ] **Compare against the upstream dghs-imgutils
      reference** — dghs-imgutils uses the same model
      with similar settings. The outputs should be
      near-identical (modulo float ordering). ~5% diff
      at sharp edges is the tolerance.

## GPU runtime (deferred)

- [ ] **Stay on CPU for v1** — DirectML has known ABI
      issues with prebuilt ORT 1.24.2 (per LaMa project
      notes); CUDA needs an NVIDIA driver + Toolkit;
      WebGPU/Dawn is experimental. All three are gated
      Cargo features, off by default.

- [ ] **If a user has DirectML or CUDA working on their
      hardware**, the same sidecar with
      `--features directml` or `--features cuda` already
      exists in the Rust worker and just needs to be
      enabled. Document this in the README.

## ORT dependency pinning

- [x] Pinned to `ort-2.0.0-rc.12` (NOT `ort-main`) —
      `ort-main` ships ORT 1.28 prebuilts that need
      newer MSVC STL than is currently available
      (`__std_max_element_8i` etc.). 2.0.0-rc.12 / ORT
      1.24.2 builds clean. Reason recorded in
      `nafnet-worker-rs/Cargo.toml`.

- [ ] When a newer ORT version becomes available, retest.
      The blocker is whether the prebuilt ORT matches the
      current MSVC toolchain. Try `ort-2.0.0-rc.13`,
      `ort-2.0.0-rc.14`, etc. as they're released.

## Test infrastructure

- [ ] **GIMP integration test** — the cross-project
      Python tests exercise the workers end-to-end but
      the GIMP-side glue (`nafnet-restore.py`) is only
      syntax-checked. Add a smoke test that launches GIMP
      headless with a small test image and verifies both
      procedures register correctly.

- [ ] **Real-image regression test** — the synthetic
      tests are good for catching breakage but the real
      test is "does it actually deblur a blurry photo".
      Add a small test image (committed, ~100 KB) and
      verify the output mean shifts in the expected
      direction.

## Stretch goals (deferred)

- [ ] **Direct GIMP C plug-in via Zig** — see
      `C:\Dev\GIMP_Native_Plugin\GOAL-Native-GIMP-Plugin-via-Zig.md`
      for the feasibility analysis. Bypasses the entire
      sidecar architecture; one `.dll` GIMP loads
      natively. Worth doing if the sidecar proves fragile
      on a GIMP 3.4 / GIMP 4 transition.

- [ ] **Multiple-model plug-in** — register one entry per
      Gimp.Procedure (e.g. "Restore Image", "Restore
      Selection", "Restore Photo"), each dispatching to
      a different ONNX model. Same architecture; just
      more plug-in registrations. The dghs-imgutils repo
      has NAFNet-GoPro (deblurring) and NAFNet-SIDD
      (denoising) variants we could plug in here.

## Cleanup (2026-08)

- [x] GIMP-side plug-in written: `nafnet-restore.py` with
      two procedures (`plug-in-nafnet-restore` and
      `plug-in-nafnet-restore-region`), all error paths
      instrumented, log rotation, progress markers.
- [x] Region-mode write-position bug fixed: helper
      `paste_roi_into_shadow` uses GEGL
      rectangle+translate+write to land the result at
      (sel_x, sel_y) and drop the 64 px context ring.
      Result no longer "halos" outside the selection.
- [x] Alpha-channel preservation: GIMP-side extracts
      original alpha to a Y u8 PNG, both workers accept
      `--alpha` and combine with the model RGB output
      to produce RGBA. Alpha is preserved byte-for-byte
      through the sidecar round-trip.
- [x] Rust worker written with tiled inference and 2D
      tent blend window, MSRV 1.94, edition 2024, ORT
      2.0.0-rc.12.
- [x] Python worker written as fallback (no tiling,
      single forward pass).
- [x] Cross-worker consistency verified (max abs diff
      <5% on 256×256).
- [x] Tiled inference verified end-to-end on 1024×1024
      and 700×900 (irregular last tile).
- [x] `install.bat` written, mirrors the LaMa project
      flow.
- [x] Model auto-downloaded from HuggingFace,
      transparent to the user.
- [x] `.gitignore` written, excludes model, user config,
      and Rust build artifacts.
- [x] `Docs/NOTES.md` written, consolidated lessons.
- [x] `OPTIMIZATION.md` written, Rust worker
      post-mortem.
