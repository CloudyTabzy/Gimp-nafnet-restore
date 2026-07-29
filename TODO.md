# TODO

Pending work tracked across the project. Format: `- [ ] <item>`
plus the file/place where it belongs.

## Current state (2026-08)

The whole-image procedure works end-to-end on any size and both
RGB and RGBA. The region procedure is back in the menu and works
identically to the whole-image procedure when the user selects
the whole image; for sub-selections it uses a simplified
pipeline (save whole image, worker processes it, crop the result
to the selection bbox and write to the shadow buffer).

Verified live:

| Test image | Color | Procedure | Worker | Time |
|------------|-------|-----------|--------|------|
| 424×640 | RGBA | Restore Image | Rust (tiled) | ~3 s |
| 424×640 | RGBA | Restore Selection (whole) | Rust (delegated to _run_whole) | ~3 s |
| 2078×1581 | RGBA | Restore Image | Rust (tiled) | 23 s |

Both procedures emit the complete pipeline trace in `nafnet.log`.
The file-based log is safe at both module load and runtime (per
Pitfall 16). The 1 s save cost on the 424×640 image is the GEGL
format conversion; it drops to <100 ms when the buffer is already
in the target format.

Committed this session:
- `b5c9bcc` Re-add alpha preservation (corrected GEGL pipeline)
- `d799ed0` Re-add progress callback (daemon-thread pulse)
- `1976386` Re-enable Rust worker selection
- `4852629` Re-add region procedure + update source docstring
- `628c8bf` Fix region GEGL pipeline + add selection sensitivity
- `5df95fd` Fix no-selection detection (use `Selection.is_empty`)
- `8ab53b0` Fix `NameError` (re-add `mask_intersect`)
- `ad7a9f2` Fix region save pipeline (`gegl:crop` not `gegl:rectangle`)
- `217f2d3` Fix transparent region (`get_buffer()` not `get_shadow_buffer()`)
- `4e897ef` Fix paste math for whole-image selection
- `57e993e` Simplify `_run_region` (save whole image, crop result on paste)
- `fbe1278` Region delegates to `_run_whole` when selection is the whole image

## Performance observations

- `save_buffer_as_png` is the GEGL bottleneck: ~1 s for a
  424×640 image. Reason: GEGL does a format conversion on
  the way out. The conversion goes away the moment we have
  alpha in the buffer (RGBA → RGBA round-trip is fast), so
  the 1 s is essentially the cost of "have to re-encode the
  format". When alpha is present the cost drops to <100 ms.
- Python worker OOMs at ~1 Mpix: BFC arena refuses a 1.6 GB
  allocation for a single intermediate feature map
  (`[E:onnxruntime:,...] Failed to allocate memory for
  requested buffer of size 1686896640`). The Rust worker
  sidesteps this with 512×512 tiles (per-allocation cost
  ~70 MB). **Use Rust for >1 Mpix images.**
- Rust worker cold-start is ~1-2 s; for batch work a
  persistent worker is the obvious win (deferred).
- The progress callback (commit `d799ed0`) makes the GIMP
  bar pulse every 250 ms during the worker run, so the
  user sees feedback during the 23 s Rust inference on
  2078×1581.

## Lessons learned (consolidated for `Docs/NOTES.md`)

These are the GIMP/GEGL/PyGObject gotchas we hit this session.
Each is a "look right, do wrong" failure mode that's hard to
catch without reading the GIMP/GEGL source.

1. **GEGL operation output flows through pads, not properties.**
   `gegl:component-extract` has properties `component`,
   `invert`, `linear` — no `buffer`. `gegl:rectangle` has
   `x`, `y`, `width`, `height`, `color` — no `buffer`. The
   output of an operation flows through the **pads**, not
   properties. Calling `node.get_property("buffer")` leaves
   the GValue at `G_TYPE_INVALID` and PyGObject's
   `pygi-value.c:782` raises `TypeError("Invalid type")`.
   The correct pipelines are:
   - Save: `buffer-source → gegl:component-extract → png-save`
     (with `saver.process()` at the end).
   - Load + paste: `png-load → gegl:crop → gegl:write-buffer`
     (no translate, since the crop's output extent is the
     desired destination position).

2. **`gegl:rectangle` is a DRAW op, not a crop.** Per
   `operations/common/rectangle.c`, the rectangle's output
   comes from an internal `gegl:color → gegl:crop` chain,
   so the input pad is effectively ignored — the result is
   a green rectangle of the requested size regardless of
   the input. We hit this three times in different places
   (alpha extract, region save, region paste) before
   catching the pattern. **Use `gegl:crop` directly when
   you want a crop.**

3. **Gimp.Drawable.get_shadow_buffer() is uninitialized
   for fresh drawables with no pending changes.** Use
   `get_buffer()` to read the current pixel state. The
   shadow buffer is for **writing** pending changes, not
   reading the current state.

4. **Gimp.Selection.is_empty(image) is the right API for
   "does the user have a selection?"** Gimp.Drawable.mask_intersect()
   always returns the full drawable bounds with True
   when there's no active selection — it's the "what's
   the bounding box of the intersection?" API, not the
   "is there a selection?" API.

5. **GIMP 3.2 has no built-in mechanism to grey out a
   procedure based on the active image's selection state.**
   `GimpProcedureSensitivityMask` is drawable-based only
   (DRAWABLE / DRAWABLES / NO_DRAWABLES / NO_IMAGE /
   ALWAYS). The `do_set_sensitivity` virtual method runs
   before the user clicks; `args[0]` is `None` at that
   time. Best workaround: `Gimp.message()` + return SUCCESS
   in the run body when the condition isn't met. The
   message shows briefly in the status bar and logs
   permanently to the Error Console.

6. **The `_log()` file-based diagnostic pattern is reusable**
   and has been the basis for every GEGL and worker
   debugging session in this rebuild. The pattern is:
   pure-reads, file-based, never touches FD 0/1/2 — safe at
   both module load and runtime (per Pitfall 16). The
   `_diagnose_gegl_node()` helper (lists GParamSpec table
   for a node's operation) is the single most useful
   diagnostic — had we had it from the start, several of
   the bugs above would have been 5-minute fixes.

## Done this session

- [x] **Re-add the progress callback** — daemon thread
      calls `Gimp.progress_pulse()` + `progress_set_text`
      every 250 ms while the worker subprocess runs. ~50
      lines (`_pulse_during_subprocess` helper). Confirmed
      live: status bar shows "Running NAFNet inference
      (~30 s for 2K images)..." during the 23 s Rust run.
      (`d799ed0`)
- [x] **Re-add region procedure** — the procedure is back
      in the menu, works for whole-image selections
      (delegates to `_run_whole` for guaranteed identical
      behavior to Restore Image), and uses a simplified
      pipeline for sub-selections (save whole image,
      worker processes it, crop result to selection bbox).
      Live-tested: identical log output to Restore Image
      for the whole-image case. (`fbe1278`)
- [x] **Update the source file docstring** — the file's
      top-level docstring now describes the current
      architecture: image types, worker selection, alpha
      side channel, progress callback, and the GIMP 3.2
      limitation around selection-aware sensitivity.
      (`4852629`)

## Next session (in order)

- [ ] **Update `Docs/NOTES.md` to consolidate the GEGL /
      GIMP gotchas** — the 6 lessons above are currently
      scattered across commit messages; the project's
      `Docs/NOTES.md` is the durable place. Per AGENTS.md
      rule 1, never delete content from `Docs/NOTES.md` —
      add a new section at the end.

- [ ] **Test on a real photo** — the only end-to-end
      tests so far are synthetic / uniform images. The
      424×640 test image the user has is too uniform to
      visually verify the region pipeline (model output
      ≈ input on a clean image, so the paste looks like
      "nothing happened"). A real noisy or blurry photo
      would let us see whether the sub-selection pipeline
      actually does anything.

- [ ] **Cross-project integration test for the region
      procedure** — `tests/test_pipeline.py` covers the
      workers but not the GIMP-side glue. Add a test that
      invokes `_run_region` with a mock worker, verifies
      the GEGL pipeline produces a non-empty
      `result.png` of the expected dimensions.

## Deferred (architectural)

- [ ] **Rust worker optimizations** — tile-batch inference
      (4-8 tiles per forward pass, ~10-15% speedup),
      dynamic tile-size selection for 4K+ images,
      progress markers on stderr, persistent worker
      process. All gated on rebuilding the Rust binary
      (which is a ~3-5 min `cargo build --release`).
- [ ] **GPU runtime** — Rust worker has `directml` and
      `cuda` features behind Cargo flags. The plug-in's
      worker selection would need to read a `device` field
      from `nafnet_config.json` and forward it to the
      Rust worker. Worth doing only after the user's
      hardware + driver setup is verified working.
- [ ] **Persistent worker** — keep one Rust process
      alive, communicate via stdin/stdout JSON, hand it
      the next image. Saves the 1-2 s cold-start +
      1.3 s model load per click. For batch workflows
      only.
- [ ] **Tiled Python worker** — port the 512×512 tile +
      2D tent-blend loop from Rust to Python (~50
      lines). With the Rust selection in place, this is
      unreachable in practice. Leave for a future
      contributor.

## Worker hardening (still pending)

- [ ] **Headless GIMP CI** — none of the tests run
      automatically inside GIMP. Add a smoke test that
      launches GIMP with a small test image, applies
      each procedure via the PDB, and asserts the result
      is non-empty. Skipped silently if GIMP isn't
      available; required for any future change to the
      GIMP-side glue.
- [ ] **Tile-batched inference (Rust worker)** — currently
      tiles are processed serially. NAFNet's graph
      accepts a dynamic batch dim, so batching 4-8 tiles
      per forward pass should give ~10-15% speedup. Defer
      until profiling shows the per-tile forward-pass
      overhead is the bottleneck.

## Real-world validation (next session)

- [ ] **Test on a real photo, not just a synthetic
      gradient** — pick a known-blurry photo (REDS
      dataset sample) and verify the worker produces a
      sharper result. Compare against the Python
      worker's output on the same input. File:
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
- [ ] **Multiple-model plug-in** — register one entry
      per Gimp.Procedure (e.g. "Restore Image", "Restore
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
      Rust build artifacts, and editor/OS junk.
- [x] `Docs/NOTES.md` written, consolidated lessons.
- [x] `OPTIMIZATION.md` written, Rust worker
      post-mortem.
