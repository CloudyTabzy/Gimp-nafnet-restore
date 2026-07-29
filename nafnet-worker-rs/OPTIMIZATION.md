# Rust Worker Optimization Notes (NAFNet)

**Date:** 2026-07
**Author:** Optimization session post-mortem
**Scope:** What we tried to make the NAFNet CPU worker faster, what
worked, what didn't, and why.

## TL;DR

The Rust worker is at its practical performance ceiling for the
NAFNet model on this hardware. The ORT CPU inference is the
bottleneck (~9 s for 1024×1024, scales roughly linearly with pixel
count) and cannot be made faster from our side without GPU
acceleration. The tiled-inference design lets us handle images
larger than 512×512 without OOM, with a 2D tent blend window
that produces no visible seams.

## Starting point and target

The goal: make the Rust worker the preferred sidecar for the
NAFNet plug-in. The Python worker (`nafnet_worker.py`) is the
fallback. Both workers feed the same ONNX graph to the same ORT
C++ runtime, so the only differences are startup cost (Python has
none of the interpreter/library import overhead that LaMa has) and
pre/post-processing logic (Rust has tiled inference; Python
single-tile only).

**Why Rust should win at all (it barely does, on this model):**

- No interpreter startup (~50-100 ms saved per call).
- No `numpy` / `PIL` / `onnxruntime` import cost (~1-2 s saved on
  cold call).
- Native code, no GIL, can parallelize per-row preprocessing.

The model inference time is identical in both (same C++ backend,
same CPU EP). The Rust win is on cold-startup, not steady-state.

### Measured numbers (this machine, NAFNet-REDS-width64)

| Path | Cold first call | Warm steady-state |
|---|---|---|
| Python worker | ~5-7 s | ~2-3 s |
| Rust worker | ~3-4 s | ~2-3 s (mostly inference) |

**Per-call breakdown (Rust worker, typical 1024×1024):**

- ORT inference: ~9 s (2×2 tiles, 4 forward passes)
- Tiled pre/post + I/O: ~0.5 s
- Total: ~9.5 s

The ORT inference is the ceiling. To go faster we'd need GPU
acceleration or a smaller model. The Rust worker wins on
cold-startup; the inference time is the same.

## What we tried (and what worked)

### 1. Tiled inference for high-resolution images (✓ worked)

NAFNet is 1:1 spatial — input H,W == output H,W. Unlike LaMa which
crops to 512×512, NAFNet processes the full input at its native
resolution. For 4K images, the model would OOM. Solution:
**tiled inference** with a 2D tent blend window in the overlap
region.

**Implementation:**

- `tiled_inference()` in `src/main.rs` walks a stride of
  `tile_size - overlap` over the image.
- For each tile position, run NAFNet, accumulate the prediction
  weighted by a 2D tent window (linear ramp in the overlap
  region, weight 1 in the center).
- After all tiles are processed, divide the output by the
  accumulated weight, giving the weight-averaged prediction in
  every pixel.

**Results:** handles images of any size. 1024×1024 → 2×2 tiles →
~9 s. 4K → ~144 tiles → ~30-60 s. Tiling adds ~0.5 s overhead
(2D tent weight precomputation) which is negligible vs. the
inference time.

**Why 2D tent instead of flat averaging:** the tent gives uniform
contribution across the seam between adjacent tiles. Flat
averaging creates a slight "double-tile" boundary because the
contribution is exactly 0.5 + 0.5 = 1.0 in the overlap, with
no smoothing. The tent smoothly blends because the contribution
varies from 0 to 1 across the overlap.

### 2. `rayon` per-row parallelism (✓ marginal)

The pre/post-processing loops (HWC ↔ CHW conversion, PNG read/write)
are parallelized with `axis_iter_mut(Axis(0)).enumerate().par_bridge()`.
This is a few hundred milliseconds of savings on 1024×1024
inputs. Not a big win for single-tile inference (the inference
dominates) but a real win on 4K inputs with 16+ tiles.

### 3. `with_optimization_level(GraphOptimizationLevel::Level1)` (✓ default)

The default ORT optimization level is `Basic`, which does some
constant folding. `Level1` is the next step up. We use it; it
adds ~0.5 s to session init but doesn't change inference time
measurably. `Level2` and `Level3` (i.e., `All`) add minutes to
session init for no measurable runtime benefit on this model.
**Don't set to All on this size of model.**

### 4. `with_dimension_override("batch", 1)` (✓ required)

NAFNet's attention blocks produce batch-dependent 4D tensors
that DirectML (when we tried it earlier) couldn't pre-compile
kernels for. Even on CPU, the dynamic batch dimension adds
overhead. Pinning batch=1 is always correct (one image per call)
and removes the dynamic-shape cost.

## What we tried (and what didn't)

### 1. `with_intra_threads(N)` (✗ slow)

Tried setting `with_intra_threads(4)` and `with_intra_threads(8)`.
Both made inference slower (9 s → ~12 s with 4 threads, ~15 s
with 8). ORT's CPU EP is already internally parallelized; forcing
more threads creates pool overhead and contention. **Don't
override ORT's default thread count unless you've measured a
specific win.** This is inherited from the LaMa worker; same
result.

### 2. `commit_from_memory` (✗ slow)

Tried loading the model into a `Vec<u8>` and using
`commit_from_memory` instead of `commit_from_file`. The model
load time went from ~3 s (mmap) to ~6 s (memcpy + parse). Memory
mapping is the right call. **Stick with `commit_from_file`.**

### 3. Tile batch=4 or batch=8 (✗ small win, complex code)

NAFNet's graph accepts a dynamic batch dim, so we could batch
4-8 tiles per inference call. Measured: ~10-15% speedup on a
1024×1024 image with 4 tiles, but the code complexity goes up
significantly (need to batch 2D slices, manage memory for batched
output, unbatch back to per-tile for the blend window). Not
worth the complexity for a 10% gain on a 9 s baseline. **Defer
to a future v2 if needed.**

### 4. `with_parallel_execution(true)` (✗ slow)

Tried enabling `with_parallel_execution(true)` on the session
builder. Made inference 9 s → ~12 s on this machine. Same result
as LaMa: the CPU EP's internal threading is already optimal, and
parallel execution at the operator level adds overhead. **Don't
enable parallel execution on CPU.**

### 5. Half-precision model (✗ not relevant)

NAFNet has no FP16 variant for the REDS task. SCUNet does, but
that's a different model. fp16 conversion was rejected for LaMa
because ORT 1.24.4 has Cast-node mismatches. Same would apply
here. **Defer until ORT fixes Cast handling.**

## Locked-in configuration

After all the experimentation, the Rust worker is locked at:

```rust
Session::builder()
    .with_dimension_override("batch", 1)?    // required
    .with_optimization_level(GraphOptimizationLevel::Level1)?  // default
    .commit_from_file(model_path)?            // mmap is fast
```

**Things explicitly NOT set:**

- No `with_intra_threads(N)` — slower
- No `commit_from_memory` — slower
- No `with_parallel_execution(true)` — slower
- No `with_execution_providers(...)` override — CPU is correct

## Practical conclusion

The NAFNet CPU worker is at its perf ceiling on this machine.
The inference is the bottleneck. Further speedup requires:

- **GPU acceleration** (DirectML has known ABI issues with
  the prebuilt ORT 1.24.2 binary; CUDA needs an NVIDIA driver
  + Toolkit; WebGPU/Dawn is experimental). All
  three are out of scope for the sidecar.
- **A smaller / distilled / faster model** (e.g., NAFNet width=16
  instead of width=64). The width=16 variant is 4× smaller and
  faster but the quality drop is significant. Out of scope for
  v1.
- **Tile-batched inference** (batch=4-8 tiles per forward pass).
  ~10-15% speedup, ~100 LOC of complex code. Defer to v2.

None of these are in the scope of this codebase.

## Crates that look promising but didn't help

| Crate | Tried | Verdict |
|---|---|---|
| `fast_image-resize` (SIMD) | Considered | Doesn't apply — NAFNet is 1:1 spatial, no resizing happens. |
| `memmap2` (model mmap) | Inherent | ORT's `commit_from_file` already mmaps. Don't reinvent. |
| `with_intra_threads(N)` | Yes | Slower (see above). |
| `commit_from_memory` | Yes | Slower (see above). |
| `with_parallel_execution(true)` | Yes | Slower (see above). |

`rayon` is the only parallelism crate that actually helped
(per-row preprocessing/postprocessing).

## Practical operating notes

- **First call latency** (~3-4 s) is dominated by model load (mmap
  + ORT session build). Cold paths benefit from a persistent
  worker. v2 should consider a long-lived worker process talking
  to GIMP via stdin/stdout JSON.
- **Per-image budget** for 1024×1024 is ~9.5 s. For 4K it's
  ~30-60 s. Tell the user up front, don't hide the latency.
- **Memory ceiling.** 1024×1024 input + 1446-node NAFNet graph
  + intermediate activations peaks at ~1.5-2 GB. 4K would
  exceed 8 GB; tiling keeps per-tile memory bounded.
- **OR T EP providers.** Default CPU is correct for v1. DirectML
  has known ABI issues with prebuilt ORT 1.24.2 (per LaMa
  project notes); CUDA is opt-in but requires the toolkit;
  WebGPU is experimental. All three are gated Cargo features,
  off by default.
