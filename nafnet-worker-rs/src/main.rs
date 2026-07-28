//! NAFNet image restoration sidecar worker (Rust CPU).
//!
//! CLI mirrors the Python `nafnet_worker.py` one-for-one so the GIMP
//! plug-in can swap between Rust and Python transparently. The model
//! (NAFNet-REDS-width64_v1) is 1:1 spatial: input `(1, 3, H, W)`
//! float32 in `[0, 1]`, output same shape and range. For images
//! larger than the tile size, the worker processes overlapping tiles
//! and blends predictions in the overlap region with a 2D tent
//! window so seams are not visible.
//!
//! Pipeline:
//!   load PNG -> HWC f32 [0, 1] -> tile (default 512x512, 32px overlap)
//!   -> for each tile: HWC -> CHW -> NAFNet -> CHW -> HWC -> place in
//!   output with ramp weight -> finalize: divide by accumulated weight
//!   -> if --alpha was given, combine the RGB output with the
//!   original alpha bytes and save RGBA; otherwise save plain RGB.
//!
//! When the image fits in a single tile, the tile loop runs once and
//! the blend is a no-op.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use ndarray::{s, Array2, Array3, Array4};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(
    name = "nafnet-worker",
    about = "NAFNet image restoration sidecar (Rust CPU)"
)]
struct Args {
    /// Input PNG path (RGB or RGBA).
    #[arg(long)]
    image: PathBuf,

    /// Output PNG path. Same spatial size as input. RGBA if
    /// `--alpha` is provided, RGB otherwise.
    #[arg(long)]
    output: PathBuf,

    /// Path to the NAFNet ONNX model.
    #[arg(long)]
    model: PathBuf,

    /// Optional Y u8 PNG containing the original alpha channel. When
    /// provided, the output is RGBA with this alpha preserved
    /// byte-for-byte. When omitted, the output is plain RGB.
    #[arg(long)]
    alpha: Option<PathBuf>,

    /// Tile size in pixels (square). Default 512. Larger tiles give
    /// better quality at higher memory cost; smaller tiles use less
    /// memory but introduce more overlap-blend artifacts.
    #[arg(long, default_value_t = 512)]
    tile_size: usize,

    /// Overlap between adjacent tiles in pixels. Default 32. Should
    /// be much smaller than tile_size (typical: 4-10% of tile).
    #[arg(long, default_value_t = 32)]
    tile_overlap: usize,
}

fn main() -> Result<()> {
    if let Err(e) = run() {
        // Print a single line on stderr so the GIMP parent can
        // surface the failure in its log without a multi-line
        // traceback that gets truncated.
        eprintln!("nafnet-worker: {:#}", e);
        std::process::exit(1);
    }
    Ok(())
}

fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .try_init();

    let args = Args::parse();

    if args.tile_size == 0 {
        return Err(anyhow!("--tile-size must be > 0"));
    }
    if args.tile_overlap >= args.tile_size {
        return Err(anyhow!(
            "--tile-overlap ({}) must be smaller than --tile-size ({})",
            args.tile_overlap,
            args.tile_size
        ));
    }

    let t_total = Instant::now();
    marker("start");

    let img = load_image(&args.image)
        .with_context(|| format!("failed to load image: {}", args.image.display()))?;
    let (h, w) = (img.dim().0, img.dim().1);
    marker(&format!("loaded {}x{}", w, h));

    let t_infer = Instant::now();
    let (mut session, _provider) = build_session(&args.model)
        .with_context(|| format!("failed to build ORT session for {}", args.model.display()))?;
    let restored = run_inference(&mut session, &img, args.tile_size, args.tile_overlap)?;
    marker(&format!(
        "inference_done ({:.0}ms)",
        t_infer.elapsed().as_secs_f32() * 1000.0
    ));

    let t_save = Instant::now();
    save_image(&args.output, &restored, args.alpha.as_ref())
        .with_context(|| format!("failed to save image: {}", args.output.display()))?;
    marker(&format!("saved ({:.0}ms)", t_save.elapsed().as_secs_f32() * 1000.0));
    marker(&format!(
        "total ({:.0}ms)",
        t_total.elapsed().as_secs_f32() * 1000.0
    ));

    Ok(())
}

/// Read a PNG into an `(H, W, 3)` f32 array in `[0, 1]`. Alpha is
/// dropped at this point (NAFNet only takes 3 channels); the alpha
/// channel is preserved separately via the `--alpha` argument if
/// the caller passes one.
fn load_image(path: &PathBuf) -> Result<Array3<f32>> {
    let img = image::open(path)
        .with_context(|| format!("image::open failed for {}", path.display()))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut arr = Array3::<f32>::zeros((h, w, 3));
    arr.axis_iter_mut(ndarray::Axis(0))
        .enumerate()
        .par_bridge()
        .for_each(|(y, mut row)| {
            let y_u32 = y as u32;
            for x in 0..w {
                let p = img.get_pixel(x as u32, y_u32);
                row[[x, 0]] = p[0] as f32 / 255.0;
                row[[x, 1]] = p[1] as f32 / 255.0;
                row[[x, 2]] = p[2] as f32 / 255.0;
            }
        });
    Ok(arr)
}

/// Write an `(H, W, 3)` f32 array in `[0, 1]` to a PNG. If
/// `alpha_path` is provided, read the Y u8 alpha from it and write
/// RGBA; otherwise write plain RGB.
fn save_image(path: &PathBuf, arr: &Array3<f32>, alpha_path: Option<&PathBuf>) -> Result<()> {
    let (h, w) = (arr.dim().0, arr.dim().1);

    if let Some(alpha_path) = alpha_path {
        // Combine the RGB model output with the original alpha
        // (read from a Y u8 grayscale PNG) and save as RGBA.
        let alpha_img = image::open(alpha_path)
            .with_context(|| {
                format!(
                    "alpha file open failed for {}",
                    alpha_path.display()
                )
            })?
            .to_luma8();
        if alpha_img.width() as usize != w || alpha_img.height() as usize != h {
            return Err(anyhow!(
                "alpha file dimensions ({}x{}) don't match image dimensions ({}x{})",
                alpha_img.width(),
                alpha_img.height(),
                w,
                h
            ));
        }

        let mut out = image::RgbaImage::new(w as u32, h as u32);
        out.as_mut()
            .par_chunks_mut(w * 4)
            .enumerate()
            .for_each(|(y, row_buf)| {
                for x in 0..w {
                    let r = quantize(arr[[y, x, 0]]);
                    let g = quantize(arr[[y, x, 1]]);
                    let b = quantize(arr[[y, x, 2]]);
                    let a = alpha_img.get_pixel(x as u32, y as u32).0[0];
                    let p = x * 4;
                    row_buf[p] = r;
                    row_buf[p + 1] = g;
                    row_buf[p + 2] = b;
                    row_buf[p + 3] = a;
                }
            });
        out.save(path)
            .with_context(|| format!("image save failed for {}", path.display()))?;
    } else {
        // Plain RGB output (alpha is dropped).
        let mut out = image::RgbImage::new(w as u32, h as u32);
        out.as_mut()
            .par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, row_buf)| {
                for x in 0..w {
                    let r = quantize(arr[[y, x, 0]]);
                    let g = quantize(arr[[y, x, 1]]);
                    let b = quantize(arr[[y, x, 2]]);
                    let p = x * 3;
                    row_buf[p] = r;
                    row_buf[p + 1] = g;
                    row_buf[p + 2] = b;
                }
            });
        out.save(path)
            .with_context(|| format!("image save failed for {}", path.display()))?;
    }
    Ok(())
}

fn quantize(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn build_session(model_path: &PathBuf) -> Result<(Session, &'static str)> {
    let session = Session::builder()
        .map_err(|e| anyhow!("failed to create ORT session builder: {}", e))?
        .with_dimension_override("batch", 1)
        .map_err(|e| anyhow!("failed to override batch dimension: {}", e))?
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .map_err(|e| anyhow!("failed to set graph optimization level: {}", e))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("ORT session build failed: {}", e))?;
    // ort 2.0.0-rc.12 has no Session::providers() method; the build-time
    // cargo feature is the only honest signal we have about which EPs
    // are statically linked into the binary.
    #[cfg(feature = "webgpu")]
    let provider_name: &'static str = "WebGPU";
    #[cfg(not(feature = "webgpu"))]
    let provider_name: &'static str = "CPU";
    Ok((session, provider_name))
}

/// Run NAFNet on the input, tiling if larger than `tile_size`.
///
/// When the image is smaller than or equal to a tile, this is a
/// single forward pass with no blending. When larger, the image is
/// processed as overlapping tiles; in the overlap region, predictions
/// are blended with a 2D tent window so seams are not visible.
fn run_inference(
    session: &mut Session,
    img: &Array3<f32>,
    tile_size: usize,
    tile_overlap: usize,
) -> Result<Array3<f32>> {
    let (h, w) = (img.dim().0, img.dim().1);
    if h <= tile_size && w <= tile_size {
        return run_single_inference(session, img);
    }
    tiled_inference(session, img, tile_size, tile_overlap)
}

fn tiled_inference(
    session: &mut Session,
    img: &Array3<f32>,
    tile_size: usize,
    overlap: usize,
) -> Result<Array3<f32>> {
    let (h, w, c) = (img.dim().0, img.dim().1, img.dim().2);
    let stride = tile_size - overlap;

    let mut output = Array3::<f32>::zeros((h, w, c));
    let mut weight = Array2::<f32>::zeros((h, w));

    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            let y_end = (y + tile_size).min(h);
            let x_end = (x + tile_size).min(w);
            let tile_h = y_end - y;
            let tile_w = x_end - x;

            // Extract tile into a contiguous Array3. ndarray's slice +
            // to_owned does a copy, so the input into ORT is owned.
            let tile_in = img.slice(s![y..y_end, x..x_end, ..]).to_owned();
            let tile_out = run_single_inference(session, &tile_in)?;

            // Pre-compute the blend window. Trimming to the actual
            // tile size at the right/bottom edge keeps the weight
            // distribution correct when the image isn't a multiple
            // of tile_size.
            let full_window = make_blend_window(tile_size, tile_size, overlap);
            let window = full_window.slice(s![..tile_h, ..tile_w]);

            for ty in 0..tile_h {
                for tx in 0..tile_w {
                    let wgt = window[[ty, tx]];
                    for ch in 0..c {
                        output[[y + ty, x + tx, ch]] += tile_out[[ty, tx, ch]] * wgt;
                    }
                    weight[[y + ty, x + tx]] += wgt;
                }
            }

            x += stride;
        }
        y += stride;
    }

    // Normalize: each pixel's output is the weight-averaged sum of
    // tile predictions. `weight[[y, x]]` is guaranteed > 0 because
    // every pixel is covered by at least one tile.
    for y in 0..h {
        for x in 0..w {
            let wgt = weight[[y, x]];
            debug_assert!(wgt > 0.0, "pixel ({}, {}) not covered by any tile", y, x);
            for ch in 0..c {
                output[[y, x, ch]] /= wgt;
            }
        }
    }

    Ok(output)
}

/// 2D tent window for tile blending. Center is 1; edges ramp to 0
/// over the overlap region. The top and bottom ramps never overlap
/// (clamped to `min(overlap, tile_h/2)`), so the center stays at 1.
///
/// The window is `tile_size x tile_size`; for partial tiles (right or
/// bottom edge of the image) we trim before accumulating.
fn make_blend_window(tile_h: usize, tile_w: usize, overlap: usize) -> Array2<f32> {
    let mut w = Array2::<f32>::ones((tile_h, tile_w));
    let top_ramp = overlap.min(tile_h / 2);
    let bot_ramp = overlap.min(tile_h / 2);
    let left_ramp = overlap.min(tile_w / 2);
    let right_ramp = overlap.min(tile_w / 2);

    for y in 0..top_ramp {
        let r = (y + 1) as f32 / (overlap + 1) as f32;
        for x in 0..tile_w {
            w[[y, x]] *= r;
        }
    }
    for y in (tile_h - bot_ramp)..tile_h {
        let r = (tile_h - y) as f32 / (overlap + 1) as f32;
        for x in 0..tile_w {
            w[[y, x]] *= r;
        }
    }
    for x in 0..left_ramp {
        let r = (x + 1) as f32 / (overlap + 1) as f32;
        for y in 0..tile_h {
            w[[y, x]] *= r;
        }
    }
    for x in (tile_w - right_ramp)..tile_w {
        let r = (tile_w - x) as f32 / (overlap + 1) as f32;
        for y in 0..tile_h {
            w[[y, x]] *= r;
        }
    }
    w
}

/// Run NAFNet on a single tile. Input `(H, W, C)` -> convert to
/// `(1, C, H, W)` -> run -> convert back to `(H, W, C)`, clipped to
/// `[0, 1]` (NAFNet outputs pixel values in the same range).
fn run_single_inference(session: &mut Session, img: &Array3<f32>) -> Result<Array3<f32>> {
    let (h, w, c) = (img.dim().0, img.dim().1, img.dim().2);
    if h == 0 || w == 0 {
        return Err(anyhow!("empty image: {}x{}", w, h));
    }
    let mut input = Array4::<f32>::zeros((1, c, h, w));
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                input[[0, ch, y, x]] = img[[y, x, ch]];
            }
        }
    }

    let input_name: String = session
        .inputs()
        .first()
        .map(|i| i.name().to_string())
        .ok_or_else(|| anyhow!("ORT session has no inputs"))?;
    let output_name: String = session
        .outputs()
        .first()
        .map(|o| o.name().to_string())
        .ok_or_else(|| anyhow!("ORT session has no outputs"))?;

    let outputs = session
        .run(inputs![
            input_name.as_str() => TensorRef::from_array_view(input.view())?,
        ])
        .map_err(|e| anyhow!("ORT run failed: {}", e))?;

    let out_view = outputs
        .get(output_name.as_str())
        .ok_or_else(|| anyhow!("ORT output `{}` missing", output_name))?
        .try_extract_array::<f32>()
        .map_err(|e| anyhow!("ORT output extraction failed: {}", e))?;
    let out_4d = out_view.to_owned();
    drop(outputs);

    if out_4d.shape()[0] != 1 || out_4d.shape()[1] != c {
        return Err(anyhow!(
            "unexpected ORT output shape {:?}; expected (1, {}, H, W)",
            out_4d.shape(),
            c
        ));
    }
    let (oc, oh, ow) = (out_4d.shape()[1], out_4d.shape()[2], out_4d.shape()[3]);
    if oh != h || ow != w {
        return Err(anyhow!(
            "NAFNet output is not 1:1: input {}x{}, output {}x{}",
            w, h, ow, oh
        ));
    }

    let mut out_hwc = Array3::<f32>::zeros((oh, ow, oc));
    out_hwc
        .axis_iter_mut(ndarray::Axis(0))
        .enumerate()
        .par_bridge()
        .for_each(|(y, mut row)| {
            for x in 0..ow {
                for ch in 0..oc {
                    let v = out_4d[[0, ch, y, x]];
                    row[[x, ch]] = if v < 0.0 {
                        0.0
                    } else if v > 1.0 {
                        1.0
                    } else {
                        v
                    };
                }
            }
        });
    Ok(out_hwc)
}

fn marker(stage: &str) {
    eprintln!("[LAMA_MARKER] phase {}", stage);
    let _ = std::io::stderr().flush();
}
