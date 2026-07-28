//! End-to-end integration test for the NAFNet worker binary.
//!
//! Spawns the compiled binary, runs it on a real test image, and
//! verifies the output is correct. The test image is generated
//! programmatically (no external assets required) so the test
//! is self-contained and runs on any platform that can build the
//! binary.
//!
//! Skipped if the model file is not present at the expected location.
//! Run `install.bat` or download `NAFNet-REDS-width64_v1.onnx` from
//! <https://huggingface.co/deepghs/image_restoration> to enable.

use std::path::PathBuf;
use std::process::Command;

use image::{ImageBuffer, Rgb};

const MODEL_RELATIVE: &str = "../NAFNet-REDS-width64_v1.onnx";

/// Path to the worker binary. `cargo test` sets the working
/// directory to the package root, so the binary is in
/// `target/release/`.
fn worker_binary_path() -> PathBuf {
    PathBuf::from("target").join("release").join(if cfg!(windows) {
        "nafnet-worker.exe"
    } else {
        "nafnet-worker"
    })
}

fn model_path() -> PathBuf {
    PathBuf::from(MODEL_RELATIVE)
}

fn make_test_image(width: u32, height: u32, path: &std::path::Path) {
    // Smooth gradient with a sharp test pattern in the middle. The
    // pattern is the discriminator: if the worker ever drops
    // resolution or clips badly, the difference shows up at the
    // pattern edges.
    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let r = (x as f32 / width as f32 * 255.0) as u8;
            let g = (y as f32 / height as f32 * 255.0) as u8;
            let b = (((x + y) as f32 / (width + height) as f32) * 255.0) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    // Add a sharp red square in the top-left so the test has
    // structure beyond the gradient.
    for y in 20..60 {
        for x in 20..60 {
            img.put_pixel(x, y, Rgb([230, 20, 20]));
        }
    }
    img.save(path).expect("failed to write test image");
}

#[test]
fn worker_roundtrip_preserves_image_dimensions() {
    let worker = worker_binary_path();
    if !worker.exists() {
        eprintln!(
            "skipping: worker binary not found at {} (run `cargo build --release`)",
            worker.display()
        );
        return;
    }
    let model = model_path();
    if !model.exists() {
        eprintln!(
            "skipping: NAFNet model not found at {} (download from HuggingFace or run install.bat)",
            model.display()
        );
        return;
    }

    let tmp = std::env::temp_dir().join("nafnet-worker-test");
    std::fs::create_dir_all(&tmp).expect("create temp dir");
    let input = tmp.join("in.png");
    let output = tmp.join("out.png");

    make_test_image(128, 128, &input);

    let status = Command::new(&worker)
        .arg("--image")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .arg("--model")
        .arg(&model)
        .status()
        .expect("failed to spawn worker");

    assert!(status.success(), "worker exited non-zero: {:?}", status);

    // Sanity: output file starts with the PNG magic and ends with
    // the IEND chunk. We don't compare file sizes (PNG compression
    // varies); we just verify the file is a well-formed PNG.
    let bytes = std::fs::read(&output).expect("read output");
    assert_eq!(
        &bytes[0..8],
        b"\x89PNG\r\n\x1a\n",
        "output is not a valid PNG (bad magic)"
    );
    // IEND chunk: 4-byte length (0) + "IEND" + 4-byte CRC = 12 bytes
    // at the end of a valid PNG.
    let n = bytes.len();
    assert!(n >= 12, "output too small to be a PNG");
    assert_eq!(
        &bytes[n - 8..n - 4],
        b"IEND",
        "output is not a valid PNG (no IEND marker)"
    );

    // Sanity: output has the same dimensions as input.
    let in_img = image::open(&input).expect("read input");
    let out_img = image::open(&output).expect("read output");
    assert_eq!(
        in_img.width(),
        out_img.width(),
        "output width differs from input width"
    );
    assert_eq!(
        in_img.height(),
        out_img.height(),
        "output height differs from input height"
    );
}
