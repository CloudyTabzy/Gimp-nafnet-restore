---
license: mit
pipeline_tag: image-to-image
tags:
- image-restoration
- image-enhancement
- computer-vision
- onnx
- nafnet
- scunet
library_name: dghs-imgutils
---

# Image Restoration Models

## Summary

This repository provides high-performance **image restoration** models including **NAFNet** and **SCUNet** implementations for various image enhancement tasks. The models are designed to restore degraded images by removing noise, artifacts, and other imperfections while preserving important visual details. These **deep learning** models leverage advanced neural network architectures optimized for computational efficiency and restoration quality.

The repository contains multiple model variants trained on different datasets: **NAFNet-REDS** for general image restoration, **NAFNet-GoPro** for deblurring tasks, and **NAFNet-SIDD** for noise reduction. Additionally, **SCUNet** provides both GAN-based and PSNR-optimized approaches for comprehensive image enhancement. All models are exported in **ONNX format** for efficient inference and cross-platform compatibility.

These restoration models support **tiled processing** to handle high-resolution images efficiently through memory-optimized batch processing with configurable tile sizes and overlaps. The implementation includes support for images with alpha channels (RGBA format) and provides flexible parameter tuning for different use cases. The models achieve state-of-the-art performance in benchmark evaluations while maintaining practical inference speeds suitable for production environments.

## Usage

### Installation

```bash
pip install dghs-imgutils
```

### Basic Usage

```python
from imgutils.restore import restore_with_nafnet, restore_with_scunet
from PIL import Image

# Load your degraded image
image = Image.open('degraded_image.jpg')

# Restore using NAFNet (recommended for general restoration)
restored_nafnet = restore_with_nafnet(image, model='REDS')

# Restore using SCUNet (alternative approach)
restored_scunet = restore_with_scunet(image, model='GAN')

# Save restored images
restored_nafnet.save('restored_nafnet.jpg')
restored_scunet.save('restored_scunet.jpg')
```

### Advanced Usage with Custom Parameters

```python
from imgutils.restore import restore_with_nafnet

# Custom processing parameters for high-resolution images
restored_image = restore_with_nafnet(
    image, 
    model='SIDD',  # Choose from 'REDS', 'GoPro', 'SIDD'
    tile_size=512,  # Larger tiles for better performance
    tile_overlap=32,  # More overlap for seamless results
    batch_size=8,  # Higher batch size for faster processing
    silent=False  # Show progress bar
)
```

## Available Models

### NAFNet Models
- **NAFNet-REDS**: General image restoration trained on REDS dataset
- **NAFNet-GoPro**: Specialized for deblurring tasks
- **NAFNet-SIDD**: Optimized for noise reduction

### SCUNet Models
- **SCUNet-GAN**: GAN-based restoration for perceptual quality
- **SCUNet-PSNR**: PSNR-optimized restoration for objective metrics

## Model Benchmarks

The models have been extensively benchmarked for various restoration tasks:

- **NAFNet** demonstrates superior performance in deblurring and general restoration
- **SCUNet** excels in noise reduction and artifact removal
- Both models support high-resolution processing through tiled inference

## Important Notes

- **Alpha Channel Support**: All models support RGBA images (with alpha channel)
- **Gaussian Noise Warning**: NAFNet may have issues with Gaussian noise - consider preprocessing with SCUNet if needed
- **Memory Optimization**: Use appropriate tile_size and batch_size parameters for your hardware

## Original Content

### Model Implementation Details

The implementation uses the `dghs-imgutils` library which provides:

- Efficient ONNX model loading and inference
- Tiled processing for memory-efficient high-resolution image handling
- Batch processing optimization for improved performance
- Support for various image formats and color spaces

### Technical Architecture

The models leverage:

- **NAFNet**: Non-linear Activation Free Network for efficient image restoration
- **SCUNet**: Swin-Conv-UNet architecture for comprehensive image enhancement
- **ONNX Runtime**: Cross-platform inference engine for optimal performance

## Citation

```bibtex
@misc{deepghs_image_restoration,
  title        = {{Image Restoration Models: NAFNet and SCUNet Implementations}},
  author       = {deepghs},
  howpublished = {\url{https://huggingface.co/deepghs/image_restoration}},
  year         = {2023},
  note         = {High-performance image restoration models including NAFNet and SCUNet implementations for noise reduction, deblurring, and general image enhancement},
  abstract     = {This repository provides high-performance image restoration models including NAFNet and SCUNet implementations for various image enhancement tasks. The models are designed to restore degraded images by removing noise, artifacts, and other imperfections while preserving important visual details. These deep learning models leverage advanced neural network architectures optimized for computational efficiency and restoration quality. The repository contains multiple model variants trained on different datasets: NAFNet-REDS for general image restoration, NAFNet-GoPro for deblurring tasks, and NAFNet-SIDD for noise reduction. Additionally, SCUNet provides both GAN-based and PSNR-optimized approaches for comprehensive image enhancement.},
  keywords     = {image-restoration, image-enhancement, computer-vision, nafnet, scunet}
}
```
