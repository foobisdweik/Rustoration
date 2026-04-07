# rusty-cunny — Image Restoration Pipeline

Fast, memory-efficient image restoration tool for removing pixelation artifacts and low-fidelity regions from digital illustrations.

## Quick Start

### Prerequisites
- Windows 11 x86_64
- Rust 1.70+ (with CUDA 13.2 support)
- NVIDIA GPU with CUDA Compute Capability ≥ 6.1 (tested on RTX 3070 Mobile)
- .NET 9.0 runtime (for GUI)

### Build

```bash
# Ensure CUDA 13.2 is installed and in PATH
cargo build --release
```

**Binary location:** `target\release\restore.exe`

### Usage (Standalone)

```bash
# CLI mode (folder watcher)
./target/release/restore.exe

# Watch: ./input/*.{png,jpg,jpeg} → Process → Save to ./output/
```

**Input structure:**
```
./input/
  ├── image1.png
  ├── image2.jpg
  └── ...
```

**Output:**
```
./output/
  ├── image1.restored.png
  ├── image2.restored.jpg
  └── ...
```

### Usage (GUI)

```bash
cd gui/RustyCunnyGui
dotnet publish -c Release
# Run: RustyCunnyGui.exe
```

Select input/output directories and restoration prompt, click "Start Restoration".

---

## Model Setup

### 1. Download Checkpoints

Download the following models:
- **YOLOv8-seg:** `yolov8l-seg.pt` from [Ultralytics](https://github.com/ultralytics/yolov8)
- **SDXL 1.0:** `sd-xl-1.0-unet.pt` + `sd-xl-1.0-vae.pt` from [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

### 2. Convert to SafeTensors

```bash
python tools/convert_yolo_to_safetensors.py yolov8l-seg.pt -o yolov8l-seg.safetensors
python tools/convert_yolo_to_safetensors.py sd-xl-1.0-unet.pt -o sd-xl-1.0-unet.safetensors
python tools/convert_yolo_to_safetensors.py sd-xl-1.0-vae.pt -o sd-xl-1.0-vae.safetensors
```

### 3. Place Models

```
./models/
├── yolov8l-seg.safetensors
├── sd-xl-1.0-unet.safetensors
└── sd-xl-1.0-vae.safetensors
```

---

## Configuration

Edit `.cargo/config.toml` for CUDA version:
```toml
[env]
CUDARC_CUDA_VERSION = { value = "13020", relative = false, force = true }
```

**Format:** `{major}0{minor}0` (e.g., `13020` for CUDA 13.2)

---

## Performance

**Hardware:** RTX 3070 Mobile (8GB VRAM)
- **Detection:** ~50–100ms per image
- **Inpainting:** ~10–20s per image (20 DDIM steps)
- **Total Throughput:** ~3–5 images/min

**Memory Usage:**
- Model loading: ~5GB
- Encoder dropout before inference: Reclaims ~1.4GB
- Peak: ~6.5GB (manages within 8GB budget)

---

## Architecture

### Detection (`src/detection.rs`)
- YOLOv8-seg backbone
- Identifies pixelation artifacts via class confidence + mask
- Outputs bounding boxes + segmentation masks

### Masking (`src/mask.rs`)
- Proto coefficient matmul decoding
- Sigmoid thresholding + morphological dilation
- Expands artifact regions for smooth inpainting

### Inpainting (`src/inpaint.rs`)
- SDXL 1.0 latent diffusion model
- CLIP-L + OpenCLIP-G text encoder (both dropped before DDIM)
- 20-step DDIM scheduler

### Pipeline (`src/pipeline.rs`)
- End-to-end orchestrator
- Async model loading
- Memory-aware inference

---

## Development

### Project Layout
```
src/
  main.rs        → Async folder watcher
  lib.rs         → Module exports
  detection.rs   → YoloDetector
  mask.rs        → MaskGenerator
  inpaint.rs     → SdxlInpaint + DDIM
  pipeline.rs    → RestorationPipeline

gui/RustyCunnyGui/
  MainWindow.xaml       → Layout
  MainWindow.xaml.cs    → Logic (subprocess launcher)
  Program.cs            → Entry point
  RustyCunnyGui.csproj  → .NET project

tools/
  convert_yolo_to_safetensors.py  → Model format conversion

.cargo/config.toml               → CUDA configuration
Cargo.toml                       → Dependencies
```

### Building from Source

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Check for issues
cargo clippy

# Format
cargo fmt
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA not found | Ensure CUDA 13.2 is installed; check `nvcc --version` |
| OOM errors | Reduce `num_inference_steps` in `src/inpaint.rs` (default: 20) |
| Slow inference | Enable release mode: `cargo build --release` |
| Model not loading | Verify SafeTensors format: `python -c "from safetensors.torch import load_file; load_file('model.safetensors')"` |

---

## License

MIT / Apache 2.0

## Contact

For issues, open an issue on GitHub or contact the maintainer.
