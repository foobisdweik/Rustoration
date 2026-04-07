
# CLAUDE.md — High-Level Project Overview

**Project Name:** rusty-cunny  
**Purpose:** High-performance image restoration utility in Rust using Candle ML framework  
**Status:** Scaffolded and iterated (v0.1.0)  
**Target Hardware:** i7-12700H (20 threads) + RTX 3070 Mobile (8GB VRAM) + 32GB DDR5  
**Environment:** Windows 11 x86_64, VS2026 (MSVC 14.29.30133), CUDA 13.2

## Core Subsystems

**1. Detection Pipeline (`src/detection.rs`)**

- YOLOv8-seg backbone with PAN neck and detection head.
- Parameterized channel slices: `boxes=0..64`, `classes=64..64+nc`, `mask=64+nc..64+nc+32`.
- Loads SafeTensors (ultralytics .pt unsupported natively).
- Type: `YoloDetector` with async model loading.

**2. Mask Generation (`src/mask.rs`)**

- Proto matmul decode: `mask = mask_coeffs @ proto.T`.
- Sigmoid + threshold confidence filtering.
- Separable O(r) dilation for artifact expansion.
- Type: `MaskGenerator` with configurable threshold & iterations.

**3. Inpainting Pipeline (`src/inpaint.rs`)**

- DDIM scheduler (configurable timesteps, default 20).
- 5-channel `InpaintUnet` wrapper for latent diffusion (SDXL architecture).
- Dual encoders: CLIP-L + OpenCLIP-G.
- **Critical:** Encoders dropped before DDIM loop → reclaims ~1.4GB VRAM
- Type: `SdxlInpaint` with explicit memory management

**4. Async Folder Watcher (`src/main.rs`)**

- Tokio-based event loop with 2sec polling interval.
- Explicit `drop()`-based VRAM handoff state machine.
- Processes PNG/JPG/JPEG/WEBP, moves to `processed/` subdir.
- Type: Binary `restore` (entry point).

**5. GUI Layer (`gui/RustyCunnyGui/`)**

- **Framework:** Avalonia 11.2.3 (.NET 9.0 win-x64).
- **MVVM:** CommunityToolkit.Mvvm for data binding.
- **Execution:** Launches `restore.exe` subprocess; pipes stdout/stderr to scrolling log panel.
- **Pattern:** FluentTheme + AvaloniaXamlLoader.

## Deployment & Portability Strategy

To achieve "AppImage-like" portability on Windows 11, the project isolates all dependencies:

### 1. Static Rust Backend

Eliminates requirement for MSVC Redistributable.
- **Flag:** `target-feature=+crt-static` via `.cargo/config.toml`.

### 2. Self-Contained GUI

Bundles .NET 9.0 runtime into a single executable.
- **Method:** `dotnet publish` with `PublishSingleFile=true` and `IncludeNativeLibrariesForSelfContained=true`.

### 3. CUDA Isolation

Local bundling of runtime DLLs in the application directory to bypass system-wide toolkit requirements.
- **Required DLLs:** `cublas64_12.dll`, `cublasLt64_12.dll`, `cudart64_12.dll`, `cudnn64_8.dll`.

## Build Configuration

### .cargo/config.toml

toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=+crt-static", "-C", "target-cpu=native"]

[env]
CUDARC_CUDA_VERSION = { value = "13020", force = true }

### Cargo.toml
- **Key Deps:** `candle-core/nn/transformers` (0.3 + cuda), `tokio` (full), `image`, `ndarray`.
- **Profile:** `release` with `lto = true` and `codegen-units = 1`.

## Key Learnings & Patterns

1. **VRAM Handoff:** Use explicit `drop()` calls between pipeline stages to ensure predictable memory release on 8GB VRAM hardware.
2. **ANSI Stripping:** Cargo/Binary output requires escape code removal before the GUI can parse log patterns.
3. **Registry Patching:** `~/.cargo/git` files require patching for CUDA 13.x CCCL headers; requires target/lock cleaning post-patch.
4. **Zero-Dependency Core:** Prioritize static linking and local DLL prioritization to ensure "drop-and-run" behavior.
5. **Deterministic Loading:** SafeTensors is the mandatory format; use `tools/convert_yolo_to_safetensors.py` for all weights.

## Directory Structure

C:\Users\Foobis\Documents\Source\rusty-cunny\
├── .cargo/
│   └── config.toml                    [Static CRT + CUDA 13.2 config]
├── src/
│   ├── main.rs                        [Watcher + VRAM state machine]
│   ├── detection.rs                   [YOLOv8-seg logic]
│   ├── mask.rs                        [Mask decoding + dilation]
│   ├── inpaint.rs                     [SDXL + DDIM + encoder drop]
│   └── pipeline.rs                    [Orchestrator service]
├── gui/
│   └── RustyCunnyGui/                 [.NET 9.0 Avalonia GUI project]
├── tools/
│   └── convert_yolo_to_safetensors.py [Weight conversion utility]
├── Cargo.toml                         [Candle + CUDA dependencies]
├── CLAUDE.md                          [Technical project state]
└── GEMINI.md                          [High-level overview & deployment guide]

## Next Steps

1. **Weight Integration:** Convert YOLOv8-seg-l.pt and SDXL 1.0 checkpoints to SafeTensors.
2. **Portability Validation:** Verify static linking and local CUDA DLL priority on a clean Windows 11 environment.
3. **VRAM Benchmarking:** Profile peak memory usage during the DDIM loop to optimize encoder drop timing.
4. **GUI Hardening:** Finalize subprocess stdio piping and error handling for model-missing states.
