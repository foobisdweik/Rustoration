
# GEMINI.md — High-Level Project Overview

**Project:** rusty-cunny  
**Summary:** Native Rust image restoration utility combining YOLOv8-segmentation for artifact detection with SDXL inpainting for high-fidelity reconstruction of stylized 2D raster images/art.

## What It Does

Automates detection and reconstruction of pixelated/low-fidelity distorted regions in digital illustrations using machine learning pipelines.

**Input:** Image files (PNG/JPG/JPEG/WEBP) + optional restoration prompt  
**Output:** High-fidelity reconstructed images with artifacts replaced

## Architecture

```
Folder Watcher → YOLOv8-seg Detection → Mask Generation → SDXL Inpainting → Save
                 (Identify artifacts)   (Threshold+dilation) (DDIM diffusion)
```

### Tech Stack

- **ML Framework:** Candle (Rust native, CUDA 13.2)
- **Detection:** YOLOv8-segmentation (SafeTensors format)
- **Inpainting:** SDXL 1.0 (dual CLIP-L + OpenCLIP-G encoders)
- **Async Runtime:** Tokio (folder watching, pipeline orchestration)
- **GUI:** Avalonia 11.2.3 (.NET 9.0) — launches binary as subprocess
- **Hardware Target:** i7-12700H (6 P-cores & 8 E-cores for 20 total threads) + RTX 3070 Mobile (8GB VRAM) + 32GB DDR5 System Memory + CUDA 13.2

## Key Design Patterns

1. **Memory Efficiency:** Explicit encoder dropout before DDIM loop reclaims ~1.4GB VRAM.
2. **Async I/O:** Tokio-based folder watcher with 2sec polling.
3. **Zero-Dependency Core:** Static linking of MSVC, self-contained .NET, and local CUDA DLLs for "AppImage" behavior.
4. **VRAM Handoff:** Explicit `drop()` statements for predictable memory release.
5. **Model Format:** SafeTensors for deterministic model loading without Python dependencies.

## File Structure

- `src/` — Core Rust pipeline (detection, masking, inpainting, orchestration).
- `gui/` — Avalonia GUI (.NET 9.0) with subprocess launcher.
- `tools/` — Python utility for model weight conversion (.pt → SafeTensors).
- `.cargo/` — Build configurations + Static CRT flags.

## Deployment & Maximum Portability

To achieve true "AppImage-like" drop-and-run portability on Windows 11, the build pipeline eliminates external dependencies by statically linking system runtimes and isolating the CUDA environment.
																		
### 1. Rust Backend (Static MSVC Linking)

Eliminates the requirement for the user to install the Microsoft Visual C++ Redistributable.
* **Configuration (`.cargo/config.toml`):**
  ```toml
  [target.x86_64-pc-windows-msvc]
  rustflags = ["-C", "target-feature=+crt-static"]
  ```
* **Build Command:** `cargo build --release --target x86_64-pc-windows-msvc`

### 2. Avalonia GUI (Self-Contained .NET Single File)

Eliminates the requirement for a pre-installed .NET 9.0 Desktop Runtime. The framework and native libraries are bundled into one executable.
* **Publish Command:**
  ```bash
  dotnet publish -c Release -r win-x64 --self-contained true /p:PublishSingleFile=true /p:IncludeNativeLibrariesForSelfContained=true
  ```

### 3. CUDA Isolation (Local DLL Bundling)

To prevent runtime crashes on systems without the full CUDA Toolkit, package the necessary runtime DLLs in the same directory as `restore.exe`. Windows prioritizes local directory DLLs over system-wide ones.
* **Essential DLLs to include:**
  - `cublas64_12.dll`
  - `cublasLt64_12.dll`
  - `cudart64_12.dll`
  - `cudnn64_8.dll` (if applicable)

## Status

**v0.1.0:** Scaffolded and fully integrated.  
**Ready for:** Model weight integration + VRAM benchmarking + production hardening.

## Integration Notes

- Requires downloaded YOLOv8-seg-l.pt + SDXL 1.0 checkpoints (convert via `tools/convert_yolo_to_safetensors.py`).
- Standalone binary `restore.exe` can be called from any application via subprocess.
- GUI is optional but highly-recommended; binary also works headless (logs to stdout).