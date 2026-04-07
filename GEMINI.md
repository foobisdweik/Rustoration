# GEMINI.md — High-Level Project Overview

**Project:** rusty-cunny  
**Summary:** Native Rust image restoration utility combining YOLOv8-segmentation for artifact detection with SDXL inpainting for high-fidelity reconstruction.

## What It Does

Automates detection and reconstruction of pixelated/low-fidelity distorted regions in digital illustrations using machine learning pipelines.

**Input:** Image files (PNG/JPG) + optional restoration prompt  
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
- **Hardware:** RTX 3070 Mobile (8GB VRAM) + CUDA 13.2

## Key Design Patterns

1. **Memory Efficiency:** Explicit encoder dropout before DDIM loop reclaims ~1.4GB VRAM
2. **Async I/O:** Tokio-based folder watcher with 2sec polling (no busy loops)
3. **Type Safety:** Pure Rust with candle-transformers (no Python runtime required)
4. **VRAM Handoff:** Explicit `drop()` statements for predictable memory release
5. **Model Format:** SafeTensors for deterministic model loading (ONNX alternative)

## File Structure

- `src/` — Core Rust pipeline (detection, masking, inpainting, orchestration)
- `gui/` — Avalonia GUI (.NET 9.0) with subprocess launcher
- `tools/` — Python utility for model weight conversion (.pt → SafeTensors)
- `.cargo/` — CUDA 13.2 build configuration

## Status

**v0.1.0:** Scaffolded and fully integrated  
**Ready for:** Model weight integration + VRAM benchmarking + production hardening

## Integration Notes

- Requires downloaded YOLOv8-seg-l.pt + SDXL 1.0 checkpoints (convert via `tools/convert_yolo_to_safetensors.py`)
- Standalone binary `restore.exe` can be called from any application via subprocess
- GUI is optional; binary also works headless (logs to stdout)
