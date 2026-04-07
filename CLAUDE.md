# CLAUDE.md — rusty-cunny Project State

**Project Name:** rusty-cunny  
**Purpose:** High-performance image restoration utility in Rust using Candle ML framework  
**Status:** Scaffolded and iterated (v0.1.0)  
**Target Hardware:** RTX 3070 Mobile (8GB VRAM) + CUDA 13.2  
**Environment:** Windows 11 x86_64, VS2026 (MSVC 14.29.30133)

---

## TOON v3.0 Project Facts

### Core Subsystems

**1. Detection Pipeline (`src/detection.rs`)**
- YOLOv8-seg backbone with PAN neck and detection head
- Parameterized channel slices: `boxes=0..64`, `classes=64..64+nc`, `mask=64+nc..64+nc+32`
- Loads SafeTensors (ultralytics .pt unsupported natively)
- Type: `YoloDetector` with async model loading

**2. Mask Generation (`src/mask.rs`)**
- Proto matmul decode: `mask = mask_coeffs @ proto.T`
- Sigmoid + threshold confidence filtering
- Separable O(r) dilation for artifact expansion
- Type: `MaskGenerator` with configurable threshold & iterations

**3. Inpainting Pipeline (`src/inpaint.rs`)**
- DDIM scheduler (configurable timesteps, default 20)
- 5-channel `InpaintUnet` wrapper for latent diffusion
- SDXL architecture: CLIP-L + OpenCLIP-G dual encoders
- **Critical:** Encoders dropped before DDIM loop → reclaims ~1.4GB VRAM
- Type: `SdxlInpaint` with explicit memory management

**4. Async Folder Watcher (`src/main.rs`)**
- Tokio-based event loop with 2sec polling interval
- Explicit `drop()`-based VRAM handoff state machine
- Processes PNG/JPG, moves to `processed/` subdir
- Type: Binary `restore` (entry point)

**5. End-to-End Pipeline (`src/pipeline.rs`)**
- `RestorationPipeline` orchestrator
- Loads models asynchronously, chains detection→mask→inpaint
- Type: Singleton async service

### GUI Layer (`gui/RustyCunnyGui/`)
- **Framework:** Avalonia 11.2.3 (.NET 9.0 win-x64)
- **MVVM:** CommunityToolkit.Mvvm for data binding
- **Pattern:** Launches `restore.exe` subprocess, pipes stdout/stderr to scrolling log panel
- **XAML:** MainWindow with input/output directory selection + prompt textbox
- **Entry:** Program.cs with FluentTheme + AvaloniaXamlLoader

### Model Conversion (`tools/convert_yolo_to_safetensors.py`)
- Ultralytics `.pt` → SafeTensors conversion
- Verifies tensors post-conversion
- CLI: `python convert_yolo_to_safetensors.py model.pt -o model.safetensors`

---

## Build Configuration

### Cargo.toml
- **Dependencies:** candle-core/nn/transformers (0.3 + cuda feature), tokio (full), image, ndarray
- **Profile:** release with LTO + codegen-units=1
- **Bin:** `restore` → `src/main.rs`

### .cargo/config.toml
- **CUDARC_CUDA_VERSION:** `13020` (encoded int format, force=true)
- **Rustflags:** `-C target-cpu=native`

---

## Key Learnings & Patterns

1. **ANSI Stripping:** Cargo output requires escape code removal before pattern matching
2. **Registry Patching:** `~/.cargo/git` files patched post-clone for CUDA 13.x CCCL headers
3. **Cache Invalidation:** After patching, must clean both `target\debug\build\*` AND `Cargo.lock`
4. **VRAM Strategy:** Explicit encoder drop before inference loop; drop() calls for immediate reclamation
5. **Model Format:** SafeTensors required; .pt export unsupported in candle-transformers

---

## Directory Structure

```
C:\Users\Foobis\Documents\Source\rusty-cunny\
├── .cargo/
│   └── config.toml                    [CUDA 13.2 env config, force=true]
├── src/
│   ├── main.rs                        [Tokio folder watcher, VRAM state machine]
│   ├── lib.rs                         [Module exports]
│   ├── detection.rs                   [YoloDetector: YOLOv8-seg backbone]
│   ├── mask.rs                        [MaskGenerator: proto matmul + dilation]
│   ├── inpaint.rs                     [SdxlInpaint: DDIM + dual encoder drop]
│   └── pipeline.rs                    [RestorationPipeline: orchestrator]
├── gui/
│   └── RustyCunnyGui/
│       ├── RustyCunnyGui.csproj       [.NET 9.0 win-x64, Avalonia 11.2.3]
│       ├── MainWindow.xaml            [Input/output dirs, prompt, log panel]
│       ├── MainWindow.xaml.cs         [Subprocess launch, stdout/stderr pipe]
│       ├── Program.cs                 [App entry, FluentTheme]
│       └── App.xaml                   [Resource definitions]
├── tools/
│   └── convert_yolo_to_safetensors.py [Ultralytics .pt → SafeTensors]
├── Cargo.toml                         [Dependencies: candle, tokio, image, ndarray]
├── CLAUDE.md                          [This file]
├── GEMINI.md                          [High-level overview for external AI]
└── README.md                          [User-facing build + usage guide]
```

---

## Next Steps

1. **Model Weights:** Acquire YOLOv8-seg-l.pt + SDXL 1.0 checkpoints, convert via `convert_yolo_to_safetensors.py`
2. **Build & Test:** `cargo build --release` + validate CUDA 13.2 linkage
3. **GUI Integration:** Complete .NET project, test subprocess stdio piping
4. **VRAM Benchmarking:** Profile memory usage across DDIM steps, optimize encoder drop timing
5. **Production Hardening:** Error handling, model caching, batch processing
