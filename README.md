# StreamDiffusion R15

Real-time Stable Diffusion runtime used by Smode's StreamDiffusion R15 engine.

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-downloads)
[![TensorRT](https://img.shields.io/badge/TensorRT-10%2B-76B900.svg)](https://developer.nvidia.com/tensorrt)

## What it is

StreamDiffusion R15 is a real-time Stable Diffusion inference engine targeting live visual production. It supports SD 1.5 and SDXL Turbo backbones, ControlNet conditioning (canny, depth, openpose, FaceID, and the xinsir SDXL Union ControlNet), and the StreamV2V temporal-consistency mechanism running natively on TensorRT.

The runtime is optimised for sub-frame latency: TensorRT engines for the UNet, VAE and ControlNet branches are pre-built and cached, CUDA Graphs are captured on stable-shape engines, and per-frame allocations are eliminated wherever possible. Models are pulled from HuggingFace on first use and engine builds live under `tensorrt_cache/`.

This package is **based on** [cumulo-autumn/StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) — the original real-time SD pipeline paper implementation — but the codebase has diverged substantially. The TensorRT path was rewritten around CUDA Graphs and per-engine optimisations, ControlNet support was added with both per-model and SDXL Union variants, StreamV2V temporal consistency was wired through the TRT engine I/O, an IPC layer for Smode integration was added, and a long tail of latency/VRAM optimisations was applied throughout. The high-level batched-denoising algorithm and rolling cache from the original StreamDiffusion are preserved.

## What it is NOT

- **Not a general-purpose diffusion library.** It is a runtime tuned for one workload (real-time img2img at low step counts) and does not aim to cover the breadth of `diffusers`.
- **Not a fine-tuning / training framework.** There is no training code, no LoRA trainer, no dataset tooling.
- **Not a standalone application.** It is designed to be driven by Smode's StreamDiffusion R15 engine over the IPC protocol in `ipc/`. Running it on its own is possible but unsupported.

## Architecture overview

```
StreamDiffusion-R15/
├── pipeline/         # StreamDiffusion SD 1.5 / SDXL inference + TensorRT acceleration
├── engines/          # Wrapper layer (high-level pipeline construction, TRT engine load/build)
├── controlnet/       # ControlNet model loading + scale management
├── preprocessors/    # ControlNet input preprocessors (canny, depth, openpose, FaceID)
├── ipc/              # Smode IPC protocol (shared CUDA texture, command channel, signaling)
└── config/           # Runtime configuration schema
```

## Requirements

- NVIDIA GPU with CUDA 12.x support (Ampere or newer recommended)
- Windows 10 / 11 (developed and tested). Linux may work but is not officially supported.
- Python 3.11 (the venv is pinned to this; 3.12+ will not work because several pinned wheels do not publish for it)
- TensorRT 10+, PyTorch 2.10+
- Approximately 10 GB of free disk for engine caches at runtime, more depending on resolution and the number of ControlNet variants you build

Exact pinned versions live in [`requirements.txt`](requirements.txt).

## Installation

**Quickstart (recommended)**

From the package root, run:

```
install.bat
```

This creates the `.venv`, installs all pinned requirements, copies the CUDA helper binaries Triton needs, and installs the pre-built insightface wheel for FaceID.

**Manual install**

For development or troubleshooting:

```
python -m virtualenv --copies .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python setup_venv.py
```

**Verify the install**

```
python verify_install.py
```

**Install status marker (for Smode Engine)**

While it runs, `install.bat` writes `install_status.json` next to itself (package root) after every phase transition, so Smode Engine can poll the file and drive a status icon without parsing console output:

```json
{
  "status": "installing",
  "step": 2,
  "totalSteps": 4,
  "message": "Installing dependencies from requirements.txt"
}
```

- `status` is one of `installing`, `success`, `failed`.
- `step` / `totalSteps` track progress through the 5 phases (0-4: prerequisites, venv, dependencies, CUDA/Triton config, verification).
- `message` is a short human-readable string (no quotes or parentheses).
- The file is absent before the first install run, and is overwritten (not deleted) on every subsequent run — the last state (`success` or `failed`) persists until the next install starts.

Smode Engine should treat `status == "installing"` as "show the install warning/progress icon" and `success`/`failed` as "clear it" (with `failed` optionally surfacing `message`).

**Runtime warning packet (for Smode Engine / StreamDiffusionTextureModifier)**

There is no live socket connection during `install.bat`, so it uses the polled JSON file above. At runtime the Python process already holds an open socket to Smode for the duration of the node (frame data, config, `STREAM_CREATION`), so the equivalent "this is going to take a while" signal for `torch.compile()` warmup and TensorRT engine builds is pushed as a packet instead of polled.

- New `CommandType.WARNING = 8`.
- Payload: `uint32 active` (0/1) + length-prefixed UTF-8 `message` (`uint32 length` + bytes) — same layout convention as the other string fields on this wire (see `UuidPacket`).
- Sent with `active = 1` and a human-readable message right before a slow one-time prep step starts, and with `active = 0` as soon as that step finishes or fails. It is never sent at all when the relevant engine/compile cache already exists on disk (warm start).
- One packet per checkpoint, mirroring `install.bat`'s per-phase `write_status` calls rather than one blanket "loading" flag: e.g. `"Building TensorRT engine (UNet) - ..."`, then `"Building TensorRT engine (VAE decoder) - ..."`, then `"Building TensorRT engine (VAE encoder) - ..."` as each is actually built (skipped individually if already cached). Same idea for `torch.compile()`: U-Net, VAE encoder, VAE decoder are separate messages.
- Multiple `WARNING` packets can arrive in sequence within one stream (re)load; treat each `active = 1` as replacing the previous message (still "busy"), not as toggling/counting independent warnings.
- **Timing relative to `STREAM_CREATION`:** TensorRT engine builds and the `torch.compile()` wrap itself happen inside `_create_stream()`, so those `WARNING` packets land between `STREAM_CREATION(False)` and `STREAM_CREATION(True)`. But `torch.compile()` compiles lazily — the actual JIT compilation only happens on the first real forward pass, which runs in the warmup step *after* `STREAM_CREATION(True)` has already been sent. So a `WARNING(active=1, "Warming up torch.compile...")` can arrive after the node is already marked "ready." Do not assume `STREAM_CREATION(True)` implies no more `WARNING` packets are coming.
- Smode Engine should treat `active = 1` as "show the node's warning icon" with `message` as the tooltip, and `active = 0` as "clear it" — same semantics as the install status icon, just delivered over the socket instead of a polled file.

## Usage

This package is launched by Smode's StreamDiffusion R15 engine and communicates with it over an IPC channel (shared CUDA textures for frames, a command channel for parameters, signaling events for sync). End users do not run it directly.

- Runtime configuration is loaded from `controlnet_config.json` at startup. A sample with all ControlNets disabled is included in the repo.
- Models are downloaded from HuggingFace on first use. TensorRT engines are built lazily and cached under `tensorrt_cache/`. First-time builds can take several minutes per engine.

If you want to drive the runtime yourself, `StartStreamDiffusion.bat` shows the entry point and the CLI arguments Smode passes in.

## Features

- SD 1.5 and SDXL Turbo support via separate wrappers
- TensorRT acceleration for UNet, VAE and ControlNet engines
- StreamV2V temporal consistency with the `kvo` cache exposed as engine I/O — the first public implementation on TensorRT we are aware of
- SDXL Union ControlNet integration ([xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0)) replacing the three legacy SDXL ControlNets
- Pre-flight engine cache fast path that skips the PyTorch UNet/VAE load entirely on warm starts
- SSF (Similar Image Filter) preprocessor gating to skip frames when the input is unchanged
- CUDA Graph capture on stable-shape engines
- Per-frame VRAM optimisations: zero-CN residuals cache, `shape_dict` cheap-key, in-place residual scaling, hoisted `init_noise` roll, FP16 safety-checker load

## Caveats / known limitations

- Tight coupling to Smode's IPC protocol — this is not a drop-in standalone library
- Changing resolution mid-session requires a restart (engine bindings are baked at build time)
- StreamV2V TensorRT support: SD 1.5 is fully wired. SDXL is implemented but the first engine build is expensive (15-25 minutes) and produces a ~3 GB engine cache

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.

## Credits

Built on the foundations of [cumulo-autumn/StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) (the original real-time SD pipeline paper by Akio Kodaira et al.), reworked extensively for Smode's real-time production use case. StreamV2V temporal consistency adapted from the [StreamV2V](https://github.com/Jeff-LiangF/streamv2v) paper. SDXL Union ControlNet from [xinsir](https://huggingface.co/xinsir).
