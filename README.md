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
- SDXL Union ControlNet runs as a TensorRT engine with CUDA Graphs when acceleration is TensorRT: one engine per active control-type set, batch and resolution, cached under `tensorrt_cache/sdxl/controlnet/union/` (first build ~4-5 min; ~23 ms vs ~46 ms in torch.compile at 1024x1024 on an RTX 5080). Conditioning scales stay live inputs, so slider changes never rebuild.
- Pre-flight engine cache fast path that skips the PyTorch UNet/VAE load entirely on warm starts. On SDXL it also applies with StreamV2V (the cached engine carries the cache ports) and with UNet-only LoRAs such as Hyper-SD (the LoRA is baked into the cached engine, whose directory carries the LoRA signature); only a LoRA with text-encoder weights, or one that cannot be inspected, still needs the full pipeline (`_loras_touch_text_encoders`).
- SSF (Similar Image Filter) preprocessor gating to skip frames when the input is unchanged
- `controlnet.skip_frames` > 1 now skips the whole ControlNet on the in-between frames (preprocessors and ControlNet forward): the cached control maps and the previous residuals are reused (single-step streams). With the SDXL Union engine this removes ~23 ms on every skipped frame at 1024x1024; the control signal refreshes at fps/N.
- **Engine precision** (`precision`: `fp16` | `mxfp8` | `nvfp4`, env `STREAMDIFFUSION_PRECISION`, set to `mxfp8` by `StartStreamDiffusion.bat`): on RTX 50 (Blackwell) TensorRT only has native low-precision GEMM kernels for the block-scaled formats, so the SDXL UNet and the SDXL Union ControlNet are quantized with NVIDIA ModelOpt at engine-build time (Linear layers only, LoRA baked first; MXFP8 needs no calibration, NVFP4 calibrates on frames generated by the pipeline) and built as strongly-typed engines (`pipeline/acceleration/quantization.py`). Engines get a `--prec-<precision>` suffix so fp16 and quantized engines coexist. Measured on an RTX 5080 @1024x1024 (UNet ms/step): fp16 41.2, mxfp8 36.0 (-12.6%, image visually identical), nvfp4 29.1 (-29%, the 1-step image drifts). Per-tensor FP8 gives nothing on this GPU. Ignored outside TensorRT. The SDXL StreamV2V engine is quantized the same way (`unet_v2vr_xl_mfN--prec-<precision>.engine`, kvo cache ports included); the Union ControlNet follows the same `precision` setting (its NVFP4 calibration runs frames generated by the live pipeline through the real preprocessors); a failed quantized build falls back to the fp16 engine.
- **Tiny VAE** (`use_tiny_vae`, default on): SDXL decodes with `madebyollin/taesdxl` (was `cqyan/hybrid-sd-tinyvae-xl`, which leaves a visible block/screen-door texture on skin; taesdxl is cleaner at the same 5.5 ms). Pass `vae_id` to use another tiny VAE; VAE engines are cached per tiny VAE (`vae_decoder--<model>.engine`). The full SDXL VAE is sharper still but costs ~120 ms/frame at 1024 as TensorRT engines, so it stays off the real-time path. SD 1.5 keeps `madebyollin/taesd`.
- **StreamV2V ring cache (SDXL)**: the attention cache is exposed as one engine input per cached frame plus one output for the current frame (`kvo_in_i_j` / `kvo_out_i`, `UNetXLV2VRing`, `KvoRingAttnProcessor2_0`). The runtime keeps `max_frames + 1` slot buffers and rotates the engine's tensor addresses, with one CUDA graph per phase (`Engine.graphs`), so the cache is never copied or shifted: the previous layout copied ~2.4 GB per frame around the engine and shifted the whole cache inside it. Measured @1024, 2 cached frames: 48.2 -> 44.3 ms/frame fp16, VRAM -1.9 GB. Engine files are `unet_v2vr_xl_mfN[...]` (the older `unet_v2v_xl_mfN` engines are not reused). SD 1.5 keeps the previous layout.
- **StreamV2V speed/quality options** (SDXL ring engine, `StreamV2VConfig`, all off by default, env `STREAMDIFFUSION_V2V_ATTN_POOL` / `_FI_LAST` / `_ATTN_DECODER`, see `StartStreamDiffusion.bat`): `attn_cache_pool=2` average-pools the cached keys/values 2x2 before the extended attention (attention over 1.5x tokens instead of 3x; 40.9 -> 37.1 ms/frame mxfp8 @1024), `fi_last_frame_only` matches the feature injection against the newest cached frame only (-1 ms, output nearly unchanged), `attn_decoder_only` keeps the down blocks on plain self-attention (38.3 ms; their unread cache ports are dropped from the engine; the profile and the runtime tolerate missing ports). All three together: 34.4 ms/frame mxfp8, i.e. the plain mxfp8 engine (29.7 ms) plus 4.7 ms. Each combination gets its own engine file tag (`--pool2--filast--attndec`). The remaining v2v cost is the extended attention and the feature injection themselves; judge these options on real video, not on stills.
- Canny preprocessor runs on the GPU (PyTorch re-implementation of OpenCV's blur + Canny + speck filter, captured in a CUDA graph per resolution/aperture; thresholds are live tensors): 0.5 ms @384, 1.6 ms @1024 on an RTX 5080 vs 1.9 / 11.5 ms for the OpenCV path, so `canny.resolution` can be raised to the stream resolution for sharper edges at no cost. `canny.gpu: false` restores the OpenCV path, which is also the automatic fallback.
- OpenPose (DWPose) preprocessor runs its two models as TensorRT engines (`preprocessors/dwpose_trt.py`: YOLOX-S 640x640 with a GPU decode + NMS, DW-LL 384x288; engines built once into `tensorrt_cache/dwpose/`) and, with `preprocessors/dwpose_gpu.py`, keeps the whole pass on the GPU: letterbox, affine person crops (`grid_sample`), SimCC decode, skeleton rasterized as batched distance fields (`draw_pose_gpu`, same primitives/colors/order as easy_dwpose's OpenPose drawing) and both engines replayed as CUDA graphs; only the keypoints come back to the CPU for the temporal filtering. ~4.7 ms/frame vs ~17 ms with the numpy/OpenCV post-processing and ~26 ms with onnxruntime, at detect_resolution 512 on an RTX 5080. The numpy path stays as the automatic fallback; `openpose.tensorrt: false` restores onnxruntime.
- TensorRT engines carry a `.trtver` marker (`pipeline/acceleration/engine_cache.py`): engines built by another TensorRT version are rebuilt instead of failing to load. Requires TensorRT 10.16 (`requirements.txt`).
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
