REM Add CUDA to PATH BEFORE venv activation to ensure Triton can find tools
if defined CUDA_PATH (
    set "PATH=%CUDA_PATH%\bin;%PATH%"
) else (
    set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%"
)

REM Create a virtual environment
if not exist .venv (
    echo Creating virtual environment...
    call "%CD%\..\python-3_11_9\python.exe" -m virtualenv --copies .venv
) else (
    echo Virtual environment already exists.
)

REM TensorRT engine precision for SDXL (fp16 | mxfp8 | nvfp4). RTX 50 only for mxfp8/nvfp4.
REM mxfp8: ~-12%% UNet time, image visually identical to fp16. nvfp4: ~-29%% but the 1-step image drifts.
REM First build per model+LoRA+steps+resolution quantizes and compiles (5-15 min), then cached.
if not defined STREAMDIFFUSION_PRECISION set STREAMDIFFUSION_PRECISION=nvfp4

REM StreamV2V speed/quality options (SDXL TensorRT engine), all off by default. Each combination
REM builds its own engine once (~5 min). Uncomment to test:
REM   ATTN_POOL=2   : cached keys/values pooled 2x2 before the extended attention
REM   FI_LAST=1     : feature injection matched against the newest cached frame only
REM   ATTN_DECODER=1: extended attention in mid/up blocks only
REM Multi-step (2+ t_index) on SDXL: steps run sequentially in the batch-1 engines (no batch-N
REM engine build, one frame of latency). Options:
REM   set STREAMDIFFUSION_CN_FIRST_STEP_ONLY=1   ControlNet on the first step only (~-16 ms/frame)
REM   set STREAMDIFFUSION_DENOISING_BATCH=1      legacy StreamDiffusion stream batch (batch-N engine)
REM Diagnostics: [PERF] log line every 60 frames (GPU breakdown, frame-time spread, wait for Smode).
REM set STREAMDIFFUSION_PROFILING=1
 set STREAMDIFFUSION_V2V_ATTN_POOL=2
 set STREAMDIFFUSION_V2V_FI_LAST=1
 set STREAMDIFFUSION_V2V_ATTN_DECODER=1

call .\.venv\Scripts\activate

echo Starting StreamDiffusion with args %1 %2 %3 %4 %5 %6
call python.exe SmodeStreamDiffusion.py --uuid %1 --port %2 --width %3 --height %4 --device %5 --model %6
