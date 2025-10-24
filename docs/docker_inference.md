# NKSR Builder & Inference Container

This repository now includes a reproducible way to package NoKSR for GPU inference without relying on the original CUDA 11.8 / PyTorch 2.0 wheels that used to be hosted at `nksr.huangjh.tech`. The Docker workflow builds the NKSR CUDA extensions from source in a dedicated stage and delivers a lean runtime image that contains only the compiled wheel and the lightweight dependencies required for evaluation.

## Build

```bash
docker build -f docker/Dockerfile.inference -t noksr-inference .
```

The multi-stage Dockerfile performs three key steps:

1. **Builder stage**  
   - Installs PyTorch 2.0.0 + CUDA 11.8, pybind11, and compiler toolchain  
   - Clones `nv-tlabs/nksr` (branch `public`)  
   - Produces an NKSR wheel with `pip wheel --no-build-isolation`

2. **Runtime stage**  
   - Installs the repository’s runtime dependencies (Lightning, PyTorch3D, FlashAttention, etc.)  
   - Copies and installs the freshly built NKSR wheel

3. **Sanity check**  
   - Records PyTorch/FlashAttention versions inside the image

The final container contains no compiler toolchain or source tree—just the minimal runtime stack plus NKSR’s CUDA extension.

## Smoke Test

A tiny synthetic dataset is provided under `sample_data/` together with `scripts/smoke_test_forward.py`. The script stubs the NKSR sparse hierarchy when the CUDA kernels are unavailable (useful for quick checks) and runs a forward pass over the synthetic sample.

```bash
docker run --rm --runtime=nvidia \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$(pwd)":/workspace \
  noksr-inference \
  python scripts/smoke_test_forward.py --use-cuda
```

Expected output includes tensor shapes for the decoder outputs and confirmation that CUDA is available.

## Using Real Data

Replace the dataset overrides in `scripts/smoke_test_forward.py` or call `eval.py` directly with your Hydra configuration:

```bash
docker run --rm --runtime=nvidia \
  -v /path/to/your/data:/data \
  -v "$(pwd)":/workspace \
  noksr-inference \
  python eval.py model=scannet_model \
    data=scannet \
    data.dataset_root_path=/data \
    model.ckpt_path=/workspace/checkpoints/ScanNet_Serial_best.ckpt
```

Hydra overrides can be supplied as additional CLI arguments exactly as in the native environment.

## Notes

- The NKSR source repository currently targets CUDA 12.8 for its prebuilt wheels, hence the local rebuild. The multi-stage approach keeps the final image compatible with CUDA 11.8 required by this codebase.
- `PYTHONDONTWRITEBYTECODE=1` is set in the runtime so that mounted worktrees do not accumulate `.pyc` files. Adjust as needed if you build derivative images.
