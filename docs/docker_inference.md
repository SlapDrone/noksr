# NKSR Builder & Inference Container

This repository ships a reproducible GPU inference environment that bundles the NoKSR CUDA
extensions with a minimal runtime stack. The multi-stage Dockerfile compiles NKSR from source,
strips out the build toolchain, and leaves you with a lightweight image ready for SDF + gradient
queries.

---

## 1. Build The Runtime Image

```bash
docker build -f docker/Dockerfile.inference -t noksr-inference .
```

What happens during the build:

1. **Builder stage**
   - Installs PyTorch 2.0.0 + CUDA 11.8, pybind11, compiler toolchain
   - Clones `nv-tlabs/nksr` (branch `public`)
   - Produces an NKSR wheel via `pip wheel --no-build-isolation`

2. **Runtime stage**
   - Installs core runtime dependencies (Lightning, PyTorch3D, flash-attn, Open3D, etc.)
   - Installs `laspy[lazrs]` so LAS/LAZ point clouds are supported out of the box
   - Copies the freshly-built NKSR wheel and verifies PyTorch/FlashAttention versions

The resulting image has no compiler toolchain or transient build artifacts—just the runtime stack
required for inference.

---

## 2. Quick Smoke Test (Synthetic Dataset)

```bash
docker run --rm --runtime=nvidia \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$(pwd)":/workspace \
  noksr-inference \
  python scripts/smoke_test_forward.py --use-cuda
```

You should see the script print tensor shapes for the decoder outputs and report that CUDA is
available. The smoke test uses a synthetic dataset bundled in `sample_data/` and stubs out NKSR’s
Sparse Feature Hierarchy when the CUDA extensions are unavailable.

---

## 3. Running Hydra-Based Evaluation

To evaluate against a known dataset and checkpoint, supply standard Hydra overrides to `eval.py`:

```bash
docker run --rm --runtime=nvidia \
  -v /abs/path/to/data:/data \
  -v "$(pwd)":/workspace \
  noksr-inference \
  python eval.py model=scannet_model \
    data=scannet \
    data.dataset_root_path=/data \
    model.ckpt_path=/workspace/checkpoints/ScanNet_Serial_best.ckpt
```

Any additional overrides can be appended exactly as you would in a native environment.

---

## 4. Field Queries & Point-Cloud Inference

Two helper scripts cover the two common workflows:

| Script | When to use it |
| --- | --- |
| `scripts/query_sdf_and_gradient.py` | You already have a Hydra dataset configuration and want the decoder’s SDF/gradient outputs on that dataset. |
| `scripts/pointcloud_inference.py` | You have an arbitrary point cloud (PLY, PCD, LAS/LAZ, XYZ/TXT, …) and want a one-shot SDF + gradient query without touching the datamodule. |

Example: direct point cloud ingestion

```bash
docker run --rm --runtime=nvidia \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$(pwd)":/workspace \
  -v /abs/path/to/point_clouds:/data \
  noksr-inference \
  python scripts/pointcloud_inference.py \
    /data/mobile_mapping/example.laz \
    --use-cuda \
    --estimate-normals \
    --center \
    --voxel-size 0.10 \
    --max-points 200000 \
    --output /workspace/query_results.npz
```

Both scripts accept `--hydra-override key=value` flags. For ad-hoc clouds you can also point
directly at a checkpoint via `--checkpoint /workspace/checkpoints/your_model.ckpt`.

---

## 5. Choosing Inference Parameters

The decoder expects reasonably scaled inputs (roughly meter units) and a manageable number of
points. Use the table below as a starting point; adjust `--voxel-size` and `--max-points` based on
GPU memory and desired fidelity.

| Scenario | Example data | Suggested flags | Notes |
| --- | --- | --- | --- |
| **Mobile mapping LiDAR (dense LAZ/LAS)** | Street-level scans with millions of points | `--estimate-normals --center --voxel-size 0.10 --max-points 200000` | Centering keeps Hilbert depth low; voxel downsampling avoids memory blowups. |
| **Tripod scan / survey (very dense LAS)** | Construction-site or Leica scans | `--estimate-normals --center --voxel-size 0.30 --max-points 50000` | Large sites can exceed GPU memory—coarser voxel size keeps the PointTransformer stable. |
| **Reconstruction mesh exports (PLY/PCD)** | Indoor/outdoor meshes from other pipelines | `--estimate-normals --center --voxel-size 0.05 --max-points 200000` | If normals are already stored in the file, you can omit `--estimate-normals`. |
| **ASCII XYZ/TXT clouds** | Sensor dumps with XYZ(+intensity) columns | `--estimate-normals --center --voxel-size 0.20 --max-points 100000` | Intensity (4th column) is treated as grayscale colour for feature construction. |
| **Small synthetic rooms (sample_data)** | Repository synthetic dataset | `--use-cuda --estimate-normals --center --voxel-size 0.02 --max-points 65536` | Matches the synthetic config used in CI smoke tests. |

Tips:

- `--center` subtracts the mean XYZ before encoding; this avoids Hilbert-code overflow for large
  coordinate magnitudes.
- If the point cloud already includes normals and you trust them, drop `--estimate-normals` to save
  time.
- When `--use-cuda` is omitted or `torch.cuda.is_available()` is false, the stack runs fully on CPU.
  Expect much longer runtimes and prohibitively large memory requirements for dense clouds.

---

## 6. Docker Tips & Troubleshooting

- **GPU access**: always run with `--runtime=nvidia` (or the equivalent as configured on your host).
  The PointTransformer encoder relies on spconv, which asserts for CUDA tensors by default.
- **Large clouds / OOM**: increase `--voxel-size`, lower `--max-points`, or split the cloud before
  inference. The NKSR backbone was tuned on ~10⁵ points per scene.
- **Normals missing**: the CLI can estimate normals via Open3D (`--estimate-normals`). For noisy
  LiDAR scans you may want to try different voxel sizes or post-filtering to get stable gradients.
- **File format support**:
  - PLY, PCD, XYZ and other Open3D-supported formats
  - LAS/LAZ via [`laspy[lazrs]`](https://laspy.readthedocs.io/)
  - Plain-text XYZ(+intensity) via `numpy.loadtxt`
- **Hydra overrides**: append `--hydra-override key=value` to either script when you need to tweak
  configuration knobs (e.g., `--hydra-override model=scannet_model`).
- **Artifact handling**: use `--output /workspace/output.npz` to persist queried coordinates, SDF
  values, and gradients; the file can be pulled from the host after the container exits.
- **Bytecode**: the runtime sets `PYTHONDONTWRITEBYTECODE=1` to avoid polluting bind mounts with
  `.pyc` files. Override the environment variable if you prefer bytecode caches.

---

---

## 7. Publishing To GitHub Container Registry (GHCR)

1. Authenticate once (PAT must have `write:packages` scope):

   ```bash
   echo "${GHCR_PAT}" | docker login ghcr.io -u <your-github-username> --password-stdin
   ```

2. Build and tag the image:

   ```bash
   docker build -f docker/Dockerfile.inference -t ghcr.io/<owner>/noksr-inference:latest .
   ```

3. Push:

   ```bash
   docker push ghcr.io/<owner>/noksr-inference:latest
   ```

For repeatable automation, a GitHub Actions workflow (`.github/workflows/publish-inference-image.yml`)
is included. Configure it by setting `IMAGE_NAME` (defaults to `noksr-inference`) and enabling the
workflow—`secrets.GITHUB_TOKEN` already has permission to publish to GHCR for the current repository.

---

## 8. Container Help Command

Running the container without overriding the command prints a concise quick-start guide:

```bash
docker run --rm noksr-inference
```

To display the full markdown documentation from inside the image, run:

```bash
docker run --rm noksr-inference python scripts/container_help.py --full
```

---

With these presets and commands you can feed virtually any point cloud into NoKSR, evaluate SDFs
and their spatial derivatives, and iterate on downstream pipelines without leaving the Docker
environment—or publish the toolchain to GHCR for downstream consumers.
