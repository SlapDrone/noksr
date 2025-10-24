import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for CI
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = [
    (
        "mobile_mapping_laz",
        "tests/data/pointcloud_samples/mobile_mapping_sample.laz",
        {"voxel_size": "0.10", "max_points": "8000"},
    ),
    (
        "opentrench_ply",
        "tests/data/pointcloud_samples/opentrench_sample.ply",
        {"voxel_size": "0.05", "max_points": "20000"},
    ),
    (
        "pinpoint_las",
        "tests/data/pointcloud_samples/pinpoint_sample.las",
        {"voxel_size": "0.20", "max_points": "10000"},
    ),
    (
        "road_txt",
        "tests/data/pointcloud_samples/road_sample.txt",
        {"voxel_size": "0.20", "max_points": "10000"},
    ),
]


@pytest.mark.parametrize("name,relative_path,params", SAMPLES)
def test_pointcloud_inference_runs(name, relative_path, params):
    if torch is None or not torch.cuda.is_available():
        pytest.skip("CUDA device is required for NKSR inference")

    sample_path = REPO_ROOT / relative_path
    assert sample_path.exists(), f"Missing sample {sample_path}"

    cmd = [
        sys.executable,
        "scripts/pointcloud_inference.py",
        str(sample_path),
        "--use-cuda",
        "--center",
        "--estimate-normals",
        f"--voxel-size={params['voxel_size']}",
        f"--max-points={params['max_points']}",
    ]

    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, f"Inference failed for {name}:\n{proc.stdout}"
    assert "[Query] SDF stats" in proc.stdout
    if "--disable-gradients" not in cmd:
        assert "[Query] Gradient L2 norm" in proc.stdout
