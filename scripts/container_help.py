#!/usr/bin/env python3
"""
Container-facing help for the NoKSR inference runtime.

This script prints a concise onboarding guide when executed inside the Docker
image (e.g. when the container is started without overriding the command).
Use `--full` to emit the complete markdown guide.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs/docker_inference.md"


SUMMARY = dedent(
    """\
    NoKSR Inference Container
    =========================

    Common commands:

      • Smoke test with synthetic data
          docker run --rm --runtime=nvidia -v "$(pwd)":/workspace \\
            noksr-inference python scripts/smoke_test_forward.py --use-cuda

      • Run eval.py with Hydra overrides
          docker run --rm --runtime=nvidia \\
            -v /abs/path/to/data:/data \\
            -v "$(pwd)":/workspace \\
            noksr-inference \\
            python eval.py model=scannet_model data=scannet \\
              data.dataset_root_path=/data \\
              model.ckpt_path=/workspace/checkpoints/ScanNet_Serial_best.ckpt

      • Query an arbitrary point cloud
          docker run --rm --runtime=nvidia \\
            -v "$(pwd)":/workspace \\
            -v /abs/path/to/point_clouds:/data \\
            noksr-inference \\
            python scripts/pointcloud_inference.py /data/example.laz \\
              --use-cuda --estimate-normals --center \\
              --voxel-size 0.1 --max-points 200000 \\
              --output /workspace/query_results.npz

    The inference documentation (docs/docker_inference.md) covers:
      • Detailed build and publish steps
      • Parameter presets for different point-cloud modalities
      • Troubleshooting tips for GPU execution, downsampling, and outputs

    Run `python scripts/container_help.py --full` inside the image to print the
    full guide.
    """
)


def print_full_documentation() -> None:
    if not DOC_PATH.exists():
        print("Full documentation not found at docs/docker_inference.md", file=sys.stderr)
        sys.exit(1)

    pager = shutil.which("less") or shutil.which("more")
    content = DOC_PATH.read_text(encoding="utf-8")

    if pager and sys.stdout.isatty():
        import subprocess

        proc = subprocess.run(
            [pager],
            input=content,
            text=True,
        )
        sys.exit(proc.returncode)

    print(content)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print NKSR inference container help.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Display the complete documentation rather than the short summary.",
    )
    args = parser.parse_args(argv)

    if args.full:
        print_full_documentation()
    else:
        print(SUMMARY)


if __name__ == "__main__":
    main()
