#!/usr/bin/env python3
"""
Lightweight forward-pass smoke test for the NoKSR inference stack.

Intended to be executed inside the inference container with the repository
mounted at /workspace. The script fabricates a minimal in-memory stub of the
`nksr` dependency so we can sanity-check the model without the official CUDA
extensions.
"""

from __future__ import annotations

import argparse
import sys
import types
from importlib import import_module

import torch
import os
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
from hydra import compose, initialize
from pathlib import Path


def install_nksr_stub() -> None:
    """Registers a minimal stub for `nksr.svh.SparseFeatureHierarchy`."""

    class _Grid:
        def __init__(self, voxel_size: float, device: torch.device) -> None:
            self.voxel_size = voxel_size
            self.device = device

        def active_grid_coords(self) -> torch.Tensor:
            return torch.zeros((1, 3), device=self.device, dtype=torch.long)

        def grid_to_world(self, coords: torch.Tensor) -> torch.Tensor:
            return coords.to(torch.float32) * self.voxel_size

    class SparseFeatureHierarchy:
        def __init__(self, voxel_size: float, depth: int, device: torch.device) -> None:
            self.voxel_size = voxel_size
            self.depth = max(depth, 1)
            self.device = device
            self.grids = [_Grid(voxel_size, device) for _ in range(self.depth)]

        def build_adaptive_normal_variation(self, *args, **kwargs) -> "SparseFeatureHierarchy":
            return self

        def build_point_splatting(self, *args, **kwargs) -> "SparseFeatureHierarchy":
            return self

    svh_module = types.ModuleType("nksr.svh")
    svh_module.SparseFeatureHierarchy = SparseFeatureHierarchy  # type: ignore[attr-defined]

    nksr_module = types.ModuleType("nksr")
    sys.modules.setdefault("nksr", nksr_module)
    sys.modules["nksr.svh"] = svh_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal forward pass with synthetic data.")
    parser.add_argument(
        "--dataset-root",
        default="/workspace/sample_data",
        help="Root directory that contains the synthetic sample dataset.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Move tensors to CUDA if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_nksr_stub()

    overrides = [
        "model=synthetic_model",
        "data=synthetic",
        f"data.dataset_root_path={args.dataset_root}",
        f"data.path={args.dataset_root}/synthetic",
        "data.classes=null",
        "data.multi_files=1",
        "data.take=1",
        "data.intake_start=0",
        "data.num_workers=0",
        "model.ckpt_path=",
    ]

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config", overrides=overrides)

    from noksr.data.data_module import DataModule

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"[SmokeTest] Using device: {device}")

    data_module = DataModule(cfg)
    data_module.setup("test")
    batch = next(iter(data_module.val_dataloader()))

    def move(item):
        return item.to(device) if hasattr(item, "to") else item

    batch = {k: move(v) for k, v in batch.items()}

    module = import_module("noksr.model")
    model_cls = getattr(module, cfg.model.network.module)
    model = model_cls(cfg).to(device)
    model.eval()

    with torch.no_grad():
        outputs, _ = model.forward(batch)

    shapes = {k: v.shape for k, v in outputs.items() if isinstance(v, torch.Tensor)}
    print(f"[SmokeTest] Forward pass completed. Tensor shapes: {shapes}")


if __name__ == "__main__":
    main()
