#!/usr/bin/env python3
"""
Utility script for querying the NoKSR decoder at arbitrary coordinates and returning
both the scalar SDF prediction and its spatial derivatives.

Intended to run either inside the inference container or a suitably provisioned
environment where the NKSR CUDA extensions are available. When the extensions are
absent, the script can fall back to the lightweight stub used by the smoke test.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from importlib import import_module
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from hydra import compose, initialize

from noksr.utils.inference import (
    SDFQueryResult,
    encode_scene,
    move_batch_to_device,
    query_sdf_and_gradient,
    summarize_query_result,
)


def install_nksr_stub() -> None:
    """Registers a minimal stub for `nksr.svh.SparseFeatureHierarchy` when the extension is absent."""
    try:
        import_module("nksr.svh")
        return
    except ModuleNotFoundError:
        pass

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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query NoKSR SDF values and gradients.")
    parser.add_argument(
        "--dataset-root",
        default="/workspace/sample_data",
        help="Root directory that contains the dataset referenced by the Hydra config.",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="Index of the validation scene to fetch from the dataloader.",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        help="Optional path to a .npy/.npz file containing query coordinates with shape [N, 3].",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=2048,
        help="Number of query points to subsample from the dataloader when --query-file is not provided.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output .npz file to store query coordinates, SDF predictions, and gradients.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Move model and tensors to CUDA if available.",
    )
    parser.add_argument(
        "--disable-gradients",
        action="store_true",
        help="Skip gradient computation and return only scalar SDF predictions.",
    )
    parser.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        help="Additional Hydra overrides (e.g. --hydra-override model=scannet_model).",
    )
    return parser.parse_args(argv)


def _load_queries_from_file(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")

    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        if "coords" in archive:
            data = archive["coords"]
        elif len(archive.files) == 1:
            data = archive[archive.files[0]]
        else:
            raise KeyError(f"NPZ archive {path} must contain a 'coords' array or exactly one entry.")
    else:
        raise ValueError(f"Unsupported query file extension '{path.suffix}'. Expected .npy or .npz.")

    tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
    if tensor.ndim != 2 or tensor.size(-1) != 3:
        raise ValueError(f"Query coordinates must have shape [N, 3]; received {tuple(tensor.shape)} from {path}.")
    return tensor


def _subsample_queries_from_batch(batch: dict, count: int) -> torch.Tensor:
    coords: torch.Tensor = batch["xyz"]
    if coords.ndim != 2 or coords.size(-1) != 3:
        raise ValueError("Batch does not contain queryable coordinates with shape [N, 3].")
    if coords.size(0) <= count:
        return coords
    selection = torch.randperm(coords.size(0))[:count]
    return coords.index_select(0, selection)


def _select_batch(loader: Iterable[dict], index: int) -> dict:
    for idx, batch in enumerate(loader):
        if idx == index:
            return batch
    raise IndexError(f"Scene index {index} is out of range for the provided dataloader.")



def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    install_nksr_stub()

    from noksr.data.data_module import DataModule

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.use_cuda and not use_cuda:
        raise RuntimeError("CUDA requested via --use-cuda but no CUDA device is available.")

    default_overrides = [
        "model=synthetic_model",
        "data=synthetic",
        f"data.dataset_root_path={args.dataset_root}",
        f"data.path={args.dataset_root}/synthetic",
        "data.classes=null",
        "data.multi_files=1",
        "data.take=-1",
        "data.intake_start=0",
        "data.num_workers=0",
        "model.ckpt_path=",
    ]
    overrides = default_overrides + args.hydra_override

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config", overrides=overrides)

    data_module = DataModule(cfg)
    data_module.setup("test")
    val_loader = data_module.val_dataloader()
    batch = _select_batch(val_loader, args.scene_index)

    query_xyz = (
        _load_queries_from_file(Path(args.query_file))
        if args.query_file
        else _subsample_queries_from_batch(batch, args.num_queries)
    )

    batch = move_batch_to_device(batch, device)
    query_xyz = query_xyz.to(device)
    if not args.disable_gradients:
        query_xyz = query_xyz.clone().detach().requires_grad_(True)

    module = import_module("noksr.model")
    model_cls = getattr(module, cfg.model.network.module)
    model = model_cls(cfg).to(device)
    model.eval()

    if getattr(cfg.model, "ckpt_path", ""):
        ckpt_path = Path(cfg.model.ckpt_path)
        if ckpt_path.is_file():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["state_dict"])
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    encoder_outputs = encode_scene(model, batch, device=device)
    result = query_sdf_and_gradient(
        model.sdf_decoder,
        encoder_outputs,
        query_xyz,
        compute_gradients=not args.disable_gradients,
    )
    result = result.as_detached()
    print(summarize_query_result(result))

    if args.output:
        payload = {
            "coords": query_xyz.detach().cpu().numpy(),
            "sdf": result.sdf.cpu().numpy(),
        }
        if result.gradients is not None:
            payload["gradients"] = result.gradients.cpu().numpy()
        np.savez(args.output, **payload)
        print(f"[Query] Results saved to {args.output}")


if __name__ == "__main__":
    main()
