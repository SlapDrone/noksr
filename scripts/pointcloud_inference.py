#!/usr/bin/env python3
"""
One-shot inference utility for arbitrary point clouds.

This script normalises an input point cloud (PLY, LAS/LAZ, etc.), prepares the
features expected by the NoKSR backbone, and queries the decoder for SDF values
and gradients—all without relying on the Hydra dataset loaders.
"""

from __future__ import annotations

import argparse
import os
from importlib import import_module
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from hydra import compose, initialize

from noksr.utils.inference import encode_scene, move_batch_to_device, query_sdf_and_gradient, summarize_query_result
from noksr.utils.pointcloud_io import (
    PointCloud,
    build_point_features,
    center_points,
    ensure_normals,
    random_subsample,
    read_point_cloud,
    voxel_downsample,
)

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NoKSR SDF + gradient inference on an arbitrary point cloud.")
    parser.add_argument("input", type=str, help="Path to the point cloud file (PLY, LAS/LAZ, PCD, etc.).")
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output .npz to store coordinates, SDF predictions, and gradients.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center the point cloud around the mean before inference.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Optional voxel size (in same units as the input cloud) for downsampling.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Randomly subsample the cloud to at most this many points after voxel downsampling.",
    )
    parser.add_argument(
        "--estimate-normals",
        action="store_true",
        help="Estimate normals when missing (required if the model expects them).",
    )
    parser.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        help="Additional Hydra overrides (e.g. --hydra-override model=scannet_model).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint. Overrides cfg.model.ckpt_path when provided.",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default=None,
        help="Optional .npy/.npz with explicit query coordinates. Defaults to the processed input cloud.",
    )
    parser.add_argument(
        "--disable-gradients",
        action="store_true",
        help="Skip gradient computation and return only SDF values.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Move model and tensors to CUDA if available.",
    )
    return parser.parse_args(argv)


def _load_external_queries(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        if "coords" in archive:
            data = archive["coords"]
        elif len(archive.files) == 1:
            data = archive[archive.files[0]]
        else:
            raise KeyError(f"NPZ archive {path} must contain a 'coords' dataset or exactly one array.")
    else:
        raise ValueError(f"Unsupported query file extension '{path.suffix}'. Expected .npy or .npz.")

    coords = np.asarray(data, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Query coordinates must have shape [N, 3]; received {coords.shape} from {path}.")
    return coords


def _prepare_cloud(args: argparse.Namespace) -> tuple[PointCloud, Optional[np.ndarray]]:
    cloud = read_point_cloud(Path(args.input))

    if args.voxel_size:
        cloud = voxel_downsample(cloud, args.voxel_size, reestimate_normals=args.estimate_normals)

    if args.max_points:
        cloud = random_subsample(cloud, args.max_points)

    if args.estimate_normals:
        cloud = ensure_normals(cloud, estimate=True)

    offset = None
    if args.center:
        centered_points, offset = center_points(cloud.points, method="mean")
        normals = cloud.normals
        colors = cloud.colors
        cloud = PointCloud(points=centered_points.astype(np.float32), normals=normals, colors=colors)

    return cloud, offset


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.use_cuda and not use_cuda:
        raise RuntimeError("CUDA requested via --use-cuda but no CUDA device is available.")

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    default_overrides = [
        "model=synthetic_model",
        "data=synthetic",
        "model.ckpt_path=",
    ]
    overrides = default_overrides + args.hydra_override

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config", overrides=overrides)

    if args.checkpoint:
        cfg.model.ckpt_path = args.checkpoint

    cloud, offset = _prepare_cloud(args)

    module = import_module("noksr.model")
    model_cls = getattr(module, cfg.model.network.module)
    model = model_cls(cfg).to(device)
    model.eval()

    ckpt_path = Path(cfg.model.ckpt_path) if cfg.model.ckpt_path else None
    if ckpt_path:
        if ckpt_path.is_file():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["state_dict"])
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    requires_normals = bool(cfg.model.network.use_normal)
    if requires_normals and (cloud.normals is None or cloud.normals.shape[0] != cloud.points.shape[0]):
        if not args.estimate_normals:
            raise ValueError("Model expects normals; rerun with --estimate-normals to compute them.")
        cloud = ensure_normals(cloud, estimate=True)

    features = build_point_features(
        cloud.points,
        normals=cloud.normals,
        colors=cloud.colors,
        use_xyz=bool(cfg.model.network.use_xyz),
        use_normal=bool(cfg.model.network.use_normal),
        use_color=bool(getattr(cfg.model.network, "use_color", False)),
    )

    batch = {
        "xyz": torch.from_numpy(cloud.points).float(),
        "point_features": torch.from_numpy(features).float(),
        "xyz_splits": torch.tensor([cloud.points.shape[0]], dtype=torch.long),
    }
    batch = move_batch_to_device(batch, device)

    encoder_outputs = encode_scene(model, batch, device=device)

    if args.query_file:
        query_coords = _load_external_queries(Path(args.query_file))
    else:
        query_coords = cloud.points

    query_xyz = torch.from_numpy(query_coords).float().to(device)
    if not args.disable_gradients:
        query_xyz = query_xyz.clone().detach().requires_grad_(True)

    result = query_sdf_and_gradient(
        model.sdf_decoder,
        encoder_outputs,
        query_xyz,
        compute_gradients=not args.disable_gradients,
    ).as_detached()
    print(summarize_query_result(result))

    if args.output:
        payload = {
            "coords": query_coords,
            "sdf": result.sdf.cpu().numpy(),
        }
        if result.gradients is not None:
            payload["gradients"] = result.gradients.cpu().numpy()
        if offset is not None:
            payload["center_offset"] = offset
        np.savez(args.output, **payload)
        print(f"[Inference] Results saved to {args.output}")


if __name__ == "__main__":
    main()
