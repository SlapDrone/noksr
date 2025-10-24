from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


@dataclass
class PointCloud:
    """Lightweight container for point-cloud attributes."""

    points: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None


def read_point_cloud(path: Path) -> PointCloud:
    """
    Loads a point cloud from disk.

    Supports common formats handled by Open3D (PLY, PCD, XYZ, etc.) and
    LAS/LAZ via laspy (optional dependency).
    """
    suffix = path.suffix.lower()
    if suffix in {".las", ".laz"}:
        return _read_las_like(path)

    if suffix in {".xyz", ".xyzi", ".txt"}:
        return _read_ascii_points(path)

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"No points were loaded from {path}.")

    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None

    return PointCloud(points=points, normals=normals, colors=colors)


def _read_las_like(path: Path) -> PointCloud:
    try:
        import laspy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LAS/LAZ support requires the 'laspy' package. "
            "Install it via `pip install laspy[lazrs]` inside the inference environment."
        ) from exc

    try:
        with laspy.open(str(path)) as reader:
            header = reader.header
            points_count = header.point_count
            if points_count == 0:
                raise ValueError(f"The LAS/LAZ file {path} does not contain points.")
            las_points = reader.read()
    except laspy.LaspyException as exc:
        raise RuntimeError(f"Failed to read {path} with laspy: {exc}") from exc

    xyz = np.vstack([las_points.x, las_points.y, las_points.z]).T.astype(np.float32)

    colors = None
    dims = set(las_points.point_format.dimension_names)
    if {"red", "green", "blue"}.issubset(dims):
        color_stack = np.vstack([las_points.red, las_points.green, las_points.blue]).T.astype(np.float32)
        max_val = np.max(color_stack)
        if max_val > 0:
            color_stack /= max_val
        colors = color_stack

    return PointCloud(points=xyz, normals=None, colors=colors)


def _read_ascii_points(path: Path) -> PointCloud:
    try:
        data = np.loadtxt(path, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Unable to parse point cloud from ASCII file {path}: {exc}") from exc

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"ASCII point cloud {path} must contain at least three columns per row.")

    points = data[:, :3].astype(np.float32)
    colors = None
    # Treat a 4th column as intensity; map to grayscale color in [0, 1].
    if data.shape[1] >= 4:
        intensity = data[:, 3]
        if intensity.max() > intensity.min():
            norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        else:
            norm = np.zeros_like(intensity)
        colors = np.stack([norm, norm, norm], axis=1).astype(np.float32)

    return PointCloud(points=points, normals=None, colors=colors)


def ensure_normals(
    cloud: PointCloud,
    *,
    estimate: bool,
    radius: Optional[float] = None,
    max_nn: int = 30,
) -> PointCloud:
    """
    Ensures a point cloud has normals. If normals are missing and estimation is
    requested, compute them using Open3D.
    """
    if cloud.normals is not None and cloud.normals.shape[0] == cloud.points.shape[0]:
        return cloud

    if not estimate:
        raise ValueError("Input point cloud lacks normals and estimation is disabled.")

    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud.points.astype(np.float64))

    if radius is None:
        bounding_box = np.linalg.norm(cloud.points.max(axis=0) - cloud.points.min(axis=0))
        radius = bounding_box * 0.01 if bounding_box > 0 else 1.0

    o3d_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=max_nn)
    )
    try:
        o3d_cloud.orient_normals_consistent_tangent_plane(max(4, 2 * max_nn))
    except RuntimeError:
        # Fallback to simple orientation if the neighbourhood graph is too small.
        o3d_cloud.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])

    normals = np.asarray(o3d_cloud.normals, dtype=np.float32)
    return PointCloud(points=cloud.points, normals=normals, colors=cloud.colors)


def voxel_downsample(
    cloud: PointCloud,
    voxel_size: float,
    *,
    reestimate_normals: bool = True,
) -> PointCloud:
    """Voxel-downsamples the cloud via Open3D."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")

    base = o3d.geometry.PointCloud()
    base.points = o3d.utility.Vector3dVector(cloud.points.astype(np.float64))
    if cloud.colors is not None:
        base.colors = o3d.utility.Vector3dVector(cloud.colors.astype(np.float64))
    if cloud.normals is not None and not reestimate_normals:
        base.normals = o3d.utility.Vector3dVector(cloud.normals.astype(np.float64))

    down = base.voxel_down_sample(voxel_size)
    points = np.asarray(down.points, dtype=np.float32)

    colors = None
    if cloud.colors is not None and down.has_colors():
        colors = np.asarray(down.colors, dtype=np.float32)

    normals = None
    if cloud.normals is not None and not reestimate_normals and down.has_normals():
        normals = np.asarray(down.normals, dtype=np.float32)
    elif reestimate_normals:
        normals = None

    downsampled = PointCloud(points=points, normals=normals, colors=colors)
    if reestimate_normals:
        downsampled = ensure_normals(downsampled, estimate=True)
    return downsampled


def random_subsample(
    cloud: PointCloud,
    max_points: int,
    *,
    seed: Optional[int] = None,
) -> PointCloud:
    """Randomly subsamples the point cloud to at most `max_points`."""
    if cloud.points.shape[0] <= max_points:
        return cloud

    rng = np.random.default_rng(seed)
    indices = rng.choice(cloud.points.shape[0], size=max_points, replace=False)

    points = cloud.points[indices]
    normals = cloud.normals[indices] if cloud.normals is not None else None
    colors = cloud.colors[indices] if cloud.colors is not None else None

    return PointCloud(points=points, normals=normals, colors=colors)


def center_points(points: np.ndarray, method: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
    """
    Centers a point set. Returns the centered points and the offset that was subtracted.
    """
    if method not in {"mean", "median"}:
        raise ValueError("Centering method must be 'mean' or 'median'.")

    if method == "mean":
        offset = points.mean(axis=0)
    else:
        offset = np.median(points, axis=0)

    return points - offset, offset.astype(np.float32)


def build_point_features(
    points: np.ndarray,
    *,
    normals: Optional[np.ndarray],
    colors: Optional[np.ndarray],
    use_xyz: bool,
    use_normal: bool,
    use_color: bool,
) -> np.ndarray:
    """Constructs the feature matrix expected by the PointTransformer encoder."""
    features = []
    if use_color:
        if colors is None:
            warnings.warn("Model expects color channels but the input cloud lacks them; padding zeros.", RuntimeWarning)
            colors = np.zeros((points.shape[0], 3), dtype=np.float32)
        features.append(colors.astype(np.float32))

    if use_normal:
        if normals is None:
            raise ValueError("Model expects normals but none are provided or estimated.")
        features.append(normals.astype(np.float32))

    if use_xyz:
        features.append(points.astype(np.float32))

    if not features:
        return np.zeros((points.shape[0], 0), dtype=np.float32)

    return np.concatenate(features, axis=1)
