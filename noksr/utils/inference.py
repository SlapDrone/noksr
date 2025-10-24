from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class SDFQueryResult:
    """Container for decoder field queries."""

    sdf: torch.Tensor
    gradients: Optional[torch.Tensor]
    knn_time: float
    aggregation_time: float

    def as_detached(self) -> "SDFQueryResult":
        """Returns a detached copy to simplify serialization."""
        sdf = self.sdf.detach()
        grads = self.gradients.detach() if self.gradients is not None else None
        return SDFQueryResult(
            sdf=sdf,
            gradients=grads,
            knn_time=self.knn_time,
            aggregation_time=self.aggregation_time,
        )


def summarize_query_result(result: SDFQueryResult) -> str:
    """Formats a short textual summary of SDF/gradient statistics."""
    sdf = result.sdf.detach()
    sdf_stats = (
        sdf.min().item(),
        sdf.max().item(),
        sdf.mean().item(),
    )
    lines = [
        f"[Query] SDF stats -> min: {sdf_stats[0]:.6f}, max: {sdf_stats[1]:.6f}, mean: {sdf_stats[2]:.6f}",
    ]

    if result.gradients is not None:
        grad = result.gradients.detach()
        grad_norm = torch.linalg.norm(grad, dim=-1)
        g_stats = (
            grad_norm.min().item(),
            grad_norm.max().item(),
            grad_norm.mean().item(),
        )
        lines.append(
            f"[Query] Gradient L2 norm -> min: {g_stats[0]:.6f}, max: {g_stats[1]:.6f}, mean: {g_stats[2]:.6f}"
        )

    return "\n".join(lines)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Recursively moves tensors inside a Lightning batch dictionary."""
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, list):
            moved[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in value]
        else:
            moved[key] = value
    return moved


def build_point_transformer_input(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    grid_size: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Formats a collated Lightning batch for the PointTransformer encoder."""
    if "xyz_splits" not in batch:
        raise KeyError("Batch dictionary is missing 'xyz_splits'; cannot derive offsets.")

    offsets = torch.cumsum(batch["xyz_splits"].to(device=device, dtype=torch.long), dim=0)
    return {
        "feat": batch["point_features"].to(device),
        "offset": offsets,
        "grid_size": grid_size,
        "coord": batch["xyz"].to(device),
    }


def encode_scene(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    grid_size: float = 0.01,
) -> Any:
    """Runs the PointTransformer backbone and returns encoder outputs."""
    if not hasattr(model, "point_transformer"):
        raise AttributeError("Model does not expose a 'point_transformer' attribute.")

    pt_data = build_point_transformer_input(batch, device, grid_size=grid_size)
    return model.point_transformer(pt_data)


def query_sdf_and_gradient(
    decoder: torch.nn.Module,
    encoder_outputs: Any,
    query_xyz: torch.Tensor,
    *,
    compute_gradients: bool = True,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> SDFQueryResult:
    """Evaluates the decoder at arbitrary query positions and optionally returns gradients."""
    if query_xyz.ndim != 2 or query_xyz.size(-1) != 3:
        raise ValueError(f"Expected query_xyz to have shape [N, 3], received {tuple(query_xyz.shape)}.")

    if compute_gradients and not query_xyz.requires_grad:
        query_xyz = query_xyz.clone().detach().requires_grad_(True)

    with torch.set_grad_enabled(True):
        sdf_values, knn_time, aggregation_time = decoder(encoder_outputs, query_xyz)

        gradients: Optional[torch.Tensor] = None
        if compute_gradients:
            grad_outputs = torch.ones_like(sdf_values, device=sdf_values.device)
            gradients = torch.autograd.grad(
                outputs=sdf_values,
                inputs=query_xyz,
                grad_outputs=grad_outputs,
                retain_graph=retain_graph,
                create_graph=create_graph,
                allow_unused=False,
            )[0]

    return SDFQueryResult(
        sdf=sdf_values,
        gradients=gradients,
        knn_time=knn_time,
        aggregation_time=aggregation_time,
    )
