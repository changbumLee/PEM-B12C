from __future__ import annotations
from typing import Dict, Sequence, Tuple
import numpy as np
from research.shape_ssb.loss.ssb_target import SSBTarget, build_ssb_target

def build_halo_ssb_target(
    halo_gt: np.ndarray,
    halo_px: int,
    crop_shape: Tuple[int, int],
    references: Dict[int, float],
    *,
    num_classes: int = 19,
    ignore_label: int = 255,
    min_contour_points: int = 16,
    radii: Sequence[int] = (2,4,8),
    d_max: float = 32.0,
    weight_lambda: float = 2.0,
    gamma: float = 2.0,
    tau: float = 0.25,
    mode: str = "full",
    compute_distance_fields: bool = True,
) -> SSBTarget:
    halo_gt = np.asarray(halo_gt)
    crop_h, crop_w = map(int, crop_shape)
    h = int(halo_px)
    expected = (crop_h + 2*h, crop_w + 2*h)
    if halo_gt.shape != expected:
        raise ValueError(f"halo_gt={halo_gt.shape}, expected={expected}")

    full = build_ssb_target(
        halo_gt,
        references=references,
        num_classes=num_classes,
        ignore_label=ignore_label,
        min_contour_points=min_contour_points,
        radii=radii,
        d_max=d_max,
        weight_lambda=weight_lambda,
        gamma=gamma,
        tau=tau,
        mode=mode,
        compute_distance_fields=compute_distance_fields,
    )

    ys = slice(h, h + crop_h)
    xs = slice(h, h + crop_w)

    center_boundary = np.ascontiguousarray(
        full.boundary[:, ys, xs]
    )

    center_valid_classes = (
        center_boundary.reshape(
            center_boundary.shape[0],
            -1,
        ).sum(axis=1) > 0
    )
    return SSBTarget(
        phi=np.ascontiguousarray(full.phi[:, ys, xs]),
        distance=np.ascontiguousarray(full.distance[:, ys, xs]),
        weight=np.ascontiguousarray(full.weight[:, ys, xs]),
        boundary=center_boundary,
        valid_classes=np.asarray(
            center_valid_classes,
            dtype=bool,
        ),
        valid_mask=np.ascontiguousarray(
            full.valid_mask[ys, xs]
        ),
    )
