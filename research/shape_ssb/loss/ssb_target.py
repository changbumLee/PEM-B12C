from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from scipy import ndimage as ndi

from research.shape_ssb.geometry.shape_score import (
    build_shape_score_maps,
    build_curvature_score_maps,
)
from research.shape_ssb.geometry.contour import (
    ignore_adjacent_mask,
)
from research.shape_ssb.loss.ssb_variant import (
    select_variant_boundary_weight,
)
_EPS = 1e-8

@dataclass
class SSBTarget:
    phi: np.ndarray
    distance: np.ndarray
    weight: np.ndarray
    boundary: np.ndarray
    valid_classes: np.ndarray
    valid_mask: np.ndarray

def shape_weight_from_score(
    score: np.ndarray,
    weight_lambda: float,
    gamma: float,
    tau: float,
    eps: float = _EPS,
) -> np.ndarray:
    """
    w_i = 1 + lambda * S_i^gamma /
                    (S_i^gamma + tau^gamma)

    score:
        finite only on valid GT contour pixels.
    """

    s = np.asarray(
        score,
        dtype=np.float64,
    )

    out = np.full(
        s.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = np.isfinite(s)

    if not valid.any():
        return out.astype(
            np.float32
        )

    if tau <= 0:
        raise ValueError(
            "tau must be > 0"
        )

    sv = np.clip(
        s[valid],
        0.0,
        None,
    )

    numerator = np.power(
        sv,
        gamma,
    )

    denominator = (
        numerator
        + float(tau) ** float(gamma)
        + eps
    )

    out[valid] = (
        1.0
        + float(weight_lambda)
        * numerator
        / denominator
    )

    return out.astype(
        np.float32
    )


def build_ssb_target(
    gt: np.ndarray,
    references: Dict[int, float],
    num_classes: int = 19,
    ignore_label: int = 255,
    min_contour_points: int = 16,
    radii: Sequence[int] = (2, 4, 8),
    d_max: float = 32.0,
    weight_lambda: float = 2.0,
    gamma: float = 2.0,
    tau: float = 0.25,
    mode: str = "full",
    curvature_tau: float = 0.25,
    compute_distance_fields: bool = True,
) -> SSBTarget:
    """
    Build class-wise GT distance potential:

        Phi_G^c = Wbar_G^c * (D_G^c / D_max)

    where:
        D_G^c   = truncated distance to valid GT contour
        D_G^c / D_max = normalized distance in [0, 1]
        Wbar_G  = nearest valid GT contour point's shape weight
    Important:
    - ignore_label is excluded
    - only finite-S contour points are used
    - class-aware
    - nearest-boundary propagation uses EDT indices
    """

    gt = np.asarray(
        gt,
        dtype=np.int32,
    )

    if gt.ndim != 2:
        raise ValueError(
            f"gt must be [H,W], got {gt.shape}"
        )

    if d_max <= 0:
        raise ValueError(
            "d_max must be > 0"
        )

    h, w = gt.shape

    # ---------------------------------------------------------
    # S map
    #
    # finite only on valid ordered contour pixels
    # ---------------------------------------------------------

    score_maps = build_shape_score_maps(
        gt,
        references=references,
        num_classes=num_classes,
        ignore_label=ignore_label,
        min_contour_points=min_contour_points,
        radii=radii,
    )
    curvature_score_maps = None

    if mode == "curvature":
        curvature_score_maps = (
            build_curvature_score_maps(
                gt,
                references=references,
                num_classes=num_classes,
                ignore_label=ignore_label,
                min_contour_points=min_contour_points,
                radii=radii,
            )
        )

        if curvature_score_maps.shape != (
            num_classes,
            h,
            w,
        ):
            raise RuntimeError(
                "Unexpected curvature score map shape: "
                f"{curvature_score_maps.shape}"
            )
    if score_maps.shape != (
        num_classes,
        h,
        w,
    ):
        raise RuntimeError(
            "Unexpected score map shape: "
            f"{score_maps.shape}"
        )

    phi = np.zeros(
        (num_classes, h, w),
        dtype=np.float32,
    )

    distance_maps = np.zeros_like(
        phi
    )

    weight_maps = np.zeros_like(
        phi
    )

    boundary_maps = np.zeros(
        (num_classes, h, w),
        dtype=bool,
    )

    valid_classes = np.zeros(
        num_classes,
        dtype=bool,
    )

    # Exclude ignore itself AND its 1-pixel adjacency.
    # This is consistent with the contour-geometry definition.
    ignore_adjacent = ignore_adjacent_mask(
        gt,
        ignore_label=ignore_label,
    )

    valid_pixel_mask = ~ignore_adjacent

    for c in range(
        num_classes
    ):
        s_map = score_maps[c]

        boundary = np.isfinite(
            s_map
        )

        # No valid geometry boundary for this class.
        if not boundary.any():
            continue

        valid_classes[c] = True
        boundary_maps[c] = boundary

        # -----------------------------------------------------
        # Shape-sensitive contour weights
        # -----------------------------------------------------

        # boundary_weight = (
        #     shape_weight_from_score(
        #         s_map,
        #         weight_lambda=weight_lambda,
        #         gamma=gamma,
        #         tau=tau,
        #     )
        # )
        # 기존 V3 계산 — 절대 변경하지 않음
        full_boundary_weight = shape_weight_from_score(
            s_map,
            weight_lambda=weight_lambda,
            gamma=gamma,
            tau=tau,
        )

        if mode == "curvature":
            curvature_score = (
                curvature_score_maps[c]
            )
        else:
            curvature_score = s_map

        boundary_weight = (
            select_variant_boundary_weight(
                mode,
                full_weight=full_boundary_weight,
                curvature_score=curvature_score,
                curvature_weight_lambda=weight_lambda,
                curvature_gamma=gamma,
                curvature_tau=curvature_tau,
            )
        )
                # -----------------------------------------------------
        # Fast CSM-only path
        #
        # CSM-B4/B5 only needs shape weight ON the actual
        # GT contour.  In the full EDT path, every contour
        # pixel has distance=0 and its nearest boundary point
        # is itself, therefore:
        #
        #     nearest_weight[boundary]
        #         == boundary_weight[boundary]
        #
        # We can preserve exactly the contour values while
        # skipping EDT / nearest-index propagation / Phi.
        # -----------------------------------------------------
        if not compute_distance_fields:
            boundary_values = boundary_weight[boundary]

            if not np.isfinite(boundary_values).all():
                raise RuntimeError(
                    f"class {c}: non-finite contour weight "
                    "in fast CSM geometry path"
                )

            weight_maps[c, boundary] = (
                boundary_values.astype(
                    np.float32,
                    copy=False,
                )
            )

            # phi and distance_maps intentionally remain zero.
            # They are not consumed when MODEL.SSB.ENABLED=False.
            continue
        # -----------------------------------------------------
        # EDT
        #
        # scipy EDT computes distance to nearest zero.
        # Therefore:
        #
        # boundary pixel   -> 0
        # non-boundary     -> 1
        #
        # distance_transform_edt(~boundary)
        # -----------------------------------------------------

        dist, nearest_indices = (
            ndi.distance_transform_edt(
                ~boundary,
                return_indices=True,
            )
        )

        dist = np.minimum(
            dist,
            float(d_max),
        ).astype(
            np.float32
        )

        nearest_y = nearest_indices[0]
        nearest_x = nearest_indices[1]

        nearest_weight = (
            boundary_weight[
                nearest_y,
                nearest_x,
            ]
        )

        # Every nearest location should correspond to
        # a finite boundary weight.
        bad_nearest = (
            valid_pixel_mask
            & (~np.isfinite(nearest_weight))
        )

        if bad_nearest.any():
            raise RuntimeError(
                f"class {c}: nearest-boundary "
                "weight propagation produced NaN"
            )

        # Ignore pixels receive no supervision.
        dist[~valid_pixel_mask] = 0.0

        nearest_weight = (
            nearest_weight.astype(
                np.float32,
                copy=False,
            )
        )

        nearest_weight[
            ~valid_pixel_mask
        ] = 0.0

        dist_norm = (
            dist
            / float(d_max)
        )

        class_phi = (
            nearest_weight
            * dist_norm
        )
        # Important sanity:
        # exactly zero on valid GT boundary.
        if not np.allclose(
            class_phi[boundary],
            0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"class {c}: Phi_G is not zero "
                "on GT boundary"
            )

        phi[c] = class_phi
        distance_maps[c] = dist
        weight_maps[c] = nearest_weight

    return SSBTarget(
        phi=phi,
        distance=distance_maps,
        weight=weight_maps,
        boundary=boundary_maps,
        valid_classes=valid_classes,
        valid_mask=valid_pixel_mask.astype(
            bool,
            copy=False,
        ),
    )
