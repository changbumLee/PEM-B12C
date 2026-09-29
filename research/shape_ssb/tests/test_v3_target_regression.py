from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from research.shape_ssb.geometry.shape_score import (
    build_shape_score_maps,
    load_curvature_references,
)
from research.shape_ssb.geometry.contour import (
    ignore_adjacent_mask,
)
from research.shape_ssb.loss.ssb_target import (
    build_ssb_target,
)


EPS = 1e-8

NUM_CLASSES = 19
IGNORE_LABEL = 255
MIN_CONTOUR_POINTS = 16
RADII = (2, 4, 8)
D_MAX = 32.0
WEIGHT_LAMBDA = 2.0
GAMMA = 2.0
TAU_V3 = 0.0644756630063057

REF_PATH = Path(
    "research/shape_ssb/geometry/"
    "curvature_reference_q95.json"
)

GT_ROOT = Path(
    "datasets/cityscapes/gtFine/train"
)


def legacy_shape_weight_from_score(
    score: np.ndarray,
    weight_lambda: float,
    gamma: float,
    tau: float,
    eps: float = EPS,
) -> np.ndarray:

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


def build_legacy_v3(
    gt: np.ndarray,
    references,
):
    gt = np.asarray(
        gt,
        dtype=np.int32,
    )

    h, w = gt.shape

    score_maps = build_shape_score_maps(
        gt,
        references=references,
        num_classes=NUM_CLASSES,
        ignore_label=IGNORE_LABEL,
        min_contour_points=MIN_CONTOUR_POINTS,
        radii=RADII,
    )

    if score_maps.shape != (
        NUM_CLASSES,
        h,
        w,
    ):
        raise RuntimeError(
            "Unexpected score map shape: "
            f"{score_maps.shape}"
        )

    phi = np.zeros(
        (
            NUM_CLASSES,
            h,
            w,
        ),
        dtype=np.float32,
    )

    distance_maps = np.zeros_like(
        phi
    )

    weight_maps = np.zeros_like(
        phi
    )

    boundary_maps = np.zeros(
        (
            NUM_CLASSES,
            h,
            w,
        ),
        dtype=bool,
    )

    valid_classes = np.zeros(
        NUM_CLASSES,
        dtype=bool,
    )

    ignore_adjacent = ignore_adjacent_mask(
        gt,
        ignore_label=IGNORE_LABEL,
    )

    valid_pixel_mask = ~ignore_adjacent

    for c in range(
        NUM_CLASSES
    ):
        s_map = score_maps[c]

        boundary = np.isfinite(
            s_map
        )

        if not boundary.any():
            continue

        valid_classes[c] = True
        boundary_maps[c] = boundary

        boundary_weight = (
            legacy_shape_weight_from_score(
                s_map,
                weight_lambda=WEIGHT_LAMBDA,
                gamma=GAMMA,
                tau=TAU_V3,
            )
        )

        dist, nearest_indices = (
            ndi.distance_transform_edt(
                ~boundary,
                return_indices=True,
            )
        )

        dist = np.minimum(
            dist,
            D_MAX,
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

        bad_nearest = (
            valid_pixel_mask
            & (~np.isfinite(nearest_weight))
        )

        if bad_nearest.any():
            raise RuntimeError(
                f"class {c}: nearest-boundary "
                "weight propagation produced NaN"
            )

        dist[
            ~valid_pixel_mask
        ] = 0.0

        nearest_weight = (
            nearest_weight.astype(
                np.float32,
                copy=False,
            )
        )

        nearest_weight[
            ~valid_pixel_mask
        ] = 0.0

        class_phi = (
            nearest_weight
            * dist
        )

        phi[c] = class_phi
        distance_maps[c] = dist
        weight_maps[c] = nearest_weight

    return {
        "phi": phi,
        "distance": distance_maps,
        "weight": weight_maps,
        "boundary": boundary_maps,
        "valid_classes": valid_classes,
        "valid_mask": valid_pixel_mask.astype(
            bool,
            copy=False,
        ),
    }


def assert_float_exact(
    name: str,
    old: np.ndarray,
    new: np.ndarray,
):

    np.testing.assert_allclose(
        old,
        new,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )

    finite = (
        np.isfinite(old)
        & np.isfinite(new)
    )

    if finite.any():
        max_diff = np.max(
            np.abs(
                old[finite]
                - new[finite]
            )
        )
    else:
        max_diff = 0.0

    print(
        f"PASS {name:15s} "
        f"max_abs_diff={max_diff}"
    )


def main():

    references = (
        load_curvature_references(
            REF_PATH
        )
    )

    gt_paths = sorted(
        GT_ROOT.glob(
            "*/*_gtFine_labelTrainIds.png"
        )
    )

    if len(gt_paths) == 0:
        raise RuntimeError(
            "No Cityscapes labelTrainIds "
            f"found under {GT_ROOT}"
        )

    gt_paths = gt_paths[:5]

    print(
        f"Testing {len(gt_paths)} GT images"
    )

    for index, gt_path in enumerate(
        gt_paths,
        start=1,
    ):

        print()
        print(
            "========================================"
        )
        print(
            f"[{index}/{len(gt_paths)}] "
            f"{gt_path}"
        )
        print(
            "========================================"
        )

        gt = np.asarray(
            Image.open(
                gt_path
            ),
            dtype=np.int32,
        )

        old = build_legacy_v3(
            gt,
            references,
        )

        new = build_ssb_target(
            gt,
            references=references,
            num_classes=NUM_CLASSES,
            ignore_label=IGNORE_LABEL,
            min_contour_points=MIN_CONTOUR_POINTS,
            radii=RADII,
            d_max=D_MAX,
            weight_lambda=WEIGHT_LAMBDA,
            gamma=GAMMA,
            tau=TAU_V3,
            mode="full",
            curvature_tau=0.25,
        )

        assert_float_exact(
            "phi",
            old["phi"],
            new.phi,
        )

        assert_float_exact(
            "distance",
            old["distance"],
            new.distance,
        )

        assert_float_exact(
            "weight",
            old["weight"],
            new.weight,
        )

        np.testing.assert_array_equal(
            old["boundary"],
            new.boundary,
        )

        print(
            "PASS boundary"
        )

        np.testing.assert_array_equal(
            old["valid_classes"],
            new.valid_classes,
        )

        print(
            "PASS valid_classes"
        )

        np.testing.assert_array_equal(
            old["valid_mask"],
            new.valid_mask,
        )

        print(
            "PASS valid_mask"
        )

    print()
    print(
        "========================================"
    )
    print(
        "ALL TESTS PASSED"
    )
    print(
        "LEGACY V3 == NEW mode='full'"
    )
    print(
        "Existing V3 lambda-search results "
        "remain valid."
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()