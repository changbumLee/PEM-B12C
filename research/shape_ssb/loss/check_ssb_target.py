#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from PIL import Image

from research.shape_ssb.geometry.shape_score import (
    load_curvature_references,
)

from research.shape_ssb.loss.ssb_target import (
    build_ssb_target,
)


GT_PATH = Path(
    "datasets/cityscapes/"
    "gtFine/train/aachen/"
    "aachen_000000_000019_gtFine_labelTrainIds.png"
)

REF_PATH = Path(
    "research/shape_ssb/geometry/"
    "curvature_reference_q95.json"
)


def main():

    refs = load_curvature_references(
        REF_PATH
    )

    gt = np.asarray(
        Image.open(
            GT_PATH
        ),
        dtype=np.int32,
    )

    target = build_ssb_target(
        gt,
        references=refs,
        num_classes=19,
        ignore_label=255,
        min_contour_points=16,
        radii=(2, 4, 8),
        d_max=32.0,
        weight_lambda=2.0,
        gamma=2.0,
        tau=0.25,
    )

    print("=" * 80)
    print("SSB target sanity")
    print("=" * 80)

    print(
        "GT shape:",
        gt.shape,
    )

    print(
        "Phi shape:",
        target.phi.shape,
    )

    print(
        "Valid classes:",
        np.flatnonzero(
            target.valid_classes
        ).tolist(),
    )

    print()

    for c in np.flatnonzero(
        target.valid_classes
    ):

        boundary = target.boundary[c]

        n_boundary = int(
            boundary.sum()
        )

        phi_boundary_max = float(
            np.max(
                target.phi[c][boundary]
            )
        )

        d_boundary_max = float(
            np.max(
                target.distance[c][boundary]
            )
        )

        class_valid = (
            gt != 255
        )

        max_dist = float(
            target.distance[c][class_valid].max()
        )

        finite_weight = (
            target.weight[c][class_valid]
        )

        positive_weight = finite_weight[
            finite_weight > 0
        ]

        if positive_weight.size > 0:
            w_min = float(
                positive_weight.min()
            )
            w_max = float(
                positive_weight.max()
            )
        else:
            w_min = np.nan
            w_max = np.nan

        print(
            f"class {c:2d} | "
            f"boundary={n_boundary:6d} | "
            f"phi@boundary max={phi_boundary_max:.8f} | "
            f"d@boundary max={d_boundary_max:.8f} | "
            f"d_max={max_dist:.4f} | "
            f"weight=[{w_min:.4f}, {w_max:.4f}]"
        )

        assert (
            phi_boundary_max
            <= 1e-6
        )

        assert (
            d_boundary_max
            <= 1e-6
        )

        assert (
            max_dist
            <= 32.0 + 1e-6
        )

        assert (
            w_min
            >= 1.0 - 1e-6
        )

        assert (
            w_max
            <= 3.0 + 1e-6
        )

    print()
    print(
        "All SSB target sanity checks passed."
    )


if __name__ == "__main__":
    main()
