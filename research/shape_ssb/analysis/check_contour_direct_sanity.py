#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from PIL import Image

from research.shape_ssb.geometry.contour import (
    extract_class_contours,
    ignore_adjacent_mask,
    rasterize_contours,
)
from research.shape_ssb.geometry.shape_score import (
    load_curvature_references,
    build_shape_score_maps,
)


NUM_CLASSES = 19
IGNORE_LABEL = 255
MIN_CONTOUR_POINTS = 16
RADII = (2, 4, 8)

CITYSCAPES_ROOT = Path("datasets/cityscapes")

Q95_PATH = Path(
    "research/shape_ssb/geometry/curvature_reference_q95.json"
)

# 우선 2장만 검사.
GT_FILES = [
    CITYSCAPES_ROOT
    / "gtFine/train/aachen/"
      "aachen_000000_000019_gtFine_labelTrainIds.png",

    CITYSCAPES_ROOT
    / "gtFine/train/aachen/"
      "aachen_000001_000019_gtFine_labelTrainIds.png",
]


def main():

    references = load_curvature_references(
        Q95_PATH
    )

    print("=" * 80)
    print("Contour-direct shape-score sanity check")
    print("=" * 80)
    print("Q95:", references)
    print()

    total_finite = 0
    total_wrong_class = 0
    total_ignore_adjacent = 0
    total_not_on_extracted_contour = 0

    for gt_path in GT_FILES:

        if not gt_path.exists():
            raise FileNotFoundError(
                f"GT file not found: {gt_path}"
            )

        gt = np.asarray(
            Image.open(gt_path),
            dtype=np.int32,
        )

        print("-" * 80)
        print("GT:", gt_path)
        print("shape:", gt.shape)

        score_maps = build_shape_score_maps(
            gt,
            references=references,
            num_classes=NUM_CLASSES,
            ignore_label=IGNORE_LABEL,
            min_contour_points=MIN_CONTOUR_POINTS,
            radii=RADII,
        )

        assert score_maps.shape == (
            NUM_CLASSES,
            gt.shape[0],
            gt.shape[1],
        )

        ignore_adjacent = ignore_adjacent_mask(
            gt,
            IGNORE_LABEL,
        )

        image_finite = 0
        image_wrong_class = 0
        image_ignore_adjacent = 0
        image_not_on_contour = 0

        for c in range(NUM_CLASSES):

            finite = np.isfinite(
                score_maps[c]
            )

            n_finite = int(
                finite.sum()
            )

            if n_finite == 0:
                continue

            image_finite += n_finite

            # -------------------------------------------------
            # CHECK 1:
            # finite S point가 실제 class c pixel인가?
            # -------------------------------------------------
            wrong_class = (
                finite
                & (gt != c)
            )

            n_wrong_class = int(
                wrong_class.sum()
            )

            image_wrong_class += (
                n_wrong_class
            )

            # -------------------------------------------------
            # CHECK 2:
            # finite S point가 ignore 또는 ignore-adjacent에
            # 남아 있는가?
            # -------------------------------------------------
            bad_ignore = (
                finite
                & ignore_adjacent
            )

            n_bad_ignore = int(
                bad_ignore.sum()
            )

            image_ignore_adjacent += (
                n_bad_ignore
            )

            # -------------------------------------------------
            # CHECK 3:
            # finite S pixel이 실제 extract_class_contours()
            # 결과에 속하는가?
            # -------------------------------------------------
            contours = extract_class_contours(
                gt,
                c,
                ignore_label=IGNORE_LABEL,
                min_contour_points=MIN_CONTOUR_POINTS,
            )

            contour_mask = rasterize_contours(
                contours,
                gt.shape,
            )

            not_on_contour = (
                finite
                & (~contour_mask)
            )

            n_not_on_contour = int(
                not_on_contour.sum()
            )

            image_not_on_contour += (
                n_not_on_contour
            )

            print(
                f"class {c:2d}: "
                f"finite={n_finite:6d} | "
                f"wrong_class={n_wrong_class:4d} | "
                f"ignore_adj={n_bad_ignore:4d} | "
                f"not_on_contour={n_not_on_contour:4d}"
            )

        print()
        print(
            "Image summary:"
        )
        print(
            f"  finite S pixels        : "
            f"{image_finite}"
        )
        print(
            f"  wrong-class finite     : "
            f"{image_wrong_class}"
        )
        print(
            f"  ignore-adjacent finite : "
            f"{image_ignore_adjacent}"
        )
        print(
            f"  finite not on contour  : "
            f"{image_not_on_contour}"
        )

        # -----------------------------------------------------
        # Hard assertions
        # -----------------------------------------------------
        assert image_wrong_class == 0, (
            "FAIL: finite S pixels exist outside "
            "their semantic GT class."
        )

        assert image_ignore_adjacent == 0, (
            "FAIL: finite S pixels exist in "
            "ignore-adjacent regions."
        )

        assert image_not_on_contour == 0, (
            "FAIL: finite S pixels exist outside "
            "extract_class_contours() output."
        )

        total_finite += image_finite
        total_wrong_class += (
            image_wrong_class
        )
        total_ignore_adjacent += (
            image_ignore_adjacent
        )
        total_not_on_extracted_contour += (
            image_not_on_contour
        )

        print("  -> IMAGE PASSED")
        print()

    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"Total finite S pixels        : "
        f"{total_finite}"
    )
    print(
        f"Wrong-class finite pixels    : "
        f"{total_wrong_class}"
    )
    print(
        f"Ignore-adjacent finite pixels: "
        f"{total_ignore_adjacent}"
    )
    print(
        f"Finite not on contour        : "
        f"{total_not_on_extracted_contour}"
    )

    assert total_finite > 0

    assert total_wrong_class == 0
    assert total_ignore_adjacent == 0
    assert total_not_on_extracted_contour == 0

    print()
    print(
        "CONTOUR-DIRECT SANITY CHECK PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
