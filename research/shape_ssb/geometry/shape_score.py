# from __future__ import annotations
# import json
# from pathlib import Path
# from typing import Dict, Sequence
# import numpy as np
# try:
#     from .contour import extract_class_contours
#     from .stability import contour_shape_score
# except ImportError:
#     from contour import extract_class_contours
#     from stability import contour_shape_score

# def load_curvature_references(json_path: str | Path) -> Dict[int,float]:
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     raw = data.get("references", data)
#     refs = {}
#     for key, value in raw.items():
#         r = int(key[1:]) if isinstance(key, str) and key.startswith("r") else int(key)
#         refs[r] = float(value)
#     return refs

# def build_shape_score_maps(gt: np.ndarray, references: Dict[int,float],
#                            num_classes: int = 19, ignore_label: int = 255,
#                            min_contour_points: int = 16,
#                            radii: Sequence[int] = (2,4,8)) -> np.ndarray:
#     gt = np.asarray(gt)
#     if gt.ndim != 2:
#         raise ValueError(f"gt must be [H,W], got {gt.shape}")

#     h, w = gt.shape
#     score_maps = np.full((num_classes,h,w), np.nan, dtype=np.float32)

#     for c in range(num_classes):
#         contours = extract_class_contours(gt, c, ignore_label, min_contour_points)
#         if not contours:
#             continue

#         class_score = np.full((h,w), -np.inf, dtype=np.float32)

#         for contour in contours:
#             S = contour_shape_score(contour, references, radii)["S"]
#             xy = np.rint(contour.points).astype(np.int64)
#             x = np.clip(xy[:,0], 0, w-1)
#             y = np.clip(xy[:,1], 0, h-1)
#             valid = np.isfinite(S)
#             if valid.any():
#                 np.maximum.at(class_score, (y[valid], x[valid]), S[valid])

#         written = np.isfinite(class_score)
#         score_maps[c, written] = class_score[written]

#     return score_maps
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

try:
    from .contour import extract_class_contours
    from .stability import contour_shape_score
except ImportError:
    from contour import extract_class_contours
    from stability import contour_shape_score


def load_curvature_references(
    json_path: str | Path,
) -> Dict[int, float]:
    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    raw = data.get(
        "references",
        data,
    )

    refs = {}

    for key, value in raw.items():
        r = (
            int(key[1:])
            if isinstance(key, str)
            and key.startswith("r")
            else int(key)
        )

        refs[r] = float(value)

    return refs


def build_shape_score_maps(
    gt: np.ndarray,
    references: Dict[int, float],
    num_classes: int = 19,
    ignore_label: int = 255,
    min_contour_points: int = 16,
    radii: Sequence[int] = (2, 4, 8),
) -> np.ndarray:
    """
    Build V3 full stable geometry score maps.

    Score:
        S = K * P

    IMPORTANT:
    This is the existing V3 path.
    Keep its numerical behavior unchanged.
    """

    gt = np.asarray(gt)

    if gt.ndim != 2:
        raise ValueError(
            f"gt must be [H,W], got {gt.shape}"
        )

    h, w = gt.shape

    score_maps = np.full(
        (
            num_classes,
            h,
            w,
        ),
        np.nan,
        dtype=np.float32,
    )

    for c in range(
        num_classes
    ):
        contours = extract_class_contours(
            gt,
            c,
            ignore_label,
            min_contour_points,
        )

        if not contours:
            continue

        class_score = np.full(
            (
                h,
                w,
            ),
            -np.inf,
            dtype=np.float32,
        )

        for contour in contours:
            result = contour_shape_score(
                contour,
                references,
                radii,
            )

            S = result["S"]

            xy = np.rint(
                contour.points
            ).astype(
                np.int64
            )

            x = np.clip(
                xy[:, 0],
                0,
                w - 1,
            )

            y = np.clip(
                xy[:, 1],
                0,
                h - 1,
            )

            valid = np.isfinite(
                S
            )

            if valid.any():
                np.maximum.at(
                    class_score,
                    (
                        y[valid],
                        x[valid],
                    ),
                    S[valid],
                )

        written = np.isfinite(
            class_score
        )

        score_maps[
            c,
            written,
        ] = class_score[
            written
        ]

    return score_maps


def build_curvature_score_maps(
    gt: np.ndarray,
    references: Dict[int, float],
    num_classes: int = 19,
    ignore_label: int = 255,
    min_contour_points: int = 16,
    radii: Sequence[int] = (2, 4, 8),
) -> np.ndarray:
    """
    Build V2 curvature-only score maps.

    Score:
        K = mean normalized multi-scale curvature

    Cross-scale stability P is NOT applied.

    Important:
    - Uses exactly the same contour extraction
      and TRAIN-derived curvature references as V3.
    - Only the score cue differs:
          V2 -> K
          V3 -> S = K * P
    """

    gt = np.asarray(gt)

    if gt.ndim != 2:
        raise ValueError(
            f"gt must be [H,W], got {gt.shape}"
        )

    h, w = gt.shape

    curvature_maps = np.full(
        (
            num_classes,
            h,
            w,
        ),
        np.nan,
        dtype=np.float32,
    )

    for c in range(
        num_classes
    ):
        contours = extract_class_contours(
            gt,
            c,
            ignore_label,
            min_contour_points,
        )

        if not contours:
            continue

        class_curvature = np.full(
            (
                h,
                w,
            ),
            -np.inf,
            dtype=np.float32,
        )

        for contour in contours:
            result = contour_shape_score(
                contour,
                references,
                radii,
            )

            K = result["K"]

            xy = np.rint(
                contour.points
            ).astype(
                np.int64
            )

            x = np.clip(
                xy[:, 0],
                0,
                w - 1,
            )

            y = np.clip(
                xy[:, 1],
                0,
                h - 1,
            )

            valid = np.isfinite(
                K
            )

            if valid.any():
                np.maximum.at(
                    class_curvature,
                    (
                        y[valid],
                        x[valid],
                    ),
                    K[valid],
                )

        written = np.isfinite(
            class_curvature
        )

        curvature_maps[
            c,
            written,
        ] = class_curvature[
            written
        ]

    return curvature_maps