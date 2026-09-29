"""
Shape-sensitive boundary metrics for the SSB project.

Definition:
- Shape score S is defined directly on valid ordered GT contour points.
- Those finite-S contour pixels themselves are used as the GT boundary
  for shape-sensitive evaluation.
- No projection of S onto another boundary representation is performed.
- Top-fraction and quartile splits are rank-based.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

try:
    from .boundary_metrics import (
        class_boundary_maps,
        boundary_f1_from_maps,
        _distance_to_true_pixels,
    )
except ImportError:
    from boundary_metrics import (
        class_boundary_maps,
        boundary_f1_from_maps,
        _distance_to_true_pixels,
    )


def _get_class_score_map(
    shape_score_maps: np.ndarray,
    class_id: int,
    num_classes: int,
) -> np.ndarray:
    """
    Return the S map for one semantic class.
    """
    arr = np.asarray(shape_score_maps)

    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)

    if arr.ndim == 3 and arr.shape[0] == num_classes:
        return arr[class_id].astype(np.float32, copy=False)

    raise ValueError(
        f"shape_score_maps must be [H,W] or [C,H,W], got {arr.shape}"
    )


def contour_boundary_from_score_map(
    score_map: np.ndarray,
) -> np.ndarray:
    """
    A finite value in score_map means that S was actually computed
    for this valid ordered GT contour point.

    Therefore these pixels directly define the GT contour used by
    shape-sensitive metrics.
    """
    return np.isfinite(score_map)


def _valid_scored_boundary(
    gt_contour: np.ndarray,
    score_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return valid GT contour mask and its S values.
    """
    gt_contour = np.asarray(gt_contour, dtype=bool)
    score_map = np.asarray(score_map)

    valid = gt_contour & np.isfinite(score_map)

    scores = score_map[
        valid
    ].astype(np.float64)

    return valid, scores


def _ranked_flat_indices(
    valid_mask: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return valid flat pixel indices and stable ascending S rank.

    Stable mergesort gives deterministic ordering when S values tie.
    """
    flat_indices = np.flatnonzero(valid_mask)

    if flat_indices.size != scores.size:
        raise ValueError(
            "valid pixel count and score count mismatch: "
            f"{flat_indices.size} vs {scores.size}"
        )

    order = np.argsort(
        scores,
        kind="mergesort",
    )

    return flat_indices, order


def select_top_fraction(
    gt_contour: np.ndarray,
    score_map: np.ndarray,
    top_fraction: float = 0.25,
) -> np.ndarray:
    """
    Select the highest-S contour points by rank.

    Rank-based selection avoids the unequal group size problem caused
    by quantile thresholds when many contour points share the same S.
    """
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError(
            "top_fraction must be in (0,1]."
        )

    valid, scores = _valid_scored_boundary(
        gt_contour,
        score_map,
    )

    selected = np.zeros_like(
        gt_contour,
        dtype=bool,
    )

    n = int(scores.size)

    if n == 0:
        return selected

    flat_indices, order = _ranked_flat_indices(
        valid,
        scores,
    )

    k = max(
        1,
        int(np.ceil(n * float(top_fraction))),
    )

    # order is ascending, so last k entries are highest-S.
    chosen = flat_indices[
        order[-k:]
    ]

    selected.flat[chosen] = True

    return selected


def _pred_boundary_assigned_to_selected_gt(
    pred_boundary: np.ndarray,
    full_gt_contour: np.ndarray,
    selected_gt_contour: np.ndarray,
) -> np.ndarray:
    """
    Assign each predicted boundary pixel to its nearest full GT contour
    point. Keep the prediction only when that nearest GT contour point
    belongs to the selected shape-sensitive subset.

    This assignment does NOT move or project S.
    It only determines which predicted boundary points belong to the
    selected GT contour region for SC-Precision / SC-BF1.
    """
    pb = np.asarray(
        pred_boundary,
        dtype=bool,
    )

    full_gt = np.asarray(
        full_gt_contour,
        dtype=bool,
    )

    selected_gt = np.asarray(
        selected_gt_contour,
        dtype=bool,
    )

    if (
        not pb.any()
        or not full_gt.any()
        or not selected_gt.any()
    ):
        return np.zeros_like(
            pb,
            dtype=bool,
        )

    _, indices = ndi.distance_transform_edt(
        ~full_gt,
        return_indices=True,
    )

    nearest_gt_is_selected = selected_gt[
        indices[0],
        indices[1],
    ]

    return pb & nearest_gt_is_selected


def shape_critical_bf1_from_maps(
    pred_boundary: np.ndarray,
    gt_contour: np.ndarray,
    score_map: np.ndarray,
    top_fraction: float = 0.25,
    tolerance_px: float = 2.0,
) -> Dict[str, float]:
    """
    Boundary F1 restricted to the highest-S GT contour points.
    """
    selected_gt = select_top_fraction(
        gt_contour,
        score_map,
        top_fraction,
    )

    if not selected_gt.any():
        return {
            "SC-BF1": np.nan,
            "SC-Precision": np.nan,
            "SC-Recall": np.nan,
            "n_sc_gt": 0,
            "n_sc_pred": 0,
        }

    selected_pred = (
        _pred_boundary_assigned_to_selected_gt(
            pred_boundary,
            gt_contour,
            selected_gt,
        )
    )

    f1, precision, recall = boundary_f1_from_maps(
        selected_pred,
        selected_gt,
        tolerance_px,
    )

    return {
        "SC-BF1": float(f1),
        "SC-Precision": float(precision),
        "SC-Recall": float(recall),
        "n_sc_gt": int(selected_gt.sum()),
        "n_sc_pred": int(selected_pred.sum()),
    }


def scle_from_maps(
    pred_boundary: np.ndarray,
    gt_contour: np.ndarray,
    score_map: np.ndarray,
    top_fraction: float = 0.25,
    max_distance_px: Optional[float] = 32.0,
) -> Dict[str, float]:
    """
    Shape-Critical Localization Error.

    Mean distance from the highest-S GT contour points to the nearest
    predicted class boundary.
    """
    selected_gt = select_top_fraction(
        gt_contour,
        score_map,
        top_fraction,
    )

    n = int(selected_gt.sum())

    if n == 0:
        return {
            "SCLE": np.nan,
            "n_sc_gt": 0,
        }

    d_to_pred = _distance_to_true_pixels(
        pred_boundary
    )

    distances = d_to_pred[
        selected_gt
    ].astype(np.float64)

    if max_distance_px is not None:
        distances = np.minimum(
            distances,
            float(max_distance_px),
        )

    value = (
        float(np.mean(distances))
        if np.isfinite(distances).all()
        else np.inf
    )

    return {
        "SCLE": value,
        "n_sc_gt": n,
    }


def quartile_boundary_localization_error(
    pred_boundary: np.ndarray,
    gt_contour: np.ndarray,
    score_map: np.ndarray,
    max_distance_px: Optional[float] = 32.0,
) -> Dict[str, Dict[str, float]]:
    """
    Rank valid GT contour points by S and divide them into four
    approximately equal-sized groups.

    Q1 = lowest S
    Q4 = highest S

    Localization error is GT-contour -> nearest predicted boundary.
    """
    valid, scores = _valid_scored_boundary(
        gt_contour,
        score_map,
    )

    if scores.size == 0:
        return {
            q: {
                "mean": np.nan,
                "median": np.nan,
                "count": 0,
            }
            for q in (
                "Q1",
                "Q2",
                "Q3",
                "Q4",
            )
        }

    d_to_pred = _distance_to_true_pixels(
        pred_boundary
    )

    distances = d_to_pred[
        valid
    ].astype(np.float64)

    if max_distance_px is not None:
        distances = np.minimum(
            distances,
            float(max_distance_px),
        )

    # Ascending S.
    order = np.argsort(
        scores,
        kind="mergesort",
    )

    # Group sizes differ by at most one point.
    groups = np.array_split(
        order,
        4,
    )

    result = {}

    for name, group in zip(
        ("Q1", "Q2", "Q3", "Q4"),
        groups,
    ):
        values = distances[group]

        if values.size == 0:
            result[name] = {
                "mean": np.nan,
                "median": np.nan,
                "count": 0,
            }
            continue

        finite = np.isfinite(values).all()

        result[name] = {
            "mean": (
                float(np.mean(values))
                if finite
                else np.inf
            ),
            "median": (
                float(np.median(values))
                if finite
                else np.inf
            ),
            "count": int(values.size),
        }

    return result


def evaluate_shape_sensitive_boundaries(
    pred: np.ndarray,
    gt: np.ndarray,
    shape_score_maps: np.ndarray,
    num_classes: int = 19,
    ignore_label: int = 255,
    top_fraction: float = 0.25,
    tolerance_px: float = 2.0,
    min_gt_boundary_pixels: int = 16,
    max_distance_px: Optional[float] = 32.0,
) -> Dict[str, object]:
    """
    Evaluate one image.

    General boundary extraction is used only for the prediction.
    Shape-sensitive GT is the finite-S ordered contour itself.
    """
    class_rows = []

    for c in range(num_classes):

        # We need the predicted class boundary.
        # The erosion-style GT boundary returned here is intentionally
        # NOT used for shape-sensitive GT evaluation.
        pred_boundary, _, _ = class_boundary_maps(
            pred,
            gt,
            c,
            ignore_label,
        )

        score_map = _get_class_score_map(
            shape_score_maps,
            c,
            num_classes,
        )

        if score_map.shape != gt.shape:
            raise ValueError(
                f"score map shape {score_map.shape} "
                f"!= gt shape {gt.shape}"
            )

        # Direct contour definition:
        # finite S == valid ordered GT contour point.
        gt_contour = contour_boundary_from_score_map(
            score_map
        )

        n_gt_contour = int(
            gt_contour.sum()
        )

        if n_gt_contour < min_gt_boundary_pixels:
            continue

        scbf = shape_critical_bf1_from_maps(
            pred_boundary,
            gt_contour,
            score_map,
            top_fraction,
            tolerance_px,
        )

        scle = scle_from_maps(
            pred_boundary,
            gt_contour,
            score_map,
            top_fraction,
            max_distance_px,
        )

        quartiles = (
            quartile_boundary_localization_error(
                pred_boundary,
                gt_contour,
                score_map,
                max_distance_px,
            )
        )

        class_rows.append(
            {
                "class_id": int(c),
                "n_gt_contour": n_gt_contour,
                **scbf,
                **scle,
                "quartiles": quartiles,
            }
        )

    if not class_rows:
        return {
            "SC-BF1": np.nan,
            "SCLE": np.nan,
            "Q1_error": np.nan,
            "Q2_error": np.nan,
            "Q3_error": np.nan,
            "Q4_error": np.nan,
            "valid_classes": [],
            "per_class": {},
        }

    q_means = {}

    for q in (
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ):
        values = [
            row["quartiles"][q]["mean"]
            for row in class_rows
            if (
                row["quartiles"][q]["count"] > 0
                and np.isfinite(
                    row["quartiles"][q]["mean"]
                )
            )
        ]

        q_means[f"{q}_error"] = (
            float(np.mean(values))
            if values
            else np.nan
        )

    return {
        "SC-BF1": float(
            np.nanmean(
                [
                    row["SC-BF1"]
                    for row in class_rows
                ]
            )
        ),

        "SCLE": float(
            np.nanmean(
                [
                    row["SCLE"]
                    for row in class_rows
                ]
            )
        ),

        **q_means,

        "valid_classes": [
            int(row["class_id"])
            for row in class_rows
        ],

        "per_class": {
            int(row["class_id"]): row
            for row in class_rows
        },
    }


class ShapeSensitiveBoundaryMeter:

    def __init__(
        self,
        num_classes=19,
        ignore_label=255,
        top_fraction=0.25,
        tolerance_px=2.0,
        min_gt_boundary_pixels=16,
        max_distance_px=32.0,
    ):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.top_fraction = top_fraction
        self.tolerance_px = tolerance_px
        self.min_gt_boundary_pixels = (
            min_gt_boundary_pixels
        )
        self.max_distance_px = max_distance_px

        self.reset()

    def reset(self):
        self.rows = []

    def update(
        self,
        pred,
        gt,
        shape_score_maps,
    ):
        result = (
            evaluate_shape_sensitive_boundaries(
                pred,
                gt,
                shape_score_maps,
                self.num_classes,
                self.ignore_label,
                self.top_fraction,
                self.tolerance_px,
                self.min_gt_boundary_pixels,
                self.max_distance_px,
            )
        )

        for c in result["valid_classes"]:
            self.rows.append(
                result["per_class"][c]
            )

    def compute(self):

        if not self.rows:
            return {
                "SC-BF1": np.nan,
                "SCLE": np.nan,
                "Q1_error": np.nan,
                "Q2_error": np.nan,
                "Q3_error": np.nan,
                "Q4_error": np.nan,
                "n_valid_class_instances": 0,
            }

        result = {
            "SC-BF1": float(
                np.nanmean(
                    [
                        row["SC-BF1"]
                        for row in self.rows
                    ]
                )
            ),

            "SCLE": float(
                np.nanmean(
                    [
                        row["SCLE"]
                        for row in self.rows
                    ]
                )
            ),

            "n_valid_class_instances":
                len(self.rows),
        }

        for q in (
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        ):
            values = [
                row["quartiles"][q]["mean"]
                for row in self.rows
                if (
                    row["quartiles"][q]["count"] > 0
                    and np.isfinite(
                        row["quartiles"][q]["mean"]
                    )
                )
            ]

            result[f"{q}_error"] = (
                float(np.mean(values))
                if values
                else np.nan
            )

        return result
