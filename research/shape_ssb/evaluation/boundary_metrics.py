"""
Boundary metrics for semantic segmentation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy import ndimage as ndi

_EPS = 1e-12

@dataclass
class ClassBoundaryResult:
    class_id: int
    bf1: float
    precision: float
    recall: float
    boundary_iou: float
    n_pred_boundary: int
    n_gt_boundary: int

def _as_bool(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=bool)

def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    mask = _as_bool(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be [H,W], got {mask.shape}")
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask, structure=np.ones((3,3), dtype=bool), border_value=0)
    return mask & (~eroded)

def _ignore_adjacent_mask(gt: np.ndarray, ignore_label: int = 255) -> np.ndarray:
    ignore = (gt == ignore_label)
    if not ignore.any():
        return np.zeros_like(ignore, dtype=bool)
    return ndi.binary_dilation(ignore, structure=np.ones((3,3), dtype=bool), iterations=1)
def class_boundary_maps(
    pred: np.ndarray,
    gt: np.ndarray,
    class_id: int,
    ignore_label: int = 255,):
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError(
            f"pred and gt must both be [H,W] with same shape; "
            f"got pred={pred.shape}, gt={gt.shape}"
        )

    # Ignore 및 그 1-pixel 인접 영역은 평가하지 않는다.
    invalid = _ignore_adjacent_mask(gt, ignore_label)
    valid = ~invalid

    # IMPORTANT:
    # valid로 class mask를 먼저 자르면 ignore 경계가 새로운 물체 경계처럼
    # 생성될 수 있으므로, 원래 mask에서 boundary를 먼저 계산한다.
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)

    pred_boundary = _binary_boundary(pred_mask)
    gt_boundary = _binary_boundary(gt_mask)

    # 그 뒤 ignore-adjacent boundary point를 제거한다.
    pred_boundary &= valid
    gt_boundary &= valid

    return pred_boundary, gt_boundary, valid

def _distance_to_true_pixels(binary_map: np.ndarray) -> np.ndarray:
    binary_map = _as_bool(binary_map)
    if not binary_map.any():
        return np.full(binary_map.shape, np.inf, dtype=np.float32)
    return ndi.distance_transform_edt(~binary_map).astype(np.float32)

def boundary_f1_from_maps(pred_boundary: np.ndarray, gt_boundary: np.ndarray, tolerance_px: float = 2.0):
    pb = _as_bool(pred_boundary)
    gb = _as_bool(gt_boundary)
    n_p, n_g = int(pb.sum()), int(gb.sum())
    if n_p == 0 and n_g == 0:
        return 1.0, 1.0, 1.0
    if n_p == 0 or n_g == 0:
        return 0.0, 0.0, 0.0
    d_to_gt = _distance_to_true_pixels(gb)
    d_to_pred = _distance_to_true_pixels(pb)
    precision = float((d_to_gt[pb] <= tolerance_px).mean())
    recall = float((d_to_pred[gb] <= tolerance_px).mean())
    f1 = 0.0 if precision + recall <= _EPS else 2.0 * precision * recall / (precision + recall)
    return float(f1), float(precision), float(recall)

def _resolve_boundary_width_px(
    image_shape: Tuple[int, int],
    dilation_ratio: float = 0.02,
    width_px: int | None = None,
) -> int:
    if width_px is not None:
        if int(width_px) < 1:
            raise ValueError("width_px must be >= 1")
        return int(width_px)

    if dilation_ratio <= 0:
        raise ValueError("dilation_ratio must be > 0")

    h, w = image_shape
    diagonal = float(np.hypot(h, w))
    return max(1, int(round(dilation_ratio * diagonal)))


def mask_to_inner_boundary_band(mask: np.ndarray, width_px: int) -> np.ndarray:
    """
    Inner boundary band:
        B(mask, d) = mask - erode(mask, d)
    """
    mask = _as_bool(mask)

    if mask.ndim != 2:
        raise ValueError(f"mask must be [H,W], got {mask.shape}")
    if width_px < 1:
        raise ValueError("width_px must be >= 1")
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    eroded = ndi.binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        iterations=int(width_px),
        border_value=0,
    )
    return mask & (~eroded)

def boundary_iou_from_masks(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    dilation_ratio: float = 0.02,
    width_px: int | None = None,
    valid_mask: np.ndarray | None = None,
) -> float:

    pm = _as_bool(pred_mask)
    gm = _as_bool(gt_mask)

    if pm.shape != gm.shape or pm.ndim != 2:
        raise ValueError(
            f"pred_mask and gt_mask must both be [H,W] with same shape; "
            f"got pred={pm.shape}, gt={gm.shape}"
        )

    d = _resolve_boundary_width_px(
        image_shape=pm.shape,
        dilation_ratio=dilation_ratio,
        width_px=width_px,
    )

    # IMPORTANT:
    # raw masks에서 먼저 boundary band 생성
    pred_band = mask_to_inner_boundary_band(pm, d)
    gt_band = mask_to_inner_boundary_band(gm, d)

    # 그 다음 ignore-adjacent 제외
    if valid_mask is not None:
        valid = _as_bool(valid_mask)

        if valid.shape != pm.shape:
            raise ValueError(
                f"valid_mask shape {valid.shape} != mask shape {pm.shape}"
            )

        pred_band &= valid
        gt_band &= valid

    union = pred_band | gt_band

    if not union.any():
        return 1.0

    return float(
        (pred_band & gt_band).sum()
        / (union.sum() + _EPS)
    )
def evaluate_semantic_boundaries(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int = 19,
    ignore_label: int = 255,
    tolerance_px: float = 2.0,
    boundary_iou_ratio: float = 0.02,
    boundary_iou_width_px: int | None = None,
    min_gt_boundary_pixels: int = 16,
) -> Dict[str, object]:

    pred = np.asarray(pred)
    gt = np.asarray(gt)

    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError(
            f"pred and gt must both be [H,W] with same shape; "
            f"got pred={pred.shape}, gt={gt.shape}"
        )

    per_class = {}
    valid_results: List[ClassBoundaryResult] = []

    invalid = _ignore_adjacent_mask(
        gt,
        ignore_label,
    )
    valid = ~invalid

    for c in range(num_classes):

        # BF1용 boundary는 class_boundary_maps 재사용
        pb, gb, _ = class_boundary_maps(
            pred,
            gt,
            c,
            ignore_label,
        )

        n_pb = int(pb.sum())
        n_gb = int(gb.sum())

        if n_gb < min_gt_boundary_pixels:
            continue

        bf1, precision, recall = (
            boundary_f1_from_maps(
                pb,
                gb,
                tolerance_px=tolerance_px,
            )
        )

        # Boundary IoU는 raw class mask를 사용
        raw_pred_mask = (pred == c)
        raw_gt_mask = (gt == c)

        biou = boundary_iou_from_masks(
            raw_pred_mask,
            raw_gt_mask,
            dilation_ratio=boundary_iou_ratio,
            width_px=boundary_iou_width_px,
            valid_mask=valid,
        )

        r = ClassBoundaryResult(
            c,
            bf1,
            precision,
            recall,
            biou,
            n_pb,
            n_gb,
        )

        valid_results.append(r)

        per_class[c] = {
            "BF1": bf1,
            "BoundaryPrecision": precision,
            "BoundaryRecall": recall,
            "BoundaryIoU": biou,
            "n_pred_boundary": n_pb,
            "n_gt_boundary": n_gb,
        }

    if not valid_results:
        return {
            "BF1": np.nan,
            "BoundaryIoU": np.nan,
            "BoundaryPrecision": np.nan,
            "BoundaryRecall": np.nan,
            "valid_classes": [],
            "per_class": per_class,
        }

    return {
        "BF1": float(
            np.mean(
                [r.bf1 for r in valid_results]
            )
        ),
        "BoundaryIoU": float(
            np.mean(
                [r.boundary_iou for r in valid_results]
            )
        ),
        "BoundaryPrecision": float(
            np.mean(
                [r.precision for r in valid_results]
            )
        ),
        "BoundaryRecall": float(
            np.mean(
                [r.recall for r in valid_results]
            )
        ),
        "valid_classes": [
            r.class_id
            for r in valid_results
        ],
        "per_class": per_class,
    }

class SemanticBoundaryMeter:
    def __init__(
        self,
        num_classes=19,
        ignore_label=255,
        tolerance_px=2.0,
        boundary_iou_ratio=0.02,
        boundary_iou_width_px=None,
        min_gt_boundary_pixels=16,
    ):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.tolerance_px = tolerance_px
        self.boundary_iou_ratio = boundary_iou_ratio
        self.boundary_iou_width_px = boundary_iou_width_px
        self.min_gt_boundary_pixels = min_gt_boundary_pixels
        self.reset()

    def reset(self):
        self.rows = []

    def update(self, pred, gt):
        result = evaluate_semantic_boundaries(
            pred=pred,
            gt=gt,
            num_classes=self.num_classes,
            ignore_label=self.ignore_label,
            tolerance_px=self.tolerance_px,
            boundary_iou_ratio=self.boundary_iou_ratio,
            boundary_iou_width_px=self.boundary_iou_width_px,
            min_gt_boundary_pixels=self.min_gt_boundary_pixels,
        )
        for c in result["valid_classes"]:
            row = {"class_id": int(c)}
            row.update(result["per_class"][c])
            self.rows.append(row)

    def compute(self):
        if not self.rows:
            return {
                "BF1": np.nan,
                "BoundaryIoU": np.nan,
                "BoundaryPrecision": np.nan,
                "BoundaryRecall": np.nan,
                "n_valid_class_instances": 0,
            }

        return {
            "BF1": float(np.mean([r["BF1"] for r in self.rows])),
            "BoundaryIoU": float(np.mean([r["BoundaryIoU"] for r in self.rows])),
            "BoundaryPrecision": float(np.mean([r["BoundaryPrecision"] for r in self.rows])),
            "BoundaryRecall": float(np.mean([r["BoundaryRecall"] for r in self.rows])),
            "n_valid_class_instances": len(self.rows),
        }

