from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from scipy import ndimage as ndi
import cv2

@dataclass(frozen=True)
class OrderedContour:
    points: np.ndarray
    closed: bool

def ignore_adjacent_mask(gt: np.ndarray, ignore_label: int = 255) -> np.ndarray:
    gt = np.asarray(gt)
    if gt.ndim != 2:
        raise ValueError(f"gt must be [H,W], got {gt.shape}")
    ignore = gt == int(ignore_label)
    if not ignore.any():
        return np.zeros_like(ignore, dtype=bool)
    return ndi.binary_dilation(ignore, structure=np.ones((3,3), dtype=bool), iterations=1)

def _find_all_ordered_contours(mask: np.ndarray) -> List[np.ndarray]:
    result = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )
    contours = result[-2]
    return [c.reshape(-1,2).astype(np.float32, copy=False) for c in contours if c is not None and len(c) > 0]

def _split_valid_runs(points: np.ndarray, valid_flags: np.ndarray, min_points: int) -> List[OrderedContour]:
    pts = np.asarray(points, dtype=np.float32)
    valid = np.asarray(valid_flags, dtype=bool)
    if len(pts) != len(valid):
        raise ValueError("points/valid_flags length mismatch")
    n = len(pts)
    if n == 0:
        return []
    if valid.all():
        return [OrderedContour(pts, True)] if n >= min_points else []
    if not valid.any():
        return []

    start = int(np.flatnonzero(~valid)[0])
    pts_rot = np.concatenate([pts[start:], pts[:start]], axis=0)
    valid_rot = np.concatenate([valid[start:], valid[:start]], axis=0)
    padded = np.pad(valid_rot.astype(np.int8), (1,1), constant_values=0)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    out = []
    for s, e in zip(starts, ends):
        seg = pts_rot[s:e]
        if len(seg) >= min_points:
            out.append(OrderedContour(seg, False))
    return out

def extract_class_contours(gt: np.ndarray, class_id: int, ignore_label: int = 255,
                           min_contour_points: int = 16) -> List[OrderedContour]:
    gt = np.asarray(gt)
    if gt.ndim != 2:
        raise ValueError(f"gt must be [H,W], got {gt.shape}")
    class_mask = (gt == int(class_id))
    if not class_mask.any():
        return []

    invalid = ignore_adjacent_mask(gt, ignore_label)
    contours = _find_all_ordered_contours(class_mask)
    h, w = gt.shape
    out = []

    for contour in contours:
        xy = np.rint(contour).astype(np.int64)
        x = np.clip(xy[:,0], 0, w-1)
        y = np.clip(xy[:,1], 0, h-1)
        valid_flags = ~invalid[y, x]
        out.extend(_split_valid_runs(contour, valid_flags, int(min_contour_points)))
    return out

def rasterize_contours(contours: List[OrderedContour], shape: tuple[int,int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h,w), dtype=bool)
    for contour in contours:
        pts = contour.points
        if len(pts) == 0:
            continue
        xy = np.rint(pts).astype(np.int64)
        x = np.clip(xy[:,0], 0, w-1)
        y = np.clip(xy[:,1], 0, h-1)
        out[y, x] = True
    return out
