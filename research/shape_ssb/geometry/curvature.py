from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
try:
    from .contour import OrderedContour
except ImportError:
    from contour import OrderedContour

_EPS = 1e-8

def turning_curvature(contour: OrderedContour, radius: int, eps: float = _EPS) -> np.ndarray:
    pts = np.asarray(contour.points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"contour.points must be [N,2], got {pts.shape}")
    if radius < 1:
        raise ValueError("radius must be >= 1")

    n = len(pts)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2 * radius + 1:
        return out.astype(np.float32)

    if contour.closed:
        idx = np.arange(n)
        prev_idx = (idx - radius) % n
        next_idx = (idx + radius) % n
        valid_idx = idx
    else:
        valid_idx = np.arange(radius, n-radius)
        prev_idx = valid_idx - radius
        next_idx = valid_idx + radius

    p_prev = pts[prev_idx]
    p_cur = pts[valid_idx]
    p_next = pts[next_idx]

    v_minus = p_cur - p_prev
    v_plus = p_next - p_cur
    norm_minus = np.linalg.norm(v_minus, axis=1)
    norm_plus = np.linalg.norm(v_plus, axis=1)
    cosang = np.sum(v_minus * v_plus, axis=1) / (norm_minus * norm_plus + eps)
    theta = np.arccos(np.clip(cosang, -1.0, 1.0))
    local_length = 0.5 * (norm_minus + norm_plus) + eps
    out[valid_idx] = theta / local_length
    return out.astype(np.float32)

def multiscale_raw_curvature(contour: OrderedContour,
                             radii: Sequence[int] = (2,4,8)) -> Dict[int, np.ndarray]:
    return {int(r): turning_curvature(contour, int(r)) for r in radii}

def normalize_curvature_global(raw_curvature: np.ndarray, reference_value: float,
                               eps: float = _EPS) -> np.ndarray:
    if reference_value <= 0:
        raise ValueError("reference_value must be > 0")
    x = np.asarray(raw_curvature, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(x)
    out[valid] = np.clip(x[valid] / (float(reference_value) + eps), 0.0, 1.0)
    return out.astype(np.float32)

def multiscale_normalized_curvature(contour: OrderedContour, references: Dict[int,float],
                                    radii: Sequence[int] = (2,4,8)) -> Dict[int,np.ndarray]:
    raw = multiscale_raw_curvature(contour, radii)
    out = {}
    for r in radii:
        r = int(r)
        if r not in references:
            raise KeyError(f"Missing curvature reference for radius {r}")
        out[r] = normalize_curvature_global(raw[r], float(references[r]))
    return out
