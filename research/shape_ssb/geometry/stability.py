from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
try:
    from .contour import OrderedContour
    from .curvature import multiscale_normalized_curvature
except ImportError:
    from contour import OrderedContour
    from curvature import multiscale_normalized_curvature

_EPS = 1e-8

def stable_geometry_score(normalized_multiscale: Dict[int,np.ndarray], eps: float = _EPS):
    if not normalized_multiscale:
        raise ValueError("normalized_multiscale is empty")
    radii = sorted(normalized_multiscale)
    stack = np.stack([np.asarray(normalized_multiscale[r], dtype=np.float64) for r in radii], axis=0)
    valid = np.all(np.isfinite(stack), axis=0)

    n = stack.shape[1]
    K = np.full(n, np.nan, dtype=np.float64)
    P = np.full(n, np.nan, dtype=np.float64)
    S = np.full(n, np.nan, dtype=np.float64)

    if valid.any():
        x = stack[:, valid]
        mu = np.mean(x, axis=0)
        sigma = np.std(x, axis=0)
        persistence = np.exp(-sigma / (mu + eps))
        K[valid] = mu
        P[valid] = persistence
        S[valid] = mu * persistence

    return K.astype(np.float32), P.astype(np.float32), S.astype(np.float32)

def contour_shape_score(contour: OrderedContour, references: Dict[int,float],
                        radii: Sequence[int] = (2,4,8)):
    normalized = multiscale_normalized_curvature(contour, references, radii)
    K, P, S = stable_geometry_score(normalized)
    return {"normalized_curvature": normalized, "K": K, "P": P, "S": S}
