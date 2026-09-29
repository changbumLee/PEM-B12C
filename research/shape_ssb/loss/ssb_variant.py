from __future__ import annotations

import numpy as np


VALID_SSB_MODES = (
    "uniform",
    "curvature",
    "full",
)


def _saturating_weight(
    score: np.ndarray,
    *,
    weight_lambda: float,
    gamma: float,
    tau: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    w = 1 + lambda * score^gamma / (score^gamma + tau^gamma)

    Used for V2 only.

    IMPORTANT:
    V3 does NOT call this function.
    V3 uses the already-computed legacy/full weight unchanged.
    """
    score = np.asarray(score, dtype=np.float32)

    out = np.full(
        score.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = np.isfinite(score)

    if not np.any(valid):
        return out

    x = np.clip(
        score[valid],
        0.0,
        None,
    ).astype(np.float32, copy=False)

    x_gamma = np.power(
        x,
        float(gamma),
    )

    tau_gamma = float(tau) ** float(gamma)

    out[valid] = (
        1.0
        + float(weight_lambda)
        * (
            x_gamma
            / (
                x_gamma
                + tau_gamma
                + float(eps)
            )
        )
    ).astype(np.float32)

    return out


def select_variant_boundary_weight(
    mode: str,
    *,
    full_weight: np.ndarray,
    curvature_score: np.ndarray,
    curvature_weight_lambda: float,
    curvature_gamma: float,
    curvature_tau: float,
) -> np.ndarray:
    """
    V1 / V2 / V3 ablation selector.

    V1:
        Uniform distance localization
        boundary weight = 1

    V2:
        Curvature-only weighting
        score = K

    V3:
        Full SSB
        EXACT existing full_weight is returned.
        No recomputation, normalization, clipping,
        casting, copy, or transformation is performed.
    """

    mode = str(mode).lower()

    if mode not in VALID_SSB_MODES:
        raise ValueError(
            f"Unknown SSB mode: {mode}. "
            f"Expected one of {VALID_SSB_MODES}"
        )

    # --------------------------------------------------------
    # V3 — FULL SSB
    #
    # CRITICAL:
    # Keep the old/current V3 result EXACTLY as it is.
    # --------------------------------------------------------
    if mode == "full":
        return full_weight

    full_weight = np.asarray(full_weight)

    finite_boundary = np.isfinite(full_weight)

    # --------------------------------------------------------
    # V1 — UNIFORM
    # --------------------------------------------------------
    if mode == "uniform":
        out = np.full(
            full_weight.shape,
            np.nan,
            dtype=np.float32,
        )

        out[finite_boundary] = 1.0

        return out

    # --------------------------------------------------------
    # V2 — CURVATURE ONLY
    # --------------------------------------------------------
    curvature_weight = _saturating_weight(
        curvature_score,
        weight_lambda=curvature_weight_lambda,
        gamma=curvature_gamma,
        tau=curvature_tau,
    )

    # Use exactly the same valid boundary support as V3.
    curvature_weight[~finite_boundary] = np.nan

    return curvature_weight
