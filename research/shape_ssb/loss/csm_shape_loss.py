from __future__ import annotations

import torch
import torch.nn.functional as F


_EPS = 1e-8


# def csm_shape_aux_loss(
#     pred_logits: torch.Tensor,
#     target: torch.Tensor,
#     boundary: torch.Tensor,
#     valid_mask: torch.Tensor,
#     band_kernel: int = 3,
#     eps: float = _EPS,
# ) -> torch.Tensor:
#     """
#     Training-only shape supervision for the final CSM feature.

#     Args:
#         pred_logits:
#             [B, 1, h, w]
#             Raw logits from the CSM auxiliary head.

#         target:
#             [B, 1, H, W]
#             Class-agnostic shape-critical GT boundary target in [0, 1].

#         boundary:
#             [B, 1, H, W]
#             Binary GT semantic contour map.

#         valid_mask:
#             [B, H, W]
#             Valid spatial supervision mask.

#         band_kernel:
#             Odd kernel size used to construct a local supervision
#             band around the downsampled GT contour.
#     """

#     if pred_logits.ndim != 4:
#         raise ValueError(
#             "pred_logits must be [B,1,h,w], "
#             f"got {tuple(pred_logits.shape)}"
#         )

#     if pred_logits.shape[1] != 1:
#         raise ValueError(
#             "pred_logits must have one channel, "
#             f"got {pred_logits.shape[1]}"
#         )

#     if target.ndim != 4 or target.shape[1] != 1:
#         raise ValueError(
#             "target must be [B,1,H,W], "
#             f"got {tuple(target.shape)}"
#         )

#     if boundary.ndim != 4 or boundary.shape[1] != 1:
#         raise ValueError(
#             "boundary must be [B,1,H,W], "
#             f"got {tuple(boundary.shape)}"
#         )

#     if valid_mask.ndim != 3:
#         raise ValueError(
#             "valid_mask must be [B,H,W], "
#             f"got {tuple(valid_mask.shape)}"
#         )

#     if band_kernel < 1 or band_kernel % 2 == 0:
#         raise ValueError(
#             "band_kernel must be a positive odd integer."
#         )

#     out_size = pred_logits.shape[-2:]

#     # ------------------------------------------------------------
#     # Downsample GT while preserving thin/critical contours.
#     # ------------------------------------------------------------
#     target_low = F.adaptive_max_pool2d(
#         target.float(),
#         output_size=out_size,
#     )

#     boundary_low = F.adaptive_max_pool2d(
#         boundary.float(),
#         output_size=out_size,
#     )

#     # A low-resolution cell is valid only if all contributing
#     # full-resolution pixels are valid.
#     invalid_full = (
#         ~valid_mask.bool()
#     ).float().unsqueeze(1)

#     invalid_low = F.adaptive_max_pool2d(
#         invalid_full,
#         output_size=out_size,
#     )

#     valid_low = (
#         1.0
#         - invalid_low.clamp(
#             min=0.0,
#             max=1.0,
#         )
#     )

#     # ------------------------------------------------------------
#     # Build a small local band around the GT contour.
#     #
#     # Boundary pixels:
#     #   soft target = shape sensitivity
#     #
#     # Nearby non-boundary pixels:
#     #   target = 0
#     #
#     # Far background:
#     #   ignored
#     # ------------------------------------------------------------
#     boundary_band = F.max_pool2d(
#         boundary_low,
#         kernel_size=band_kernel,
#         stride=1,
#         padding=band_kernel // 2,
#     ).clamp(
#         min=0.0,
#         max=1.0,
#     )

#     supervision_mask = (
#         boundary_band
#         * valid_low
#     )

#     per_pixel_loss = (
#         F.binary_cross_entropy_with_logits(
#             pred_logits,
#             target_low,
#             reduction="none",
#         )
#     )

#     numerator = (
#         per_pixel_loss
#         * supervision_mask
#     ).sum()

#     denominator = (
#         supervision_mask.sum()
#         + float(eps)
#     )

#     return numerator / denominator



#b2
# def csm_shape_aux_loss(
#     pred_logits,
#     target,
#     boundary,
#     valid_mask,
#     band_kernel=9,
#     eps=1e-6,
# ):
#     """
#     pred_logits:
#         [B, 1, h, w] stride-4 CSM logits

#     target:
#         [B, 1, H, W] full-resolution shape-sensitive boundary target

#     boundary:
#         [B, 1, H, W] full-resolution GT boundary

#     valid_mask:
#         [B, H, W] or [B, 1, H, W]
#     """

#     pred = pred_logits

#     if pred.ndim != 4 or pred.shape[1] != 1:
#         raise ValueError(
#             f"pred must be [B,1,h,w], got {tuple(pred.shape)}"
#         )

#     if target.ndim != 4 or target.shape[1] != 1:
#         raise ValueError(
#             f"target must be [B,1,H,W], got {tuple(target.shape)}"
#         )

#     if boundary.ndim != 4 or boundary.shape[1] != 1:
#         raise ValueError(
#             f"boundary must be [B,1,H,W], got {tuple(boundary.shape)}"
#         )

#     if band_kernel < 1 or band_kernel % 2 == 0:
#         raise ValueError(
#             f"band_kernel must be positive odd, got {band_kernel}"
#         )

#     # ---------------------------------------------------------
#     # B2:
#     # preserve full-resolution GT geometry.
#     # Upsample prediction instead of max-pooling GT.
#     # ---------------------------------------------------------
#     pred_full = F.interpolate(
#         pred,
#         size=target.shape[-2:],
#         mode="bilinear",
#         align_corners=False,
#     )

#     target_full = target.to(
#         device=pred.device,
#         dtype=pred.dtype,
#     ).clamp_(0.0, 1.0)

#     boundary_full = boundary.to(
#         device=pred.device,
#         dtype=pred.dtype,
#     )

#     valid_full = valid_mask.to(
#         device=pred.device,
#     )

#     if valid_full.ndim == 3:
#         valid_full = valid_full.unsqueeze(1)

#     valid_full = valid_full.to(
#         dtype=pred.dtype,
#     )

#     # ---------------------------------------------------------
#     # Full-resolution narrow boundary band.
#     #
#     # Old kernel=3 at stride 4 corresponded roughly to
#     # a several-pixel neighborhood in the original image.
#     # Use k=9 here => radius 4 px.
#     # ---------------------------------------------------------
#     pad = band_kernel // 2

#     boundary_band = F.max_pool2d(
#         boundary_full,
#         kernel_size=band_kernel,
#         stride=1,
#         padding=pad,
#     )

#     supervision_mask = (
#         boundary_band
#         * valid_full
#     )

#     loss_map = F.binary_cross_entropy_with_logits(
#         pred_full,
#         target_full,
#         reduction="none",
#     )

#     numerator = (
#         loss_map
#         * supervision_mask
#     ).sum()

#     denominator = (
#         supervision_mask.sum()
#         + float(eps)
#     )

#     return numerator / denominator


# import torch
# from torch.nn import functional as F


def csm_shape_aux_loss(
    pred_logits,
    target,
    boundary,
    valid_mask,
    pixel_weight=None,
    band_kernel=9,
    eps=1e-6,
):
    """
    B6: One-Sided Shape Lower-Bound CSM loss.

    Target semantics are EXACTLY B2:

        T = B * S

    On GT contour pixels:
        p < T  -> same BCE gradient as B2 (push upward)
        p >= T -> zero auxiliary gradient

    On nearby non-contour pixels inside the kxk boundary band:
        ordinary BCE with target 0

    Far-background pixels are ignored.

    No new hyperparameter is introduced.
    """

    pred = pred_logits

    if pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError(
            "pred_logits must be [B,1,h,w], "
            f"got {tuple(pred.shape)}"
        )

    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(
            "target must be [B,1,H,W], "
            f"got {tuple(target.shape)}"
        )

    if boundary.ndim != 4 or boundary.shape[1] != 1:
        raise ValueError(
            "boundary must be [B,1,H,W], "
            f"got {tuple(boundary.shape)}"
        )

    if band_kernel < 1 or band_kernel % 2 == 0:
        raise ValueError(
            "band_kernel must be a positive odd integer, "
            f"got {band_kernel}"
        )

    # ---------------------------------------------------------
    # Preserve B2 full-resolution supervision.
    # Upsample prediction; never downsample GT contour target.
    # ---------------------------------------------------------
    pred_full = F.interpolate(
        pred,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    target_full = target.to(
        device=pred.device,
        dtype=pred.dtype,
    ).clamp(
        min=0.0,
        max=1.0,
    )

    boundary_full = boundary.to(
        device=pred.device,
        dtype=pred.dtype,
    )

    valid_full = valid_mask.to(
        device=pred.device,
    )

    if valid_full.ndim == 3:
        valid_full = valid_full.unsqueeze(1)

    if valid_full.ndim != 4:
        raise ValueError(
            "valid_mask must be [B,H,W] or [B,1,H,W], "
            f"got {tuple(valid_mask.shape)}"
        )

    valid_full = valid_full.to(
        dtype=pred.dtype,
    )

    # ---------------------------------------------------------
    # Exact same kxk full-resolution boundary band as B2.
    # ---------------------------------------------------------
    pad = band_kernel // 2

    boundary_band = F.max_pool2d(
        boundary_full,
        kernel_size=band_kernel,
        stride=1,
        padding=pad,
    ).clamp(
        min=0.0,
        max=1.0,
    )

    supervision_mask = (
        boundary_band
        * valid_full
    )

    contour_mask = (
        boundary_full > 0.5
    )

    # ---------------------------------------------------------
    # Ordinary B2 BCE map.
    #
    # Its logit gradient is:
    #     dL/dz = sigmoid(z) - T
    # ---------------------------------------------------------
    bce_map = (
        F.binary_cross_entropy_with_logits(
            pred_full,
            target_full,
            reduction="none",
        )
    )

    # ---------------------------------------------------------
    # B6 contour loss
    #
    # BCE(z,T) - H(T)
    # is Bernoulli KL divergence D(T || p).
    #
    # It is:
    #   0 at p == T
    #   positive away from T
    #
    # We activate it only when p < T.
    # Therefore the active gradient is EXACTLY the B2 gradient:
    #
    #     dL/dz = p - T
    #
    # but no downward gradient is applied when p >= T.
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Target entropy H(T).
    #
    # Compute explicitly in FP32 for AMP safety.
    # target_full can be FP16 under autocast, while log/log1p
    # are promoted to FP32. Indexed assignment therefore must
    # also use an FP32 destination.
    # ---------------------------------------------------------
    target_fp32 = target_full.float()

    entropy = torch.zeros_like(
        target_fp32,
        dtype=torch.float32,
    )

    interior_target = (
        (target_fp32 > 0.0)
        & (target_fp32 < 1.0)
    )

    if torch.any(interior_target):
        t = target_fp32[
            interior_target
        ]

        entropy[
            interior_target
        ] = -(
            t * torch.log(t)
            + (1.0 - t)
            * torch.log1p(-t)
        )

    # Match the BCE map dtype before subtraction.
    entropy = entropy.to(
        dtype=bce_map.dtype
    )

    contour_divergence = (
        bce_map - entropy
    ).clamp_min(0.0)

    prob_full = torch.sigmoid(
        pred_full
    )

    # Gate is intentionally non-differentiable:
    # it chooses whether the B2 contour gradient is allowed.
    active_contour = (
        prob_full.detach()
        < target_full
    )

    contour_loss = torch.where(
        active_contour,
        contour_divergence,
        torch.zeros_like(
            contour_divergence
        ),
    )

    # On non-contour band pixels, target is exactly zero and
    # ordinary B2 BCE is retained.
    loss_map = torch.where(
        contour_mask,
        contour_loss,
        bce_map,
    )

    # ---------------------------------------------------------
    # B6 requires B2 uniform pixel weighting.
    # Keep the argument for compatibility with maskformer_model.py,
    # but reject accidental B3/B4-style weights.
    # ---------------------------------------------------------
    if pixel_weight is not None:
        weight_full = pixel_weight.to(
            device=pred.device,
            dtype=pred.dtype,
        )

        if weight_full.shape != target_full.shape:
            raise ValueError(
                "pixel_weight shape mismatch: "
                f"{tuple(weight_full.shape)} vs "
                f"{tuple(target_full.shape)}"
            )

        if not torch.all(
            weight_full == 1.0
        ):
            raise RuntimeError(
                "B6 requires uniform pixel_weight == 1"
            )

    # IMPORTANT:
    # denominator stays EXACTLY B2-style.
    #
    # Do NOT normalize by the number of active contour pixels.
    # Otherwise gradient gating and loss-scale changes would be mixed.
    numerator = (
        loss_map
        * supervision_mask
    ).sum()

    denominator = (
        supervision_mask.sum()
        + float(eps)
    )

    return numerator / denominator
def csm_boundary_presence_loss(
    pred_logits,
    boundary,
    valid_mask,
    band_kernel=9,
    eps=1e-6,
):
    """
    B7 boundary-presence auxiliary loss.

    Target:
        GT contour            -> 1
        nearby non-contour    -> 0
        far background        -> ignored

    Prediction is upsampled to full GT resolution.
    """

    pred = pred_logits

    if pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError(
            "pred_logits must be [B,1,h,w], "
            f"got {tuple(pred.shape)}"
        )

    if boundary.ndim != 4 or boundary.shape[1] != 1:
        raise ValueError(
            "boundary must be [B,1,H,W], "
            f"got {tuple(boundary.shape)}"
        )

    if band_kernel < 1 or band_kernel % 2 == 0:
        raise ValueError(
            "band_kernel must be a positive odd integer, "
            f"got {band_kernel}"
        )

    pred_full = F.interpolate(
        pred,
        size=boundary.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    boundary_full = boundary.to(
        device=pred.device,
        dtype=pred.dtype,
    ).clamp(
        min=0.0,
        max=1.0,
    )

    valid_full = valid_mask.to(
        device=pred.device,
    )

    if valid_full.ndim == 3:
        valid_full = valid_full.unsqueeze(1)

    if valid_full.ndim != 4:
        raise ValueError(
            "valid_mask must be [B,H,W] or [B,1,H,W], "
            f"got {tuple(valid_mask.shape)}"
        )

    valid_full = valid_full.to(
        dtype=pred.dtype,
    )

    pad = band_kernel // 2

    boundary_band = F.max_pool2d(
        boundary_full,
        kernel_size=band_kernel,
        stride=1,
        padding=pad,
    ).clamp(
        min=0.0,
        max=1.0,
    )

    supervision_mask = (
        boundary_band
        * valid_full
    )

    loss_map = F.binary_cross_entropy_with_logits(
        pred_full,
        boundary_full,
        reduction="none",
    )

    numerator = (
        loss_map
        * supervision_mask
    ).sum()

    denominator = (
        supervision_mask.sum()
        + float(eps)
    )

    return numerator / denominator