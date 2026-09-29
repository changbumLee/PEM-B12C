from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-8


# ============================================================
# PEM query output -> differentiable semantic map
# ============================================================

def compose_semantic_probability(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
) -> torch.Tensor:
    """
    PEM / MaskFormer-style differentiable semantic composition.

    pred_logits:
        [B, Q, C+1]

    pred_masks:
        [B, Q, H, W]

    Returns
    -------
    sem_prob:
        [B, C, H, W]

    Important:
        NO argmax.
        Gradient flows to both pred_logits and pred_masks.
    """

    if pred_logits.ndim != 3:
        raise ValueError(
            "pred_logits must be [B,Q,C+1], "
            f"got {tuple(pred_logits.shape)}"
        )

    if pred_masks.ndim != 4:
        raise ValueError(
            "pred_masks must be [B,Q,H,W], "
            f"got {tuple(pred_masks.shape)}"
        )

    if pred_logits.shape[0] != pred_masks.shape[0]:
        raise ValueError(
            "batch-size mismatch"
        )

    if pred_logits.shape[1] != pred_masks.shape[1]:
        raise ValueError(
            "query-count mismatch"
        )

    # Exclude no-object class.
    class_prob = F.softmax(
        pred_logits,
        dim=-1,
    )[..., :-1]

    mask_prob = pred_masks.sigmoid()

    sem_prob = torch.einsum(
        "bqc,bqhw->bchw",
        class_prob,
        mask_prob,
    )

    return sem_prob


# ============================================================
# Differentiable class-wise soft boundary
# ============================================================

def soft_sobel_boundary(
    sem_prob: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Class-wise differentiable Sobel magnitude.

        B_P^c = sqrt(Gx^2 + Gy^2 + eps)

    Uses grouped/depthwise convolution:
        one Sobel filter per semantic class.

    Replicate padding is used to avoid artificial
    zero-padding image-border gradients.
    """

    if sem_prob.ndim != 4:
        raise ValueError(
            "sem_prob must be [B,C,H,W], "
            f"got {tuple(sem_prob.shape)}"
        )

    _, c, _, _ = sem_prob.shape

    device = sem_prob.device
    dtype = sem_prob.dtype

    sobel_x = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0],
        ],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    kernel_x = sobel_x.repeat(
        c,
        1,
        1,
        1,
    )

    kernel_y = sobel_y.repeat(
        c,
        1,
        1,
        1,
    )

    x = F.pad(
        sem_prob,
        (1, 1, 1, 1),
        mode="replicate",
    )

    gx = F.conv2d(
        x,
        kernel_x,
        groups=c,
    )

    gy = F.conv2d(
        x,
        kernel_y,
        groups=c,
    )

    mag_sq = (
        gx.square()
        + gy.square()
    )

    boundary = (
        torch.sqrt(
            mag_sq + float(eps)
        )
        - float(eps) ** 0.5
    ).clamp_min(0.0)

    return boundary


# ============================================================
# Shape-Sensitive Boundary Loss
# ============================================================
def shape_sensitive_boundary_loss(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    phi: torch.Tensor,
    weight: torch.Tensor,
    valid_classes: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = _EPS,
) -> torch.Tensor:
    """
    Asymmetric GT-distance-potential SSB.

    For each valid class c:

                sum_xy B_P^c(x,y) Phi_G^c(x,y)
        L_c = ---------------------------------
                   sum_xy B_P^c(x,y) + eps

    Final loss:
        average over valid (batch,class) pairs.

    Parameters
    ----------
    pred_logits:
        [B,Q,C+1]

    pred_masks:
        [B,Q,h,w]

    phi:
        [B,C,H,W]

    valid_classes:
        [B,C] bool

    valid_mask:
        [B,H,W] bool, optional.
        False pixels are excluded from BOTH numerator
        and denominator.

    Important:
        phi is a GT-side constant.
        No gradient should flow into phi.
    """

    if phi.ndim != 4:
        raise ValueError(
            "phi must be [B,C,H,W]"
        )
    if weight.ndim != 4:
        raise ValueError(
            "weight must be [B,C,H,W]"
        )
    if weight.shape != phi.shape:
        raise ValueError(
            "weight shape mismatch: "
            f"{tuple(weight.shape)} "
            f"vs phi {tuple(phi.shape)}"
        )
    if valid_classes.ndim != 2:
        raise ValueError(
            "valid_classes must be [B,C]"
        )

    b, c, h, w = phi.shape

    if valid_classes.shape != (
        b,
        c,
    ):
        raise ValueError(
            "valid_classes shape mismatch: "
            f"{tuple(valid_classes.shape)} "
            f"vs {(b,c)}"
        )

    sem_prob = compose_semantic_probability(
        pred_logits,
        pred_masks,
    )

    if sem_prob.shape[:2] != (
        b,
        c,
    ):
        raise ValueError(
            "semantic class shape mismatch: "
            f"sem_prob={tuple(sem_prob.shape)}, "
            f"phi={tuple(phi.shape)}"
        )

    # PEM mask output can be lower resolution.
    # Compare at GT / Phi resolution.
    if sem_prob.shape[-2:] != (
        h,
        w,
    ):
        sem_prob = F.interpolate(
            sem_prob,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

    boundary = soft_sobel_boundary(
        sem_prob
    )

    phi = phi.to(
        device=boundary.device,
        dtype=boundary.dtype,
    )
    weight = weight.to(
        device=boundary.device,
        dtype=boundary.dtype,
    )
    valid_classes = valid_classes.to(
        device=boundary.device,
        dtype=torch.bool,
    )

    if valid_mask is not None:

        if valid_mask.shape != (
            b,
            h,
            w,
        ):
            raise ValueError(
                "valid_mask must be [B,H,W], "
                f"got {tuple(valid_mask.shape)}"
            )

        spatial_valid = valid_mask.to(
            device=boundary.device,
            dtype=boundary.dtype,
        ).unsqueeze(1)

        boundary_for_loss = (
            boundary
            * spatial_valid
        )

    else:
        boundary_for_loss = boundary

    # numerator = (
    #     boundary_for_loss
    #     * phi
    # ).sum(
    #     dim=(-2, -1)
    # )

    # denominator = (
    #     boundary_for_loss.sum(
    #         dim=(-2, -1)
    #     )
    #     + float(eps)
    # )
    #loss change no1********************************************
    # numerator = (
    #     boundary_for_loss
    #     * phi
    # ).sum(
    #     dim=(-2, -1)
    # )

    # weighted_boundary = (
    #     boundary_for_loss
    #     * weight
    # )

    # denominator = (
    #     weighted_boundary.sum(
    #         dim=(-2, -1)
    #     )
    #     + float(eps)
    # )

    # class_loss = (
    #     numerator
    #     / denominator
    # )
# ------------------------------------------------------------
# Self-normalized mean + shape-contrastive localization
#
# phi = W * D'
#
# L_ssb = L_mean + beta * L_contrast
# ------------------------------------------------------------
# ------------------------------------------------------------
# Self-normalized mean + shape-weighted GT boundary coverage
# ------------------------------------------------------------

    # ------------------------------------------------------------
    # Self-normalized mean + shape-aware tolerance hinge
    #
    # phi = W * D'
    # D'  = normalized GT distance in [0, 1]
    # ------------------------------------------------------------

    positive_weight = weight > 0.0

    distance_norm = torch.where(
        positive_weight,
        phi / weight.clamp_min(float(eps)),
        torch.zeros_like(phi),
    )

    # ============================================================
    # 1) Self-normalized P -> GT localization
    # ============================================================

    weighted_boundary = (
        boundary_for_loss
        * weight
    )

    mean_numerator = (
        weighted_boundary
        * distance_norm
    ).sum(
        dim=(-2, -1)
    )

    mean_denominator = (
        weighted_boundary.sum(
            dim=(-2, -1)
        )
        + float(eps)
    )

    mean_loss = (
        mean_numerator
        / mean_denominator
    )

    class_loss = mean_loss
    
    valid = valid_classes

    if not torch.any(valid):
        # Keep graph connected.
        return (
            pred_logits.sum()
            + pred_masks.sum()
        ) * 0.0

    loss = class_loss[
        valid
    ].mean()

    return loss


# ============================================================
# lambda_B warm-up
# ============================================================

def ssb_warmup_weight(
    iteration: int,
    max_weight: float,
    warmup_start: int = 1000,
    warmup_end: int = 5000,
) -> float:
    """
    0                 : iter < warmup_start
    linear ramp       : warmup_start <= iter < warmup_end
    max_weight        : iter >= warmup_end
    """

    iteration = int(
        iteration
    )

    if warmup_end <= warmup_start:
        raise ValueError(
            "warmup_end must be > warmup_start"
        )

    if iteration < warmup_start:
        return 0.0

    if iteration >= warmup_end:
        return float(
            max_weight
        )

    alpha = (
        iteration
        - warmup_start
    ) / float(
        warmup_end
        - warmup_start
    )

    return (
        float(max_weight)
        * alpha
    )


class SSBLoss(nn.Module):
    def __init__(
        self,
        eps: float = _EPS,
    ):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        pred_logits,
        pred_masks,
        phi,
        weight,
        valid_classes,
        valid_mask=None,
    ):
        return shape_sensitive_boundary_loss(
            pred_logits=pred_logits,
            pred_masks=pred_masks,
            phi=phi,
            weight=weight,
            valid_classes=valid_classes,
            valid_mask=valid_mask,
            eps=self.eps,
        )
