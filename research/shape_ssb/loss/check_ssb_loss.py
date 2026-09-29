#!/usr/bin/env python3

import torch

from research.shape_ssb.loss.ssb_loss import (
    compose_semantic_probability,
    soft_sobel_boundary,
    shape_sensitive_boundary_loss,
    ssb_warmup_weight,
)


def main():

    torch.manual_seed(
        2026
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("SSB differentiable-loss sanity")
    print("=" * 80)
    print("device:", device)

    # Small synthetic PEM-like output.
    B = 2
    Q = 100
    C = 19

    H_MASK = 64
    W_MASK = 128

    H_GT = 128
    W_GT = 256

    pred_logits = torch.randn(
        B,
        Q,
        C + 1,
        device=device,
        requires_grad=True,
    )

    pred_masks = torch.randn(
        B,
        Q,
        H_MASK,
        W_MASK,
        device=device,
        requires_grad=True,
    )

    # Synthetic GT distance + nonuniform shape weight.
    # Keep phi consistent with the actual target definition:
    #     phi = weight * distance
    distance = torch.rand(
        B,
        C,
        H_GT,
        W_GT,
        device=device,
    )

    weight = (
        1.0
        + 2.0
        * torch.rand(
            B,
            C,
            H_GT,
            W_GT,
            device=device,
        )
    )

    phi = weight * distance

    valid_classes = torch.zeros(
        B,
        C,
        dtype=torch.bool,
        device=device,
    )

    valid_classes[
        :,
        :10
    ] = True

    valid_mask = torch.ones(
        B,
        H_GT,
        W_GT,
        dtype=torch.bool,
        device=device,
    )

    # Fake ignore region.
    valid_mask[
        :,
        :10,
        :
    ] = False

    # --------------------------------------------------------
    # Semantic composition
    # --------------------------------------------------------

    sem_prob = compose_semantic_probability(
        pred_logits,
        pred_masks,
    )

    print(
        "semantic shape :",
        tuple(
            sem_prob.shape
        ),
    )

    assert sem_prob.shape == (
        B,
        C,
        H_MASK,
        W_MASK,
    )

    # --------------------------------------------------------
    # Soft boundary
    # --------------------------------------------------------

    boundary = soft_sobel_boundary(
        sem_prob
    )

    print(
        "boundary shape :",
        tuple(
            boundary.shape
        ),
    )

    assert boundary.shape == (
        B,
        C,
        H_MASK,
        W_MASK,
    )

    assert torch.isfinite(
        boundary
    ).all()

    # --------------------------------------------------------
    # Full SSB
    # --------------------------------------------------------

    loss = (
        shape_sensitive_boundary_loss(
            pred_logits=pred_logits,
            pred_masks=pred_masks,
            phi=phi,
            weight=weight,
            valid_classes=valid_classes,
            valid_mask=valid_mask,
        )
    )

    print(
        "loss           :",
        float(
            loss.detach()
        ),
    )

    assert torch.isfinite(
        loss
    )

    # Self-normalized SSB should be invariant to global weight scaling.
    scale = 3.0

    loss_scaled = (
        shape_sensitive_boundary_loss(
            pred_logits=pred_logits,
            pred_masks=pred_masks,
            phi=phi * scale,
            weight=weight * scale,
            valid_classes=valid_classes,
            valid_mask=valid_mask,
        )
    )

    print(
        "loss scaled    :",
        float(
            loss_scaled.detach()
        ),
    )

    assert torch.allclose(
        loss.detach(),
        loss_scaled.detach(),
        rtol=1e-5,
        atol=1e-6,
    ), (
        "Self-normalized SSB must be invariant "
        "to global scaling of weight."
    )

    loss.backward()

    logits_grad = (
        pred_logits.grad
    )

    masks_grad = (
        pred_masks.grad
    )

    assert logits_grad is not None
    assert masks_grad is not None

    logits_grad_abs = float(
        logits_grad.abs().sum()
    )

    masks_grad_abs = float(
        masks_grad.abs().sum()
    )

    print(
        "|grad logits|  :",
        logits_grad_abs,
    )

    print(
        "|grad masks|   :",
        masks_grad_abs,
    )

    assert logits_grad_abs > 0.0
    assert masks_grad_abs > 0.0

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    max_weight = 1.0

    checks = {
        0:
            0.0,
        999:
            0.0,
        1000:
            0.0,
        3000:
            0.5,
        5000:
            1.0,
        60000:
            1.0,
    }

    print()
    print("lambda_B warm-up")

    for iteration, expected in (
        checks.items()
    ):

        value = ssb_warmup_weight(
            iteration,
            max_weight=max_weight,
            warmup_start=1000,
            warmup_end=5000,
        )

        print(
            f"iter {iteration:5d}: "
            f"{value:.6f}"
        )

        assert abs(
            value - expected
        ) < 1e-8

    print()
    print(
        "All SSB differentiable-loss "
        "sanity checks passed."
    )


if __name__ == "__main__":
    main()
