#!/usr/bin/env python3

from __future__ import annotations

import argparse

import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import ImageList

from pem import add_maskformer2_config
from train_net import Trainer

from research.shape_ssb.geometry.shape_score import (
    load_curvature_references,
)

from research.shape_ssb.loss.ssb_target import (
    build_ssb_target,
)

from research.shape_ssb.loss.ssb_loss import (
    compose_semantic_probability,
    soft_sobel_boundary,
    shape_sensitive_boundary_loss,
)


def build_cfg(
    config_file,
    checkpoint,
):
    cfg = get_cfg()

    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(
        config_file
    )

    cfg.MODEL.WEIGHTS = checkpoint

    # Sanity test only:
    # one augmented TRAIN sample.
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.freeze()

    return cfg


def get_augmented_gt(
    input_dict,
):
    """
    IMPORTANT:
    We need the TRAIN mapper's already-augmented semantic GT,
    not the original PNG, because it must correspond exactly
    to the augmented input image.
    """

    if "sem_seg" not in input_dict:
        raise KeyError(
            "TRAIN mapper output does not contain 'sem_seg'. "
            f"keys={list(input_dict.keys())}"
        )

    gt = input_dict[
        "sem_seg"
    ]

    if torch.is_tensor(gt):
        gt = (
            gt.detach()
            .cpu()
            .numpy()
        )
    else:
        gt = np.asarray(gt)

    if gt.ndim != 2:
        raise ValueError(
            "sem_seg must be [H,W], "
            f"got {gt.shape}"
        )

    return gt.astype(
        np.int32,
        copy=False,
    )


def raw_pem_forward(
    model,
    batched_inputs,
):
    """
    Equivalent to the raw-output part of
    MaskFormer.forward():

        image
        -> normalize
        -> ImageList
        -> backbone
        -> sem_seg_head
        -> pred_logits/pred_masks

    We stop BEFORE criterion/inference postprocessing.
    """

    images = [
        x["image"].to(
            model.device
        )
        for x in batched_inputs
    ]

    images = [
        (
            x - model.pixel_mean
        ) / model.pixel_std
        for x in images
    ]

    images = ImageList.from_tensors(
        images,
        model.size_divisibility,
    )

    features = model.backbone(
        images.tensor
    )

    outputs = model.sem_seg_head(
        features
    )

    if "pred_logits" not in outputs:
        raise KeyError(
            "pred_logits missing"
        )

    if "pred_masks" not in outputs:
        raise KeyError(
            "pred_masks missing"
        )

    return outputs, images


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--curvature-reference",
        default=(
            "research/shape_ssb/geometry/"
            "curvature_reference_q95.json"
        ),
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.25,
        help=(
            "TEMPORARY sanity value only. "
            "Not the final experimental tau."
        ),
    )

    args = parser.parse_args()

    cfg = build_cfg(
        args.config_file,
        args.checkpoint,
    )

    refs = (
        load_curvature_references(
            args.curvature_reference
        )
    )

    print("=" * 90)
    print(
        "REAL PEM + REAL TRAIN GT "
        "SSB SANITY"
    )
    print("=" * 90)

    print(
        "Checkpoint :",
        args.checkpoint,
    )

    print(
        "Train data :",
        list(
            cfg.DATASETS.TRAIN
        ),
    )

    print(
        "Q95        :",
        refs,
    )

    print(
        "tau        :",
        args.tau,
        "(TEMPORARY)",
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Trainer.build_model(
        cfg
    )

    DetectionCheckpointer(
        model
    ).load(
        args.checkpoint
    )

    # We need gradients, but do NOT want to update BN stats.
    model.eval()

    model.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # Real TRAIN loader
    # Includes the same resize/crop/flip distribution.
    # --------------------------------------------------------

    loader = (
        Trainer.build_train_loader(
            cfg
        )
    )

    batch = next(
        iter(loader)
    )

    # IMS_PER_BATCH was forced to 1.
    if len(batch) != 1:
        raise RuntimeError(
            f"Expected batch=1, got {len(batch)}"
        )

    input_dict = batch[0]

    print()
    print(
        "Input keys  :",
        sorted(
            input_dict.keys()
        ),
    )

    print(
        "Image shape :",
        tuple(
            input_dict[
                "image"
            ].shape
        ),
    )

    gt = get_augmented_gt(
        input_dict
    )

    print(
        "GT shape    :",
        gt.shape,
    )

    print(
        "GT labels   :",
        np.unique(gt).tolist(),
    )

    # --------------------------------------------------------
    # Real PEM raw forward
    # --------------------------------------------------------

    with torch.enable_grad():

        outputs, images = (
            raw_pem_forward(
                model,
                batch,
            )
        )

        pred_logits = (
            outputs[
                "pred_logits"
            ]
        )

        pred_masks = (
            outputs[
                "pred_masks"
            ]
        )

        # We want to inspect gradient directly
        # on these non-leaf tensors.
        pred_logits.retain_grad()
        pred_masks.retain_grad()

        print()
        print(
            "pred_logits :",
            tuple(
                pred_logits.shape
            ),
        )

        print(
            "pred_masks  :",
            tuple(
                pred_masks.shape
            ),
        )

        # ----------------------------------------------------
        # Real GT -> Phi_G
        #
        # NOTE:
        # This is an interface/gradient sanity check.
        # Final training still needs halo-aware target creation.
        # ----------------------------------------------------

        target = build_ssb_target(
            gt,
            references=refs,
            num_classes=19,
            ignore_label=255,
            min_contour_points=16,
            radii=(2, 4, 8),
            d_max=32.0,
            weight_lambda=2.0,
            gamma=2.0,
            tau=args.tau,
        )

        phi = torch.from_numpy(
            target.phi
        ).unsqueeze(0).to(
            device=model.device,
            dtype=torch.float32,
        )

        weight = torch.from_numpy(
            target.weight
        ).unsqueeze(0).to(
            device=model.device,
            dtype=torch.float32,
        )

        valid_classes = (
            torch.from_numpy(
                target.valid_classes
            )
            .unsqueeze(0)
            .to(
                device=model.device,
                dtype=torch.bool,
            )
        )

        valid_mask = (
            torch.from_numpy(
                target.valid_mask
            )
            .unsqueeze(0)
            .to(
                device=model.device,
                dtype=torch.bool,
            )
        )

        print()
        print(
            "Phi shape   :",
            tuple(
                phi.shape
            ),
        )

        print(
            "Weight shape:",
            tuple(
                weight.shape
            ),
        )

        print(
            "Weight min/max:",
            float(
                weight.min().detach()
            ),
            float(
                weight.max().detach()
            ),
        )

        print(
            "Valid cls   :",
            torch.nonzero(
                valid_classes[0],
                as_tuple=False,
            )
            .flatten()
            .tolist(),
        )

        print(
            "Valid pixels:",
            int(
                valid_mask.sum()
            ),
        )

        # ----------------------------------------------------
        # semantic map
        # ----------------------------------------------------

        sem_prob = (
            compose_semantic_probability(
                pred_logits,
                pred_masks,
            )
        )

        print()
        print(
            "sem_prob    :",
            tuple(
                sem_prob.shape
            ),
        )

        print(
            "sem min/max :",
            float(
                sem_prob.min()
                .detach()
            ),
            float(
                sem_prob.max()
                .detach()
            ),
        )

        # Boundary at native mask resolution
        boundary_native = (
            soft_sobel_boundary(
                sem_prob
            )
        )

        print(
            "boundary    :",
            tuple(
                boundary_native.shape
            ),
        )

        print(
            "boundary min/max:",
            float(
                boundary_native.min()
                .detach()
            ),
            float(
                boundary_native.max()
                .detach()
            ),
        )

        # ----------------------------------------------------
        # Real SSB
        # ----------------------------------------------------

        loss_ssb = (
            shape_sensitive_boundary_loss(
                pred_logits=pred_logits,
                pred_masks=pred_masks,
                phi=phi,
                weight=weight,
                valid_classes=valid_classes,
                valid_mask=valid_mask,
            )
        )

        print()
        print(
            "loss_ssb    :",
            float(
                loss_ssb.detach()
            ),
        )

        assert torch.isfinite(
            loss_ssb
        ), "loss_ssb is not finite"

        loss_ssb.backward()

    # --------------------------------------------------------
    # Gradient checks
    # --------------------------------------------------------

    if pred_logits.grad is None:
        raise RuntimeError(
            "pred_logits.grad is None"
        )

    if pred_masks.grad is None:
        raise RuntimeError(
            "pred_masks.grad is None"
        )

    grad_logits = float(
        pred_logits.grad
        .abs()
        .sum()
    )

    grad_masks = float(
        pred_masks.grad
        .abs()
        .sum()
    )

    print()
    print(
        "|grad pred_logits|:",
        grad_logits,
    )

    print(
        "|grad pred_masks| :",
        grad_masks,
    )

    assert (
        grad_logits > 0.0
    ), "No gradient to pred_logits"

    assert (
        grad_masks > 0.0
    ), "No gradient to pred_masks"

    # --------------------------------------------------------
    # Confirm gradient reaches actual PEM decoder parameters.
    # --------------------------------------------------------

    class_grad = 0.0
    mask_grad = 0.0

    class_params = 0
    mask_params = 0

    for name, param in (
        model.named_parameters()
    ):

        if param.grad is None:
            continue

        grad_sum = float(
            param.grad.abs().sum()
        )

        if "class_embed" in name:
            class_grad += grad_sum
            class_params += 1

        if "mask_embed" in name:
            mask_grad += grad_sum
            mask_params += 1

    print()
    print(
        "class_embed params with grad:",
        class_params,
    )

    print(
        "|grad class_embed|:",
        class_grad,
    )

    print(
        "mask_embed params with grad :",
        mask_params,
    )

    print(
        "|grad mask_embed| :",
        mask_grad,
    )

    assert class_grad > 0.0
    assert mask_grad > 0.0

    print()
    print("=" * 90)
    print(
        "REAL PEM + REAL GT SSB "
        "SANITY PASSED"
    )
    print("=" * 90)

    print()
    print(
        "NOTE: This validates the real PEM/SSB interface "
        "and gradient flow only."
    )

    print(
        "Final training integration still requires "
        "halo-aware GT target construction and "
        "final TRAIN-derived tau."
    )


if __name__ == "__main__":
    main()
