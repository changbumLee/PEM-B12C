#!/usr/bin/env python3

from __future__ import annotations

import argparse
import numpy as np
import torch
from scipy import ndimage as ndi

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from pem import add_maskformer2_config
from train_net import Trainer

from research.shape_ssb.evaluation.boundary_metrics import (
    class_boundary_maps,
    _binary_boundary,
    _ignore_adjacent_mask,
)

NUM_CLASSES = 19
IGNORE_LABEL = 255


def build_cfg(config_file, checkpoint):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = checkpoint
    cfg.freeze()

    return cfg


def prediction_to_label_map(output):
    sem_seg = output["sem_seg"]

    return (
        sem_seg.argmax(dim=0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32)
    )


def load_gt(input_dict):
    if "sem_seg" in input_dict:
        x = input_dict["sem_seg"]

        if torch.is_tensor(x):
            return (
                x.detach()
                .cpu()
                .numpy()
                .astype(np.int32)
            )

        return np.asarray(
            x,
            dtype=np.int32,
        )

    from PIL import Image

    return np.asarray(
        Image.open(
            input_dict["sem_seg_file_name"]
        ),
        dtype=np.int32,
    )


def old_gt_boundary(
    gt,
    class_id,
):
    """
    OLD bugged behavior:
    valid를 먼저 class mask에 적용한 뒤 erosion.
    """

    invalid = _ignore_adjacent_mask(
        gt,
        IGNORE_LABEL,
    )

    valid = ~invalid

    gt_mask_old = (
        (gt == class_id)
        & valid
    )

    gb_old = (
        _binary_boundary(gt_mask_old)
        & valid
    )

    return gb_old


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
        "--max-images",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    cfg = build_cfg(
        args.config_file,
        args.checkpoint,
    )

    model = Trainer.build_model(cfg)
    model.eval()

    DetectionCheckpointer(
        model
    ).load(
        args.checkpoint
    )

    dataset_name = cfg.DATASETS.TEST[0]

    loader = Trainer.build_test_loader(
        cfg,
        dataset_name,
    )

    total_old_only = 0
    total_old_only_near_invalid = 0

    processed = 0

    print("=" * 100)
    print("OLD-only GT boundary vs ignore-neighborhood sanity")
    print("=" * 100)

    with torch.no_grad():

        for batch in loader:

            outputs = model(batch)

            for input_dict, output_dict in zip(
                batch,
                outputs,
            ):

                # prediction은 사실 이 sanity에서 필요 없지만,
                # 같은 evaluation loader 흐름을 그대로 유지.
                _ = prediction_to_label_map(
                    output_dict
                )

                gt = load_gt(
                    input_dict
                )

                invalid = _ignore_adjacent_mask(
                    gt,
                    IGNORE_LABEL,
                )

                # invalid 바깥쪽 1-pixel ring
                near_invalid = (
                    ndi.binary_dilation(
                        invalid,
                        structure=np.ones(
                            (3, 3),
                            dtype=bool,
                        ),
                        iterations=1,
                    )
                    & (~invalid)
                )

                processed += 1

                print()
                print(
                    f"[Image {processed}] "
                    f"{input_dict.get('file_name', 'unknown')}"
                )
                print("-" * 100)

                image_old_only = 0
                image_near = 0

                for c in range(
                    NUM_CLASSES
                ):

                    gb_old = old_gt_boundary(
                        gt,
                        c,
                    )

                    # NEW corrected GT boundary
                    _, gb_new, _ = (
                        class_boundary_maps(
                            pred=np.zeros_like(
                                gt,
                                dtype=np.int32,
                            ),
                            gt=gt,
                            class_id=c,
                            ignore_label=IGNORE_LABEL,
                        )
                    )

                    old_only = (
                        gb_old
                        & (~gb_new)
                    )

                    n_old_only = int(
                        old_only.sum()
                    )

                    if n_old_only == 0:
                        continue

                    n_near = int(
                        (
                            old_only
                            & near_invalid
                        ).sum()
                    )

                    ratio = (
                        n_near
                        / n_old_only
                    )

                    image_old_only += (
                        n_old_only
                    )

                    image_near += (
                        n_near
                    )

                    print(
                        f"class {c:2d} | "
                        f"old_only={n_old_only:6d} | "
                        f"near_invalid={n_near:6d} | "
                        f"ratio={ratio * 100:7.3f}%"
                    )

                image_ratio = (
                    image_near
                    / max(
                        image_old_only,
                        1,
                    )
                )

                print(
                    f"Image summary | "
                    f"old_only={image_old_only} | "
                    f"near_invalid={image_near} | "
                    f"ratio={image_ratio * 100:.3f}%"
                )

                total_old_only += (
                    image_old_only
                )

                total_old_only_near_invalid += (
                    image_near
                )

                if (
                    processed
                    >= args.max_images
                ):
                    break

            if (
                processed
                >= args.max_images
            ):
                break

    total_ratio = (
        total_old_only_near_invalid
        / max(
            total_old_only,
            1,
        )
    )

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(
        f"Images                         : "
        f"{processed}"
    )

    print(
        f"OLD-only GT boundary pixels   : "
        f"{total_old_only}"
    )

    print(
        f"Near-invalid OLD-only pixels  : "
        f"{total_old_only_near_invalid}"
    )

    print(
        f"Near-invalid ratio            : "
        f"{total_ratio * 100:.4f}%"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
