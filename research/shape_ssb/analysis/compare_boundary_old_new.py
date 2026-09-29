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
    boundary_f1_from_maps,
    boundary_iou_from_masks,
    _binary_boundary,
    _ignore_adjacent_mask,
)


NUM_CLASSES = 19
IGNORE_LABEL = 255
TOLERANCE_PX = 2.0
BOUNDARY_IOU_RATIO = 0.02
MIN_GT_BOUNDARY_PIXELS = 16


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


def resolve_width(shape, ratio):
    h, w = shape
    diagonal = float(
        np.hypot(h, w)
    )

    return max(
        1,
        int(round(
            ratio * diagonal
        )),
    )


def mask_to_inner_band(mask, width_px):
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if not mask.any():
        return np.zeros_like(
            mask,
            dtype=bool,
        )

    eroded = ndi.binary_erosion(
        mask,
        structure=np.ones(
            (3, 3),
            dtype=bool,
        ),
        iterations=int(width_px),
        border_value=0,
    )

    return mask & (~eroded)


def old_metrics_for_class(
    pred,
    gt,
    class_id,
):
    """
    OLD behavior:
    valid를 먼저 class mask에 적용한 뒤
    boundary / boundary band를 계산.
    """

    invalid = _ignore_adjacent_mask(
        gt,
        IGNORE_LABEL,
    )
    valid = ~invalid

    pred_mask = (
        (pred == class_id)
        & valid
    )

    gt_mask = (
        (gt == class_id)
        & valid
    )

    # OLD BF1
    pb = (
        _binary_boundary(
            pred_mask
        )
        & valid
    )

    gb = (
        _binary_boundary(
            gt_mask
        )
        & valid
    )

    n_pb = int(pb.sum())
    n_gb = int(gb.sum())

    if n_gb < MIN_GT_BOUNDARY_PIXELS:
        return None

    bf1, precision, recall = (
        boundary_f1_from_maps(
            pb,
            gb,
            tolerance_px=TOLERANCE_PX,
        )
    )

    # OLD Boundary IoU
    width_px = resolve_width(
        pred.shape,
        BOUNDARY_IOU_RATIO,
    )

    pred_band = mask_to_inner_band(
        pred_mask,
        width_px,
    )

    gt_band = mask_to_inner_band(
        gt_mask,
        width_px,
    )

    union = (
        pred_band
        | gt_band
    )

    if union.any():
        biou = float(
            (
                pred_band
                & gt_band
            ).sum()
            / union.sum()
        )
    else:
        biou = 1.0

    return {
        "n_pb": n_pb,
        "n_gb": n_gb,
        "BF1": bf1,
        "Precision": precision,
        "Recall": recall,
        "BoundaryIoU": biou,
    }


def new_metrics_for_class(
    pred,
    gt,
    class_id,
):
    """
    NEW behavior:
    raw mask에서 boundary/band 계산 후
    valid를 마지막에 적용.
    """

    pb, gb, valid = (
        class_boundary_maps(
            pred,
            gt,
            class_id,
            IGNORE_LABEL,
        )
    )

    n_pb = int(pb.sum())
    n_gb = int(gb.sum())

    if n_gb < MIN_GT_BOUNDARY_PIXELS:
        return None

    bf1, precision, recall = (
        boundary_f1_from_maps(
            pb,
            gb,
            tolerance_px=TOLERANCE_PX,
        )
    )

    raw_pred_mask = (
        pred == class_id
    )

    raw_gt_mask = (
        gt == class_id
    )

    biou = boundary_iou_from_masks(
        raw_pred_mask,
        raw_gt_mask,
        dilation_ratio=BOUNDARY_IOU_RATIO,
        width_px=None,
        valid_mask=valid,
    )

    return {
        "n_pb": n_pb,
        "n_gb": n_gb,
        "BF1": bf1,
        "Precision": precision,
        "Recall": recall,
        "BoundaryIoU": biou,
    }


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

    model = Trainer.build_model(
        cfg
    )
    model.eval()

    DetectionCheckpointer(
        model
    ).load(
        args.checkpoint
    )

    dataset_name = (
        cfg.DATASETS.TEST[0]
    )

    loader = (
        Trainer.build_test_loader(
            cfg,
            dataset_name,
        )
    )

    total_old_pb = 0
    total_new_pb = 0

    total_old_gb = 0
    total_new_gb = 0

    old_bf1 = []
    new_bf1 = []

    old_biou = []
    new_biou = []

    old_precision = []
    new_precision = []

    old_recall = []
    new_recall = []

    processed = 0
    class_instances = 0
    flip_old_only = 0
    flip_new_only = 0

    print("=" * 100)
    print("OLD vs NEW boundary metric comparison")
    print("=" * 100)

    with torch.no_grad():

        for batch in loader:

            outputs = model(batch)

            for input_dict, output_dict in zip(
                batch,
                outputs,
            ):

                pred = (
                    prediction_to_label_map(
                        output_dict
                    )
                )

                gt = load_gt(
                    input_dict
                )

                processed += 1

                print()
                print(
                    f"[Image {processed}] "
                    f"{input_dict.get('file_name', 'unknown')}"
                )

                print(
                    "-" * 100
                )

                for c in range(
                    NUM_CLASSES
                ):

                    old = (
                        old_metrics_for_class(
                            pred,
                            gt,
                            c,
                        )
                    )

                    new = (
                        new_metrics_for_class(
                            pred,
                            gt,
                            c,
                        )
                    )

                    # 한쪽만 valid class가 된 경우도 표시
                    # 둘 다 invalid이면 비교 대상 아님
                    if old is None and new is None:
                        continue

                    # NEW에서만 valid
                    if old is None and new is not None:
                        flip_new_only += 1

                        print(
                            f"class {c:2d} | "
                            f"OLD skipped(<{MIN_GT_BOUNDARY_PIXELS}) | "
                            f"NEW gb={new['n_gb']:6d} "
                            f"pb={new['n_pb']:6d} "
                            f"BF1={new['BF1']:.6f} "
                            f"BIoU={new['BoundaryIoU']:.6f}"
                        )

                        continue

                    # OLD에서만 valid
                    if new is None and old is not None:
                        flip_old_only += 1

                        print(
                            f"class {c:2d} | "
                            f"OLD gb={old['n_gb']:6d} "
                            f"pb={old['n_pb']:6d} "
                            f"BF1={old['BF1']:.6f} "
                            f"BIoU={old['BoundaryIoU']:.6f} | "
                            f"NEW skipped(<{MIN_GT_BOUNDARY_PIXELS})"
                        )

                        continue

                    # 둘 다 valid
                    print(
                        f"class {c:2d} | "
                        f"OLD gb={old['n_gb']:6d} "
                        f"pb={old['n_pb']:6d} "
                        f"BF1={old['BF1']:.6f} "
                        f"BIoU={old['BoundaryIoU']:.6f} | "
                        f"NEW gb={new['n_gb']:6d} "
                        f"pb={new['n_pb']:6d} "
                        f"BF1={new['BF1']:.6f} "
                        f"BIoU={new['BoundaryIoU']:.6f}"
                    )
                    class_instances += 1

                    total_old_pb += old["n_pb"]
                    total_new_pb += new["n_pb"]

                    total_old_gb += old["n_gb"]
                    total_new_gb += new["n_gb"]

                    old_bf1.append(
                        old["BF1"]
                    )
                    new_bf1.append(
                        new["BF1"]
                    )

                    old_biou.append(
                        old["BoundaryIoU"]
                    )
                    new_biou.append(
                        new["BoundaryIoU"]
                    )

                    old_precision.append(
                        old["Precision"]
                    )
                    new_precision.append(
                        new["Precision"]
                    )

                    old_recall.append(
                        old["Recall"]
                    )
                    new_recall.append(
                        new["Recall"]
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

    print()
    print("=" * 100)
    print("FINAL OLD vs NEW SUMMARY")
    print("=" * 100)

    print(
        f"Images              : "
        f"{processed}"
    )

    print(
        f"Compared class-inst.: "
        f"{class_instances}"
    )
    print(
    f"Valid only in OLD    : "
    f"{flip_old_only}"
    )

    print(
        f"Valid only in NEW    : "
        f"{flip_new_only}"
    )

    print()
    print(
        f"GT boundary pixels  OLD: "
        f"{total_old_gb}"
    )
    print(
        f"GT boundary pixels  NEW: "
        f"{total_new_gb}"
    )
    print(
        f"GT boundary delta      : "
        f"{total_new_gb - total_old_gb:+d}"
    )

    print()
    print(
        f"Pred boundary pixels OLD: "
        f"{total_old_pb}"
    )
    print(
        f"Pred boundary pixels NEW: "
        f"{total_new_pb}"
    )
    print(
        f"Pred boundary delta     : "
        f"{total_new_pb - total_old_pb:+d}"
    )

    if class_instances > 0:

        old_bf1_m = float(
            np.mean(old_bf1)
        )
        new_bf1_m = float(
            np.mean(new_bf1)
        )

        old_biou_m = float(
            np.mean(old_biou)
        )
        new_biou_m = float(
            np.mean(new_biou)
        )

        old_p_m = float(
            np.mean(old_precision)
        )
        new_p_m = float(
            np.mean(new_precision)
        )

        old_r_m = float(
            np.mean(old_recall)
        )
        new_r_m = float(
            np.mean(new_recall)
        )

        print()
        print(
            f"BF1 OLD : "
            f"{old_bf1_m * 100:.6f}"
        )
        print(
            f"BF1 NEW : "
            f"{new_bf1_m * 100:.6f}"
        )
        print(
            f"BF1 Δ   : "
            f"{(new_bf1_m - old_bf1_m) * 100:+.6f}"
        )

        print()
        print(
            f"BIoU OLD: "
            f"{old_biou_m * 100:.6f}"
        )
        print(
            f"BIoU NEW: "
            f"{new_biou_m * 100:.6f}"
        )
        print(
            f"BIoU Δ  : "
            f"{(new_biou_m - old_biou_m) * 100:+.6f}"
        )

        print()
        print(
            f"Prec OLD: "
            f"{old_p_m * 100:.6f}"
        )
        print(
            f"Prec NEW: "
            f"{new_p_m * 100:.6f}"
        )
        print(
            f"Prec Δ  : "
            f"{(new_p_m - old_p_m) * 100:+.6f}"
        )

        print()
        print(
            f"Rec OLD : "
            f"{old_r_m * 100:.6f}"
        )
        print(
            f"Rec NEW : "
            f"{new_r_m * 100:.6f}"
        )
        print(
            f"Rec Δ   : "
            f"{(new_r_m - old_r_m) * 100:+.6f}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
