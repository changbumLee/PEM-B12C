#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from pem import add_maskformer2_config
from train_net import Trainer

from research.shape_ssb.geometry.shape_score import (
    load_curvature_references,
    build_shape_score_maps,
)
from research.shape_ssb.evaluation.shape_metrics import (
    ShapeSensitiveBoundaryMeter,
)
from research.shape_ssb.evaluation.boundary_metrics import (
    class_boundary_maps,
)


def build_cfg(config_file, checkpoint):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = checkpoint
    cfg.freeze()

    return cfg


def prediction_to_label_map(output):
    if "sem_seg" not in output:
        raise KeyError(
            f"'sem_seg' missing from model output. "
            f"keys={list(output.keys())}"
        )

    sem_seg = output["sem_seg"]

    if not torch.is_tensor(sem_seg):
        raise TypeError(type(sem_seg))

    if sem_seg.ndim != 3:
        raise ValueError(
            f"sem_seg must be [C,H,W], got {tuple(sem_seg.shape)}"
        )

    return (
        sem_seg.argmax(dim=0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32)
    )


def load_gt(input_dict):
    # Prefer already-loaded semantic GT if available.
    if "sem_seg" in input_dict:
        x = input_dict["sem_seg"]

        if torch.is_tensor(x):
            gt = x.detach().cpu().numpy()
        else:
            gt = np.asarray(x)

        return gt.astype(np.int32, copy=False)

    gt_path = input_dict.get("sem_seg_file_name")

    if gt_path is None:
        raise KeyError(
            f"No sem_seg or sem_seg_file_name. "
            f"keys={list(input_dict.keys())}"
        )

    return np.asarray(
        Image.open(gt_path),
        dtype=np.int32,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument(
        "--curvature-reference",
        default=(
            "research/shape_ssb/geometry/"
            "curvature_reference_q95.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "output/kroc_curve/"
            "baseline_shape_eval.json"
        ),
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Smoke test only. Omit for full val.",
    )

    parser.add_argument(
        "--print-period",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Config / references
    # ---------------------------------------------------------
    cfg = build_cfg(
        args.config_file,
        args.checkpoint,
    )

    refs = load_curvature_references(
        args.curvature_reference
    )

    print("=" * 78)
    print("Baseline Shape-Sensitive Evaluation")
    print("=" * 78)
    print("Checkpoint :", args.checkpoint)
    print("Dataset    :", list(cfg.DATASETS.TEST))
    print("Reference  :", args.curvature_reference)
    print("Q95        :", refs)
    print("SC top frac: 0.25")
    print("BF tol     : 2 px")
    print("SCLE cap   : 32 px")
    print("L_min      : 16")
    print("=" * 78)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = Trainer.build_model(cfg)
    model.eval()

    DetectionCheckpointer(model).load(
        args.checkpoint
    )

    if len(cfg.DATASETS.TEST) != 1:
        raise RuntimeError(
            f"Expected one TEST dataset, got "
            f"{list(cfg.DATASETS.TEST)}"
        )

    dataset_name = cfg.DATASETS.TEST[0]

    loader = Trainer.build_test_loader(
        cfg,
        dataset_name,
    )

    # ---------------------------------------------------------
    # Shape-sensitive meter
    # ---------------------------------------------------------
    meter = ShapeSensitiveBoundaryMeter(
        num_classes=19,
        ignore_label=255,
        top_fraction=0.25,
        tolerance_px=2.0,
        min_gt_boundary_pixels=16,
        max_distance_px=32.0,
    )

    processed = 0

    with torch.no_grad():

        for batch in loader:

            outputs = model(batch)

            for input_dict, output_dict in zip(
                batch, outputs
            ):
                pred = prediction_to_label_map(
                    output_dict
                )

                gt = load_gt(input_dict)

                if pred.shape != gt.shape:
                    raise ValueError(
                        f"pred={pred.shape}, "
                        f"gt={gt.shape}"
                    )

                # ---------------------------------------------
                # Build S maps from GT using frozen TRAIN Q95
                # ---------------------------------------------
                shape_maps = build_shape_score_maps(
                    gt,
                    references=refs,
                    num_classes=19,
                    ignore_label=255,
                    min_contour_points=16,
                    radii=(2, 4, 8),
                )

                assert shape_maps.shape == (
                    19,
                    gt.shape[0],
                    gt.shape[1],
                )

                # ---------------------------------------------
                # SC-BF1 / SCLE / Q1-Q4
                # ---------------------------------------------
                meter.update(
                    pred,
                    gt,
                    shape_maps,
                )

                processed += 1

                if (
                    processed == 1
                    or processed % args.print_period == 0
                ):

                    print(
                        f"[{processed}] processed"
                    )

                if (
                    args.max_images is not None
                    and processed >= args.max_images
                ):
                    break

            if (
                args.max_images is not None
                and processed >= args.max_images
            ):
                break

    # ---------------------------------------------------------
    # Final metrics
    # ---------------------------------------------------------
    result = meter.compute()

    # Aggregate quartile point counts for tie/coverage sanity.
    q_counts = {
        "Q1": 0,
        "Q2": 0,
        "Q3": 0,
        "Q4": 0,
    }

    for row in meter.rows:
        quartiles = row.get("quartiles", {})

        for q in q_counts:
            if q in quartiles:
                q_counts[q] += int(
                    quartiles[q].get("count", 0)
                )

    final = {
        "processed_images": processed,

        "curvature_reference": {
            str(k): float(v)
            for k, v in refs.items()
        },

        "SC-BF1": float(result["SC-BF1"]),
        "SC-BF1_percent":
            float(result["SC-BF1"]) * 100.0,

        "SCLE_px":
            float(result["SCLE"]),

        "Q1_error_px":
            float(result["Q1_error"]),

        "Q2_error_px":
            float(result["Q2_error"]),

        "Q3_error_px":
            float(result["Q3_error"]),

        "Q4_error_px":
            float(result["Q4_error"]),

        "n_valid_class_instances":
            int(result["n_valid_class_instances"]),

        "quartile_point_counts":
            q_counts,
    }

    print()
    print("=" * 78)
    print("FINAL SHAPE-SENSITIVE BASELINE RESULTS")
    print("=" * 78)

    print(
        f"Images              : "
        f"{final['processed_images']}"
    )

    print(
        f"Valid class-inst.    : "
        f"{final['n_valid_class_instances']}"
    )
    print(
        f"SC-BF1              : "
        f"{final['SC-BF1_percent']:.4f}"
    )

    print(
        f"SCLE                : "
        f"{final['SCLE_px']:.6f} px"
    )

    print(
        f"Q1 localization err : "
        f"{final['Q1_error_px']:.6f} px"
    )

    print(
        f"Q2 localization err : "
        f"{final['Q2_error_px']:.6f} px"
    )

    print(
        f"Q3 localization err : "
        f"{final['Q3_error_px']:.6f} px"
    )

    print(
        f"Q4 localization err : "
        f"{final['Q4_error_px']:.6f} px"
    )

    print(
        "Quartile counts      : "
        f"{q_counts}"
    )

    print("=" * 78)

    out_path = Path(args.output)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
        json.dumps(
            final,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
