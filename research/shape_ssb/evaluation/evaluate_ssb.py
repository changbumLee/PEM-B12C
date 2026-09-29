#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PEM_ROOT = Path(__file__).resolve().parents[3]
if str(PEM_ROOT) not in sys.path:
    sys.path.insert(0, str(PEM_ROOT))

import argparse
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.evaluation import (
    DatasetEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
)
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from pem import add_maskformer2_config
from train_net import Trainer

try:
    from research.shape_ssb.evaluation.boundary_metrics import SemanticBoundaryMeter
except ImportError:
    from boundary_metrics import SemanticBoundaryMeter


class SSBBoundaryEvaluator(DatasetEvaluator):
    def __init__(
        self,
        num_classes: int = 19,
        ignore_label: int = 255,
        tolerance_px: float = 2.0,
        boundary_iou_ratio: float = 0.02,
        boundary_iou_width_px=None,
        min_gt_boundary_pixels: int = 16,
    ) -> None:
        self.meter = SemanticBoundaryMeter(
            num_classes=num_classes,
            ignore_label=ignore_label,
            tolerance_px=tolerance_px,
            boundary_iou_ratio=boundary_iou_ratio,
            boundary_iou_width_px=boundary_iou_width_px,
            min_gt_boundary_pixels=min_gt_boundary_pixels,
        )

    def reset(self) -> None:
        self.meter.reset()

    @staticmethod
    def _prediction_to_label_map(output: Dict) -> np.ndarray:
        if "sem_seg" not in output:
            raise KeyError(
                "Model output does not contain 'sem_seg'. "
                f"Available keys: {list(output.keys())}"
            )

        sem_seg = output["sem_seg"]

        if not torch.is_tensor(sem_seg):
            raise TypeError(
                f"output['sem_seg'] must be torch.Tensor, got {type(sem_seg)}"
            )

        if sem_seg.ndim != 3:
            raise ValueError(
                f"output['sem_seg'] must be [C,H,W], got {tuple(sem_seg.shape)}"
            )

        pred = sem_seg.argmax(dim=0)
        return pred.detach().cpu().numpy().astype(np.int32)

    @staticmethod
    def _load_gt_train_ids(input_dict: Dict) -> np.ndarray:
        """
        Load GT semantic trainId map [H,W].

        PEM test loader may provide either:
        1) input_dict["sem_seg"]          -> already-loaded Tensor/ndarray
        2) input_dict["sem_seg_file_name"] -> path to labelTrainIds PNG
        """

        # Preferred path for current PEM dataloader
        if "sem_seg" in input_dict:
            gt = input_dict["sem_seg"]

            if torch.is_tensor(gt):
                gt = gt.detach().cpu().numpy()
            else:
                gt = np.asarray(gt)

            if gt.ndim != 2:
                raise ValueError(
                    f"input['sem_seg'] must be [H,W], got {gt.shape}"
                )

            return gt.astype(np.int32, copy=False)

        # Fallback for standard Detectron2-style dataset dicts
        gt_path = input_dict.get("sem_seg_file_name")

        if gt_path is None:
            raise KeyError(
                "Input dict has neither 'sem_seg' nor 'sem_seg_file_name'. "
                f"Available keys: {list(input_dict.keys())}"
            )

        gt = np.asarray(Image.open(gt_path))

        if gt.ndim != 2:
            raise ValueError(
                f"GT semantic label must be [H,W], got {gt.shape} from {gt_path}"
            )

        return gt.astype(np.int32, copy=False)

    def process(self, inputs: List[Dict], outputs: List[Dict]) -> None:
        if len(inputs) != len(outputs):
            raise ValueError(
                f"inputs/outputs length mismatch: {len(inputs)} vs {len(outputs)}"
            )

        for input_dict, output_dict in zip(inputs, outputs):
            pred = self._prediction_to_label_map(output_dict)
            gt = self._load_gt_train_ids(input_dict)
            
            if not hasattr(self, "_debug_printed"):
                print("pred shape:", pred.shape, "gt shape:", gt.shape)
                self._debug_printed = True
            if pred.shape != gt.shape:
                raise ValueError(
                    "Prediction/GT spatial size mismatch.\n"
                    f"pred={pred.shape}, gt={gt.shape}, "
                    f"file={input_dict.get('file_name', 'unknown')}"
                )

            self.meter.update(pred, gt)

    def evaluate(self):
        result = self.meter.compute()

        def pct(x):
            if x is None:
                return x
            try:
                if np.isnan(x):
                    return float("nan")
            except TypeError:
                pass
            return float(x) * 100.0

        return OrderedDict(
            {
                "boundary": OrderedDict(
                    {
                        "BF1": pct(result["BF1"]),
                        "BoundaryIoU": pct(result["BoundaryIoU"]),
                        "BoundaryPrecision": pct(result["BoundaryPrecision"]),
                        "BoundaryRecall": pct(result["BoundaryRecall"]),
                        "n_valid_class_instances": int(
                            result["n_valid_class_instances"]
                        ),
                    }
                )
            }
        )


def build_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(args.config_file)

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.MODEL.WEIGHTS = args.checkpoint
    cfg.freeze()
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PEM + SSB boundary metrics on Cityscapes val"
    )

    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        default="./output/kroc_curve/eval_ssb",
    )
    parser.add_argument("--bf-tolerance", type=float, default=2.0)
    parser.add_argument("--boundary-iou-ratio", type=float, default=0.02)
    parser.add_argument("--boundary-iou-width-px", type=int, default=None)
    parser.add_argument("--l-min", type=int, default=16)
    parser.add_argument("opts", nargs=argparse.REMAINDER)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    setup_logger(
        output=args.output_dir,
        name="shape_ssb_eval",
    )

    cfg = build_cfg(args)

    if len(cfg.DATASETS.TEST) == 0:
        raise RuntimeError("cfg.DATASETS.TEST is empty.")

    print("=" * 78)
    print("SSB evaluation configuration")
    print("=" * 78)
    print(f"Config:                {args.config_file}")
    print(f"Checkpoint:            {args.checkpoint}")
    print(f"Datasets:              {list(cfg.DATASETS.TEST)}")
    print(f"BF1 tolerance:         {args.bf_tolerance} px")
    print(f"Boundary IoU ratio:    {args.boundary_iou_ratio}")
    print(f"Boundary IoU width px: {args.boundary_iou_width_px}")
    print(f"L_min:                 {args.l_min} px")
    print("=" * 78)

    model = Trainer.build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(args.checkpoint)

    all_results = OrderedDict()

    with torch.no_grad():
        for dataset_name in cfg.DATASETS.TEST:
            print("\n" + "=" * 78)
            print(f"Evaluating dataset: {dataset_name}")
            print("=" * 78)

            data_loader = Trainer.build_test_loader(cfg, dataset_name)

            standard_evaluator = Trainer.build_evaluator(
                cfg,
                dataset_name,
                output_folder=os.path.join(
                    args.output_dir,
                    dataset_name,
                    "standard",
                ),
            )

            boundary_evaluator = SSBBoundaryEvaluator(
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                tolerance_px=args.bf_tolerance,
                boundary_iou_ratio=args.boundary_iou_ratio,
                boundary_iou_width_px=args.boundary_iou_width_px,
                min_gt_boundary_pixels=args.l_min,
            )

            evaluator = DatasetEvaluators(
                [standard_evaluator, boundary_evaluator]
            )

            results = inference_on_dataset(
                model,
                data_loader,
                evaluator,
            )

            all_results[dataset_name] = results

            print("\n" + "=" * 78)
            print("FINAL RESULTS")
            print("=" * 78)

            sem = results.get("sem_seg", {})
            boundary = results.get("boundary", {})

            if "IoU" in sem:
                print(f"mIoU               : {sem['IoU']:.4f}")

            if "BF1" in boundary:
                print(f"BF1                : {boundary['BF1']:.4f}")
                print(f"Boundary IoU       : {boundary['BoundaryIoU']:.4f}")
                print(
                    f"Boundary Precision : "
                    f"{boundary['BoundaryPrecision']:.4f}"
                )
                print(
                    f"Boundary Recall    : "
                    f"{boundary['BoundaryRecall']:.4f}"
                )
                print(
                    f"Valid class-inst.  : "
                    f"{boundary['n_valid_class_instances']}"
                )

            print("=" * 78)

    summary_path = Path(args.output_dir) / "ssb_eval_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        for dataset_name, results in all_results.items():
            f.write(f"[{dataset_name}]\n")

            sem = results.get("sem_seg", {})
            boundary = results.get("boundary", {})

            for k, v in sem.items():
                f.write(f"sem_seg/{k}: {v}\n")

            for k, v in boundary.items():
                f.write(f"boundary/{k}: {v}\n")

            f.write("\n")

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
