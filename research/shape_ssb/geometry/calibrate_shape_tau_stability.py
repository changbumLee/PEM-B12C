#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.env import seed_all_rng

from pem import add_maskformer2_config
from pem.data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

from research.shape_ssb.geometry.contour import extract_class_contours
from research.shape_ssb.geometry.shape_score import load_curvature_references
from research.shape_ssb.geometry.stability import contour_shape_score


NUM_CLASSES = 19
IGNORE_LABEL = 255
RADII = (2, 4, 8)
MIN_CONTOUR_POINTS = 16


def npy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def collect_center_scores(halo_gt, halo_px, crop_shape, refs):
    crop_h, crop_w = map(int, crop_shape)
    h = int(halo_px)
    y0, y1 = h, h + crop_h
    x0, x1 = h, h + crop_w

    chunks = []
    for c in range(NUM_CLASSES):
        contours = extract_class_contours(
            halo_gt, c, IGNORE_LABEL, MIN_CONTOUR_POINTS
        )
        for contour in contours:
            S = np.asarray(
                contour_shape_score(contour, refs, RADII)["S"],
                dtype=np.float32,
            )
            xy = np.rint(contour.points).astype(np.int64)
            x = xy[:, 0]
            y = xy[:, 1]
            keep = (
                np.isfinite(S)
                & (x >= x0) & (x < x1)
                & (y >= y0) & (y < y1)
            )
            if keep.any():
                chunks.append(S[keep])

    if chunks:
        return np.concatenate(chunks)
    return np.empty(0, dtype=np.float32)


def build_cfg(config_file, seed):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.SEED = int(seed)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    return cfg


def select_dataset(cfg, split):
    if split == "train":
        if len(cfg.DATASETS.TRAIN) != 1:
            raise RuntimeError(
                f"Expected exactly one TRAIN dataset, got {list(cfg.DATASETS.TRAIN)}"
            )
        return cfg.DATASETS.TRAIN[0]

    if split == "val":
        if len(cfg.DATASETS.TEST) != 1:
            raise RuntimeError(
                f"Expected exactly one TEST/VAL dataset, got {list(cfg.DATASETS.TEST)}"
            )
        return cfg.DATASETS.TEST[0]

    raise ValueError(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument(
        "--curvature-reference",
        default="research/shape_ssb/geometry/curvature_reference_q95.json",
    )
    ap.add_argument("--split", choices=("train", "val"), default="train")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional smoke limit. Omit for full unique split.",
    )
    ap.add_argument("--print-period", type=int, default=100)
    ap.add_argument(
        "--output",
        default=None,
        help=(
            "Default: research/shape_ssb/geometry/"
            "shape_score_tau_<split>_seed<seed>.json"
        ),
    )
    a = ap.parse_args()

    cfg = build_cfg(a.config_file, a.seed)
    refs = load_curvature_references(a.curvature_reference)

    dataset_name = select_dataset(cfg, a.split)
    records = list(DatasetCatalog.get(dataset_name))
    if a.max_images is not None:
        records = records[: min(a.max_images, len(records))]

    # Iterate each dataset record exactly once.
    # This avoids the randomized/infinite train-loader sampler.
    mapper = MaskFormerSemanticDatasetMapper(cfg, True)

    # Seed controls only the stochastic PEM training augmentations
    # applied while mapping this fixed list of unique records.
    seed_all_rng(a.seed)

    vals = []
    npts = 0

    for i, record in enumerate(records, start=1):
        mapped = mapper(record)

        gt = npy(mapped["sem_seg"])
        halo = npy(mapped["ssb_sem_seg_halo"])
        halo_px = int(mapped["ssb_halo_px"])

        expected = (
            gt.shape[0] + 2 * halo_px,
            gt.shape[1] + 2 * halo_px,
        )
        if halo.shape != expected:
            raise RuntimeError(
                f"Halo shape mismatch at image {i}: "
                f"gt={gt.shape}, halo={halo.shape}, expected={expected}"
            )

        center = halo[
            halo_px : halo_px + gt.shape[0],
            halo_px : halo_px + gt.shape[1],
        ]
        mismatch = int(np.count_nonzero(center != gt))
        if mismatch != 0:
            raise RuntimeError(
                f"Halo/GT center mismatch at image {i}: {mismatch} pixels"
            )

        s = collect_center_scores(halo, halo_px, gt.shape, refs)
        if s.size:
            vals.append(s.astype(np.float32, copy=False))
            npts += int(s.size)

        if i % a.print_period == 0 or i == len(records):
            print(
                f"[{i}/{len(records)}] split={a.split} "
                f"seed={a.seed} valid S={npts:,}"
            )

    if not vals:
        raise RuntimeError("No finite S collected")

    v = np.concatenate(vals).astype(np.float32, copy=False)
    q = np.quantile(v, [0.05, 0.25, 0.50, 0.75, 0.95])

    quantiles = {
        "q05": float(q[0]),
        "q25": float(q[1]),
        "q50": float(q[2]),
        "q75": float(q[3]),
        "q95": float(q[4]),
    }
    tau = float(q[2])

    if a.output is None:
        output = Path(
            "research/shape_ssb/geometry/"
            f"shape_score_tau_{a.split}_seed{a.seed}.json"
        )
    else:
        output = Path(a.output)

    out = {
        "dataset": dataset_name,
        "split": a.split,
        "num_images": len(records),
        "num_valid_S_points": int(v.size),
        "seed": int(a.seed),
        "coverage": "each DatasetCatalog record exactly once",
        "augmentation": (
            "same PEM training mapper distribution "
            "(resize/crop/color/flip); color does not affect GT geometry"
        ),
        "tau_rule": "Q50(valid S)",
        "tau": tau,
        "quantiles": quantiles,
        "radii": list(RADII),
        "min_contour_points": MIN_CONTOUR_POINTS,
        "ignore_label": IGNORE_LABEL,
        "curvature_reference": a.curvature_reference,
        "calibration_distribution": (
            "S computed with halo context; "
            "only finite contour points inside final center crop retained"
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("SHAPE-SCORE DISTRIBUTION")
    print("=" * 78)
    print("dataset      =", dataset_name)
    print("split        =", a.split)
    print("seed         =", a.seed)
    print("images       =", len(records))
    print("valid S      =", f"{v.size:,}")
    print("q05          =", quantiles["q05"])
    print("q25          =", quantiles["q25"])
    print("q50 / tau    =", quantiles["q50"])
    print("q75          =", quantiles["q75"])
    print("q95          =", quantiles["q95"])
    print("saved to     =", output)


if __name__ == "__main__":
    main()
