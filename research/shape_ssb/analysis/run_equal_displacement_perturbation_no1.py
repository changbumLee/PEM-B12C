#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config

from pem import add_maskformer2_config

from research.shape_ssb.geometry.contour import (
    extract_class_contours,
)
from research.shape_ssb.geometry.stability import (
    contour_shape_score,
)
from research.shape_ssb.geometry.shape_score import (
    load_curvature_references,
)


NUM_CLASSES = 19
IGNORE_LABEL = 255
RADII = (2, 4, 8)
MIN_CONTOUR_POINTS = 16


def build_cfg(config_file):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def load_gt(record):
    gt_path = record.get("sem_seg_file_name")

    if gt_path is None:
        raise KeyError(
            f"sem_seg_file_name missing: keys={list(record.keys())}"
        )

    return np.asarray(
        Image.open(gt_path),
        dtype=np.int32,
    )


def unit_normal(points, i, closed):
    """
    Local unit normal from the tangent p_{i+1} - p_{i-1}.
    """

    n = len(points)

    if n < 3:
        return None

    if closed:
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
    else:
        if i <= 0 or i >= n - 1:
            return None

        im1 = i - 1
        ip1 = i + 1

    tangent = (
        points[ip1].astype(np.float64)
        -
        points[im1].astype(np.float64)
    )

    norm = np.linalg.norm(tangent)

    if norm < 1e-8:
        return None

    tangent /= norm

    # 90-degree rotation
    normal = np.array(
        [-tangent[1], tangent[0]],
        dtype=np.float64,
    )

    return normal


def turning_angle(points, i, closed):
    """
    Turning angle at contour point i, radians in [0, pi].
    """

    n = len(points)

    if n < 3:
        return np.nan

    if closed:
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
    else:
        if i <= 0 or i >= n - 1:
            return np.nan

        im1 = i - 1
        ip1 = i + 1

    p0 = points[im1].astype(np.float64)
    p1 = points[i].astype(np.float64)
    p2 = points[ip1].astype(np.float64)

    v1 = p1 - p0
    v2 = p2 - p1

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan

    cosv = np.dot(v1, v2) / (n1 * n2)
    cosv = np.clip(cosv, -1.0, 1.0)

    return float(np.arccos(cosv))


def perturb_one_point(points, i, delta, closed):
    """
    Move only the center contour point i by exactly delta pixels
    along its local normal.
    """

    normal = unit_normal(
        points,
        i,
        closed,
    )

    if normal is None:
        return None

    perturbed = (
        points.astype(np.float64).copy()
    )

    perturbed[i] = (
        perturbed[i]
        +
        float(delta) * normal
    )

    return perturbed


def ranked_quartiles(scores):
    """
    Same policy as current shape evaluation:
    stable ascending rank + np.array_split.
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    valid_idx = np.flatnonzero(
        np.isfinite(scores)
    )

    if valid_idx.size < 4:
        return None

    order_local = np.argsort(
        scores[valid_idx],
        kind="mergesort",
    )

    ranked_idx = valid_idx[
        order_local
    ]

    groups = np.array_split(
        ranked_idx,
        4,
    )

    return {
        "Q1": groups[0],
        "Q2": groups[1],
        "Q3": groups[2],
        "Q4": groups[3],
    }


def process_contour(
    contour,
    references,
    deltas,
):
    """
    For one contour:
      1. compute S
      2. divide finite-S points into Q1-Q4 by rank
      3. move each point independently by the same delta
      4. measure absolute local turning-angle change
    """

    points = np.asarray(
        contour.points,
        dtype=np.float64,
    )

    geom = contour_shape_score(
        contour,
        references,
        RADII,
    )

    scores = np.asarray(
        geom["S"],
        dtype=np.float64,
    )

    quartiles = ranked_quartiles(
        scores
    )

    if quartiles is None:
        return None

    # Precompute original local angles.
    original_angle = np.array(
        [
            turning_angle(
                points,
                i,
                contour.closed,
            )
            for i in range(len(points))
        ],
        dtype=np.float64,
    )

    result = {
        float(delta): {
            "Q1": [],
            "Q2": [],
            "Q3": [],
            "Q4": [],
        }
        for delta in deltas
    }

    for qname, indices in quartiles.items():

        for i in indices:

            theta0 = original_angle[i]

            if not np.isfinite(theta0):
                continue

            for delta in deltas:

                perturbed = perturb_one_point(
                    points,
                    int(i),
                    float(delta),
                    contour.closed,
                )

                if perturbed is None:
                    continue

                theta1 = turning_angle(
                    perturbed,
                    int(i),
                    contour.closed,
                )

                if not np.isfinite(theta1):
                    continue

                distortion = abs(
                    theta1 - theta0
                )

                result[
                    float(delta)
                ][qname].append(
                    float(distortion)
                )

    return result


def merge_result(
    accum,
    local,
):
    for delta in accum:
        for q in (
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        ):
            accum[delta][q].extend(
                local[delta][q]
            )


def summarize(accum):
    summary = {}

    for delta, qdict in accum.items():
        delta_result = {}

        for q, values in qdict.items():
            arr = np.asarray(
                values,
                dtype=np.float64,
            )

            delta_result[q] = {
                "mean_rad": (
                    float(np.mean(arr))
                    if arr.size
                    else np.nan
                ),
                "mean_deg": (
                    float(
                        np.degrees(
                            np.mean(arr)
                        )
                    )
                    if arr.size
                    else np.nan
                ),
                "median_deg": (
                    float(
                        np.degrees(
                            np.median(arr)
                        )
                    )
                    if arr.size
                    else np.nan
                ),
                "count": int(arr.size),
            }

        summary[str(delta)] = (
            delta_result
        )

    return summary


def print_summary(
    summary,
    images,
    contours,
):
    print()
    print("=" * 88)
    print(
        "GT-ONLY EQUAL-DISPLACEMENT "
        "PERTURBATION RESULTS"
    )
    print("=" * 88)

    print(
        f"Images   : {images}"
    )
    print(
        f"Contours : {contours}"
    )

    print()
    print(
        "Metric: absolute local "
        "turning-angle change"
    )
    print(
        "(same delta, different "
        "shape-sensitivity quartile)"
    )

    for delta, qdict in summary.items():

        print()
        print(
            f"delta = {delta} px"
        )
        print("-" * 88)

        vals = []

        for q in (
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        ):
            row = qdict[q]

            mean_deg = row[
                "mean_deg"
            ]

            vals.append(mean_deg)

            print(
                f"{q}: "
                f"mean={mean_deg:.6f} deg | "
                f"median={row['median_deg']:.6f} deg | "
                f"count={row['count']}"
            )

        monotonic = (
            vals[0]
            < vals[1]
            < vals[2]
            < vals[3]
        )

        print(
            "Q1 < Q2 < Q3 < Q4 : "
            f"{monotonic}"
        )

        if (
            np.isfinite(vals[0])
            and vals[0] > 0
        ):
            print(
                "Q4 / Q1 ratio        : "
                f"{vals[3] / vals[0]:.4f}"
            )

    print("=" * 88)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
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
        "--output",
        default=(
            "output/kroc_curve/"
            "gt_equal_displacement.json"
        ),
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--print-period",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
    )

    args = parser.parse_args()

    cfg = build_cfg(
        args.config_file
    )

    references = (
        load_curvature_references(
            args.curvature_reference
        )
    )

    if len(cfg.DATASETS.TEST) != 1:
        raise RuntimeError(
            "Expected exactly one TEST "
            f"dataset, got {cfg.DATASETS.TEST}"
        )

    dataset_name = (
        cfg.DATASETS.TEST[0]
    )

    dataset = DatasetCatalog.get(
        dataset_name
    )

    deltas = tuple(
        float(x)
        for x in args.deltas
    )

    accum = {
        delta: {
            "Q1": [],
            "Q2": [],
            "Q3": [],
            "Q4": [],
        }
        for delta in deltas
    }

    processed_images = 0
    processed_contours = 0

    print("=" * 88)
    print(
        "GT-only Equal-Displacement "
        "Shape Perturbation"
    )
    print("=" * 88)
    print(
        "Dataset   :",
        dataset_name,
    )
    print(
        "Reference :",
        args.curvature_reference,
    )
    print(
        "Q95       :",
        references,
    )
    print(
        "Deltas    :",
        deltas,
    )
    print(
        "Metric    : local turning-angle distortion"
    )
    print("=" * 88)

    for record in dataset:

        gt = load_gt(
            record
        )

        for c in range(
            NUM_CLASSES
        ):

            contours = (
                extract_class_contours(
                    gt,
                    c,
                    IGNORE_LABEL,
                    MIN_CONTOUR_POINTS,
                )
            )

            for contour in contours:

                local = process_contour(
                    contour,
                    references,
                    deltas,
                )

                if local is None:
                    continue

                merge_result(
                    accum,
                    local,
                )

                processed_contours += 1

        processed_images += 1

        if (
            processed_images == 1
            or (
                args.print_period > 0
                and processed_images
                % args.print_period == 0
            )
        ):
            print(
                f"[{processed_images}] "
                "processed"
            )

        if (
            args.max_images is not None
            and processed_images
            >= args.max_images
        ):
            break

    summary = summarize(
        accum
    )

    print_summary(
        summary,
        processed_images,
        processed_contours,
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "dataset": dataset_name,
        "images": processed_images,
        "contours": processed_contours,
        "deltas_px": list(deltas),
        "metric": (
            "absolute_local_turning_angle_change"
        ),
        "curvature_reference": str(
            args.curvature_reference
        ),
        "results": summary,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
