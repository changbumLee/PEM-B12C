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
EPS = 1e-8


# ============================================================
# Config / GT
# ============================================================

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
            "sem_seg_file_name missing: "
            f"keys={list(record.keys())}"
        )

    return np.asarray(
        Image.open(gt_path),
        dtype=np.int32,
    )


# ============================================================
# Radius-aware geometry
# ============================================================

def radius_indices(n, i, radius, closed):
    r = int(radius)

    if closed:
        return (
            (i - r) % n,
            (i + r) % n,
        )

    if i - r < 0 or i + r >= n:
        return None

    return (
        i - r,
        i + r,
    )


def turning_geometry_at_radius(
    points,
    i,
    radius,
    closed,
):
    """
    Same geometric definition as curvature.py.

    Returns
    -------
    theta:
        turning angle [rad]
    curvature:
        theta / local_length
    tangent:
        centered unit tangent
    """

    pts = np.asarray(
        points,
        dtype=np.float64,
    )

    n = len(pts)

    idx = radius_indices(
        n,
        int(i),
        int(radius),
        bool(closed),
    )

    if idx is None:
        return None

    i_prev, i_next = idx

    p_prev = pts[i_prev]
    p_cur = pts[i]
    p_next = pts[i_next]

    v_minus = p_cur - p_prev
    v_plus = p_next - p_cur

    len_minus = np.linalg.norm(
        v_minus
    )
    len_plus = np.linalg.norm(
        v_plus
    )

    if (
        len_minus < EPS
        or len_plus < EPS
    ):
        return None

    cosang = (
        np.dot(v_minus, v_plus)
        /
        (
            len_minus
            * len_plus
            + EPS
        )
    )

    theta = float(
        np.arccos(
            np.clip(
                cosang,
                -1.0,
                1.0,
            )
        )
    )

    local_length = (
        0.5
        * (len_minus + len_plus)
        + EPS
    )

    curvature = (
        theta / local_length
    )

    tangent = (
        p_next - p_prev
    )

    tangent_norm = np.linalg.norm(
        tangent
    )

    if tangent_norm < EPS:
        return None

    tangent = (
        tangent / tangent_norm
    )

    return {
        "theta": theta,
        "curvature": float(curvature),
        "tangent": tangent,
    }


# ============================================================
# Multi-scale local normal
# ============================================================

def multiscale_unit_normal(
    points,
    i,
    radii,
    closed,
):
    """
    Estimate a stable local normal using the same scales
    r={2,4,8} used for S.

    Tangents at different radii are orientation-aligned
    before averaging.
    """

    tangents = []

    for r in radii:
        g = turning_geometry_at_radius(
            points,
            i,
            r,
            closed,
        )

        if g is None:
            continue

        tangents.append(
            g["tangent"].copy()
        )

    if not tangents:
        return None

    reference = tangents[0]

    aligned = []

    for tangent in tangents:
        t = tangent.copy()

        if np.dot(
            t,
            reference,
        ) < 0:
            t = -t

        aligned.append(t)

    tangent = np.mean(
        np.stack(
            aligned,
            axis=0,
        ),
        axis=0,
    )

    tangent_norm = np.linalg.norm(
        tangent
    )

    if tangent_norm < EPS:
        return None

    tangent /= tangent_norm

    normal = np.array(
        [
            -tangent[1],
            tangent[0],
        ],
        dtype=np.float64,
    )

    normal_norm = np.linalg.norm(
        normal
    )

    if normal_norm < EPS:
        return None

    return (
        normal / normal_norm
    )


# ============================================================
# Smooth local-segment perturbation
# ============================================================

def contour_offset_index(
    center,
    offset,
    n,
    closed,
):
    idx = center + offset

    if closed:
        return idx % n

    if idx < 0 or idx >= n:
        return None

    return idx


def hann_weight(offset, support_radius):
    """
    Smooth window:

        g(0) = 1
        g(+-H) = 0

    g(k) = 0.5 * (1 + cos(pi*k/H))
    """

    H = int(support_radius)

    if H <= 0:
        raise ValueError(
            "support_radius must be > 0"
        )

    k = abs(int(offset))

    if k > H:
        return 0.0

    return float(
        0.5
        * (
            1.0
            + np.cos(
                np.pi * k / H
            )
        )
    )


def smooth_local_perturbation(
    points,
    center_idx,
    delta,
    support_radius,
    radii,
    closed,
):
    """
    Smoothly deform a local contour segment.

    For each point j around center i:

        p'_j = p_j + delta * g(j-i) * n_j

    where:
        g(0) = 1
        g(+/-H) = 0

    Therefore the center point moves exactly delta pixels,
    while the displacement smoothly decreases toward
    the segment boundary.
    """

    pts = np.asarray(
        points,
        dtype=np.float64,
    )

    perturbed = pts.copy()

    n_points = len(pts)

    affected = []

    weights = []

    for offset in range(
        -support_radius,
        support_radius + 1,
    ):
        idx = contour_offset_index(
            center_idx,
            offset,
            n_points,
            closed,
        )

        if idx is None:
            continue

        weight = hann_weight(
            offset,
            support_radius,
        )

        if weight <= 0:
            continue

        normal = multiscale_unit_normal(
            pts,
            idx,
            radii,
            closed,
        )

        if normal is None:
            continue

        perturbed[idx] = (
            pts[idx]
            + float(delta)
            * weight
            * normal
        )

        affected.append(
            int(idx)
        )
        weights.append(
            float(weight)
        )

    if not affected:
        return None

    # Center must exist and move exactly delta.
    center_normal = (
        multiscale_unit_normal(
            pts,
            center_idx,
            radii,
            closed,
        )
    )

    if center_normal is None:
        return None

    actual_delta = np.linalg.norm(
        perturbed[center_idx]
        - pts[center_idx]
    )

    if not np.isclose(
        actual_delta,
        float(delta),
        atol=1e-6,
    ):
        return None

    return {
        "points": perturbed,
        "affected_indices": np.asarray(
            affected,
            dtype=np.int64,
        ),
        "weights": np.asarray(
            weights,
            dtype=np.float64,
        ),
    }


# ============================================================
# Quartile split
# ============================================================

def ranked_quartiles(scores):
    """
    Same policy as current shape-sensitive evaluation.

    Q1 = lowest S
    Q4 = highest S
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

    order = np.argsort(
        scores[valid_idx],
        kind="mergesort",
    )

    ranked = valid_idx[
        order
    ]

    groups = np.array_split(
        ranked,
        4,
    )

    return {
        "Q1": groups[0],
        "Q2": groups[1],
        "Q3": groups[2],
        "Q4": groups[3],
    }


# ============================================================
# Distortion measurement
# ============================================================

def geometry_distortion_at_point(
    before_points,
    after_points,
    idx,
    radius,
    closed,
):
    before = turning_geometry_at_radius(
        before_points,
        idx,
        radius,
        closed,
    )

    after = turning_geometry_at_radius(
        after_points,
        idx,
        radius,
        closed,
    )

    if before is None or after is None:
        return None

    dtheta = abs(
        after["theta"]
        - before["theta"]
    )

    dkappa = abs(
        after["curvature"]
        - before["curvature"]
    )

    return {
        "dtheta": float(dtheta),
        "dkappa": float(dkappa),
    }


def weighted_local_distortion(
    before_points,
    after_points,
    affected_indices,
    weights,
    radius,
    closed,
):
    """
    Geometry damage across the entire smoothly perturbed
    local segment.

    Hann weights are also used during measurement so that
    the center region contributes most.
    """

    dtheta_values = []
    dkappa_values = []
    valid_weights = []

    for idx, weight in zip(
        affected_indices,
        weights,
    ):
        d = geometry_distortion_at_point(
            before_points,
            after_points,
            int(idx),
            radius,
            closed,
        )

        if d is None:
            continue

        dtheta_values.append(
            d["dtheta"]
        )

        dkappa_values.append(
            d["dkappa"]
        )

        valid_weights.append(
            float(weight)
        )

    if not dtheta_values:
        return None

    w = np.asarray(
        valid_weights,
        dtype=np.float64,
    )

    dtheta_arr = np.asarray(
        dtheta_values,
        dtype=np.float64,
    )

    dkappa_arr = np.asarray(
        dkappa_values,
        dtype=np.float64,
    )

    w_sum = float(
        np.sum(w)
    )

    if w_sum <= EPS:
        return None

    return {
        "dtheta":
            float(
                np.sum(
                    w * dtheta_arr
                )
                / w_sum
            ),
        "dkappa":
            float(
                np.sum(
                    w * dkappa_arr
                )
                / w_sum
            ),
    }


# ============================================================
# Accumulator
# ============================================================

def new_metric_bucket():
    return {
        "center_dtheta": [],
        "center_dkappa": [],
        "local_dtheta": [],
        "local_dkappa": [],
    }


def new_accumulator(deltas):
    result = {}

    for delta in deltas:
        result[float(delta)] = {}

        for r in RADII:
            result[float(delta)][r] = {
                "Q1": new_metric_bucket(),
                "Q2": new_metric_bucket(),
                "Q3": new_metric_bucket(),
                "Q4": new_metric_bucket(),
            }

    return result


# ============================================================
# Per-contour experiment
# ============================================================

def process_contour(
    contour,
    references,
    deltas,
    support_radius,
):
    points = np.asarray(
        contour.points,
        dtype=np.float64,
    )

    geometry = contour_shape_score(
        contour,
        references,
        RADII,
    )

    S = np.asarray(
        geometry["S"],
        dtype=np.float64,
    )

    quartiles = ranked_quartiles(S)

    if quartiles is None:
        return None

    local_result = (
        new_accumulator(
            deltas
        )
    )

    for qname, indices in (
        quartiles.items()
    ):

        for center_idx in indices:

            center_idx = int(
                center_idx
            )

            # Require full multi-scale support at center.
            valid_center = True

            for r in RADII:
                if (
                    turning_geometry_at_radius(
                        points,
                        center_idx,
                        r,
                        contour.closed,
                    )
                    is None
                ):
                    valid_center = False
                    break

            if not valid_center:
                continue

            for delta in deltas:

                deformation = (
                    smooth_local_perturbation(
                        points,
                        center_idx,
                        float(delta),
                        support_radius,
                        RADII,
                        contour.closed,
                    )
                )

                if deformation is None:
                    continue

                perturbed = deformation[
                    "points"
                ]

                affected_indices = (
                    deformation[
                        "affected_indices"
                    ]
                )

                weights = deformation[
                    "weights"
                ]

                for r in RADII:

                    center_dist = (
                        geometry_distortion_at_point(
                            points,
                            perturbed,
                            center_idx,
                            r,
                            contour.closed,
                        )
                    )

                    local_dist = (
                        weighted_local_distortion(
                            points,
                            perturbed,
                            affected_indices,
                            weights,
                            r,
                            contour.closed,
                        )
                    )

                    if (
                        center_dist is None
                        or local_dist is None
                    ):
                        continue

                    bucket = (
                        local_result[
                            float(delta)
                        ][r][qname]
                    )

                    bucket[
                        "center_dtheta"
                    ].append(
                        center_dist[
                            "dtheta"
                        ]
                    )

                    bucket[
                        "center_dkappa"
                    ].append(
                        center_dist[
                            "dkappa"
                        ]
                    )

                    bucket[
                        "local_dtheta"
                    ].append(
                        local_dist[
                            "dtheta"
                        ]
                    )

                    bucket[
                        "local_dkappa"
                    ].append(
                        local_dist[
                            "dkappa"
                        ]
                    )

    return local_result


def merge_result(
    accum,
    local,
):
    for delta in accum:
        for r in RADII:
            for q in (
                "Q1",
                "Q2",
                "Q3",
                "Q4",
            ):
                for metric in (
                    "center_dtheta",
                    "center_dkappa",
                    "local_dtheta",
                    "local_dkappa",
                ):
                    accum[
                        delta
                    ][r][q][metric].extend(
                        local[
                            delta
                        ][r][q][metric]
                    )


# ============================================================
# Summary
# ============================================================

def summarize_array(values):
    arr = np.asarray(
        values,
        dtype=np.float64,
    )

    if arr.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "count": 0,
        }

    return {
        "mean": float(
            np.mean(arr)
        ),
        "median": float(
            np.median(arr)
        ),
        "count": int(
            arr.size
        ),
    }


def rad_to_deg(x):
    if not np.isfinite(x):
        return np.nan

    return float(
        np.degrees(x)
    )


def summarize(accum):
    output = {}

    for delta, rdict in (
        accum.items()
    ):
        delta_out = {}

        for r, qdict in (
            rdict.items()
        ):
            radius_out = {}

            for q, metrics in (
                qdict.items()
            ):
                center_theta = (
                    summarize_array(
                        metrics[
                            "center_dtheta"
                        ]
                    )
                )

                center_kappa = (
                    summarize_array(
                        metrics[
                            "center_dkappa"
                        ]
                    )
                )

                local_theta = (
                    summarize_array(
                        metrics[
                            "local_dtheta"
                        ]
                    )
                )

                local_kappa = (
                    summarize_array(
                        metrics[
                            "local_dkappa"
                        ]
                    )
                )

                radius_out[q] = {
                    "center_dtheta_mean_deg":
                        rad_to_deg(
                            center_theta[
                                "mean"
                            ]
                        ),

                    "center_dkappa_mean":
                        center_kappa[
                            "mean"
                        ],

                    "local_dtheta_mean_deg":
                        rad_to_deg(
                            local_theta[
                                "mean"
                            ]
                        ),

                    "local_dkappa_mean":
                        local_kappa[
                            "mean"
                        ],

                    "count":
                        local_theta[
                            "count"
                        ],
                }

            delta_out[str(r)] = (
                radius_out
            )

        output[str(delta)] = (
            delta_out
        )

    return output


def increasing(values):
    if not all(
        np.isfinite(x)
        for x in values
    ):
        return False

    return (
        values[0]
        < values[1]
        < values[2]
        < values[3]
    )


def print_summary(
    summary,
    images,
    contours,
    support_radius,
):
    print()
    print("=" * 105)
    print(
        "GT-ONLY SMOOTH LOCAL-SEGMENT "
        "EQUAL-DISPLACEMENT RESULTS"
    )
    print("=" * 105)

    print(
        f"Images          : {images}"
    )
    print(
        f"Contours        : {contours}"
    )
    print(
        f"Radii           : {RADII}"
    )
    print(
        f"Support radius  : {support_radius} contour points"
    )

    print()
    print(
        "PRIMARY  : weighted local-segment |Δkappa|"
    )
    print(
        "SECONDARY: weighted local-segment |Δtheta|"
    )
    print(
        "CENTER   : center-point diagnostics"
    )

    for delta, rdict in (
        summary.items()
    ):
        print()
        print("=" * 105)
        print(
            f"delta = {delta} px"
        )
        print("=" * 105)

        for r in RADII:
            qdict = rdict[
                str(r)
            ]

            print()
            print(
                f"[radius = {r}]"
            )

            local_kappa = []
            local_theta = []

            for q in (
                "Q1",
                "Q2",
                "Q3",
                "Q4",
            ):
                row = qdict[q]

                local_kappa.append(
                    row[
                        "local_dkappa_mean"
                    ]
                )

                local_theta.append(
                    row[
                        "local_dtheta_mean_deg"
                    ]
                )

                print(
                    f"{q} | "
                    f"local Δkappa="
                    f"{row['local_dkappa_mean']:.8f} | "
                    f"local Δtheta="
                    f"{row['local_dtheta_mean_deg']:.6f} deg | "
                    f"center Δkappa="
                    f"{row['center_dkappa_mean']:.8f} | "
                    f"center Δtheta="
                    f"{row['center_dtheta_mean_deg']:.6f} deg | "
                    f"n={row['count']}"
                )

            print(
                "local Δkappa "
                "Q1<Q2<Q3<Q4 : "
                f"{increasing(local_kappa)}"
            )

            print(
                "local Δtheta "
                "Q1<Q2<Q3<Q4 : "
                f"{increasing(local_theta)}"
            )

            if (
                np.isfinite(
                    local_kappa[0]
                )
                and local_kappa[0] > 0
            ):
                print(
                    "local Δkappa Q4/Q1 : "
                    f"{local_kappa[3] / local_kappa[0]:.4f}"
                )

            if (
                np.isfinite(
                    local_theta[0]
                )
                and local_theta[0] > 0
            ):
                print(
                    "local Δtheta Q4/Q1 : "
                    f"{local_theta[3] / local_theta[0]:.4f}"
                )

    print(
        "=" * 105
    )


# ============================================================
# Main
# ============================================================

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
            "gt_equal_displacement_smooth.json"
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
        default=[
            1.0,
            2.0,
            3.0,
        ],
    )

    parser.add_argument(
        "--support-radius",
        type=int,
        default=16,
        help=(
            "Half-width of smooth local contour deformation "
            "in contour points."
        ),
    )

    args = parser.parse_args()

    if args.support_radius < max(RADII):
        raise ValueError(
            "support-radius should be >= max(RADII). "
            f"Got {args.support_radius}, RADII={RADII}"
        )

    cfg = build_cfg(
        args.config_file
    )

    references = (
        load_curvature_references(
            args.curvature_reference
        )
    )

    if len(
        cfg.DATASETS.TEST
    ) != 1:
        raise RuntimeError(
            "Expected exactly one TEST dataset, "
            f"got {cfg.DATASETS.TEST}"
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

    accum = new_accumulator(
        deltas
    )

    processed_images = 0
    processed_contours = 0

    print("=" * 105)
    print(
        "GT-only Smooth Local-Segment "
        "Equal-Displacement Perturbation"
    )
    print("=" * 105)

    print(
        "Dataset         :",
        dataset_name,
    )
    print(
        "Reference       :",
        args.curvature_reference,
    )
    print(
        "Q95             :",
        references,
    )
    print(
        "Radii           :",
        RADII,
    )
    print(
        "Deltas          :",
        deltas,
    )
    print(
        "Support radius  :",
        args.support_radius,
    )
    print(
        "Window          : Hann"
    )
    print(
        "Normal          : per-point multi-scale normal"
    )
    print(
        "Primary metric  : weighted local |Δkappa_r|"
    )
    print(
        "Secondary metric: weighted local |Δtheta_r|"
    )

    print("=" * 105)

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
                    args.support_radius,
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
                f"[{processed_images}] processed"
            )

        if (
            args.max_images
            is not None
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
        args.support_radius,
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "dataset":
            dataset_name,

        "images":
            processed_images,

        "contours":
            processed_contours,

        "deltas_px":
            list(deltas),

        "radii":
            list(RADII),

        "support_radius":
            int(
                args.support_radius
            ),

        "window":
            "hann",

        "perturbation":
            (
                "per_point_multiscale_normal_"
                "smooth_local_segment"
            ),

        "primary_metric":
            (
                "weighted_local_absolute_"
                "curvature_change"
            ),

        "secondary_metric":
            (
                "weighted_local_absolute_"
                "turning_angle_change"
            ),

        "curvature_reference":
            str(
                args.curvature_reference
            ),

        "results":
            summary,
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
