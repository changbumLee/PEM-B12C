
#!/usr/bin/env python3



from __future__ import annotations



import argparse

import json

import os

from concurrent.futures import ProcessPoolExecutor

from pathlib import Path



import cv2

import numpy as np

from PIL import Image



from detectron2.config import get_cfg

from detectron2.data import DatasetCatalog

from detectron2.projects.deeplab import add_deeplab_config



from pem import add_maskformer2_config



from research.shape_ssb.geometry.contour import extract_class_contours

from research.shape_ssb.geometry.stability import contour_shape_score

from research.shape_ssb.geometry.shape_score import load_curvature_references





NUM_CLASSES = 19

IGNORE_LABEL = 255

RADII = (2, 4, 8)

MIN_CONTOUR_POINTS = 16

EPS = 1e-8



Q_NAMES = ("Q1", "Q2", "Q3", "Q4")

METRIC_NAMES = (

    "center_dtheta",

    "center_dkappa",

    "local_dtheta",

    "local_dkappa",

)





# ============================================================

# Config

# ============================================================



def build_cfg(config_file):

    cfg = get_cfg()

    add_deeplab_config(cfg)

    add_maskformer2_config(cfg)

    cfg.merge_from_file(config_file)

    cfg.freeze()

    return cfg





# ============================================================

# Vectorized contour geometry

# ============================================================



def geometry_arrays(points, radius, closed):

    """

    Vectorized equivalent of the current turning geometry.



    Returns

    -------

    theta   : [N]

    kappa   : [N]

    tangent : [N,2]

    valid   : [N]

    """



    pts = np.asarray(points, dtype=np.float64)

    n = len(pts)

    r = int(radius)



    theta = np.full(n, np.nan, dtype=np.float64)

    kappa = np.full(n, np.nan, dtype=np.float64)

    tangent = np.full((n, 2), np.nan, dtype=np.float64)

    valid = np.zeros(n, dtype=bool)



    if n < 2 * r + 1:

        return theta, kappa, tangent, valid



    if closed:

        idx = np.arange(n, dtype=np.int64)

        prev_idx = (idx - r) % n

        next_idx = (idx + r) % n

    else:

        idx = np.arange(r, n - r, dtype=np.int64)

        prev_idx = idx - r

        next_idx = idx + r



    p_prev = pts[prev_idx]

    p_cur = pts[idx]

    p_next = pts[next_idx]



    v_minus = p_cur - p_prev

    v_plus = p_next - p_cur



    norm_minus = np.linalg.norm(v_minus, axis=1)

    norm_plus = np.linalg.norm(v_plus, axis=1)



    tangent_raw = p_next - p_prev

    tangent_norm = np.linalg.norm(tangent_raw, axis=1)



    ok = (

        (norm_minus >= EPS)

        & (norm_plus >= EPS)

        & (tangent_norm >= EPS)

    )



    if not np.any(ok):

        return theta, kappa, tangent, valid



    idx_ok = idx[ok]



    vm = v_minus[ok]

    vp = v_plus[ok]



    nm = norm_minus[ok]

    np_ = norm_plus[ok]



    cosang = (

        np.sum(vm * vp, axis=1)

        / (nm * np_ + EPS)

    )



    th = np.arccos(

        np.clip(cosang, -1.0, 1.0)

    )



    local_length = (

        0.5 * (nm + np_) + EPS

    )



    theta[idx_ok] = th

    kappa[idx_ok] = th / local_length



    tangent[idx_ok] = (

        tangent_raw[ok]

        / tangent_norm[ok, None]

    )



    valid[idx_ok] = True



    return theta, kappa, tangent, valid





def precompute_geometry(points, closed):

    geom = {}



    for r in RADII:

        theta, kappa, tangent, valid = geometry_arrays(

            points,

            r,

            closed,

        )



        geom[r] = {

            "theta": theta,

            "kappa": kappa,

            "tangent": tangent,

            "valid": valid,

        }



    return geom





# ============================================================

# Vectorized multi-scale normal

# ============================================================



def precompute_multiscale_normals(geom):

    """

    Same conceptual rule as previous code:



    - collect valid tangents from r=2,4,8

    - first valid tangent = orientation reference

    - flip opposite tangents

    - average

    - rotate by 90 deg

    """



    n = len(

        geom[RADII[0]]["valid"]

    )



    refs = np.zeros((n, 2), dtype=np.float64)

    have_ref = np.zeros(n, dtype=bool)



    for r in RADII:

        tangent = geom[r]["tangent"]

        valid = geom[r]["valid"]



        pick = valid & (~have_ref)



        refs[pick] = tangent[pick]

        have_ref[pick] = True



    tangent_sum = np.zeros((n, 2), dtype=np.float64)

    tangent_count = np.zeros(n, dtype=np.int32)



    for r in RADII:

        tangent = geom[r]["tangent"]

        valid = geom[r]["valid"] & have_ref



        if not np.any(valid):

            continue



        t = tangent[valid].copy()

        ref = refs[valid]



        dots = np.sum(

            t * ref,

            axis=1,

        )



        t[dots < 0.0] *= -1.0



        tangent_sum[valid] += t

        tangent_count[valid] += 1



    normals = np.full((n, 2), np.nan, dtype=np.float64)



    valid_normal = (

        have_ref

        & (tangent_count > 0)

    )



    mean_tangent = np.zeros(

        (n, 2),

        dtype=np.float64,

    )



    mean_tangent[valid_normal] = (

        tangent_sum[valid_normal]

        / tangent_count[valid_normal, None]

    )



    norm = np.linalg.norm(

        mean_tangent,

        axis=1,

    )



    valid_normal &= norm >= EPS



    unit_tangent = np.zeros_like(

        mean_tangent

    )



    unit_tangent[valid_normal] = (

        mean_tangent[valid_normal]

        / norm[valid_normal, None]

    )



    normals[valid_normal, 0] = (

        -unit_tangent[valid_normal, 1]

    )



    normals[valid_normal, 1] = (

        unit_tangent[valid_normal, 0]

    )



    return normals, valid_normal





# ============================================================

# Quartiles

# ============================================================



def ranked_quartiles(scores):

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



    ranked = valid_idx[order]



    groups = np.array_split(

        ranked,

        4,

    )



    return groups





# ============================================================

# Aggregation

#

# Dimensions:

#   D x R x Q x M

#

# D = #delta

# R = 3 radii

# Q = 4 quartiles

# M = 4 metrics

# ============================================================



def empty_aggregate(num_deltas):

    shape = (

        num_deltas,

        len(RADII),

        4,

        len(METRIC_NAMES),

    )



    return (

        np.zeros(shape, dtype=np.float64),

        np.zeros(shape, dtype=np.int64),

    )





def add_value(

    sums,

    counts,

    di,

    ri,

    qi,

    mi,

    value,

):

    if not np.isfinite(value):

        return



    sums[di, ri, qi, mi] += float(value)

    counts[di, ri, qi, mi] += 1





# ============================================================

# Local displacement helpers

# ============================================================



def make_local_deformation(

    points,

    normals,

    valid_normals,

    center_idx,

    support_radius,

    closed,

):

    """

    Build unit-delta deformation field for one center.



    The actual delta is applied later:



        p' = p + delta * base_displacement



    This avoids rebuilding normals/window for delta=1,2,3.

    """



    n = len(points)

    H = int(support_radius)



    offsets = np.arange(

        -H,

        H + 1,

        dtype=np.int64,

    )



    weights = (

        0.5

        * (

            1.0

            + np.cos(

                np.pi

                * offsets.astype(np.float64)

                / float(H)

            )

        )

    )



    positive = weights > 0.0

    offsets = offsets[positive]

    weights = weights[positive]



    raw_idx = center_idx + offsets



    if closed:

        idx = raw_idx % n

        keep = np.ones(

            len(idx),

            dtype=bool,

        )

    else:

        keep = (

            (raw_idx >= 0)

            & (raw_idx < n)

        )



        idx = raw_idx[keep]

        weights = weights[keep]



    normal_ok = valid_normals[idx]



    idx = idx[normal_ok]

    weights = weights[normal_ok]



    if idx.size == 0:

        return None



    if not valid_normals[center_idx]:

        return None



    base_disp = (

        weights[:, None]

        * normals[idx]

    )



    # Exact semantics for duplicated wrapped indices:

    # later offsets overwrite earlier positions, as in the old code.

    disp_field = {}



    for point_idx, disp in zip(

        idx.tolist(),

        base_disp,

    ):

        disp_field[int(point_idx)] = disp



    return {

        "affected_idx": idx,

        "weights": weights,

        "disp_field": disp_field,

    }





def displaced_coords(

    points,

    indices,

    disp_field,

    delta,

):

    """

    Fetch coordinates after local deformation without copying

    the whole contour.

    """



    indices = np.asarray(

        indices,

        dtype=np.int64,

    )



    out = points[indices].copy()



    for k, idx in enumerate(

        indices.tolist()

    ):

        disp = disp_field.get(

            int(idx)

        )



        if disp is not None:

            out[k] += (

                float(delta) * disp

            )



    return out





# ============================================================

# Vectorized after-perturbation geometry

# ============================================================



def after_geometry_for_indices(

    points,

    target_idx,

    radius,

    closed,

    disp_field,

    delta,

):

    n = len(points)

    r = int(radius)



    target_idx = np.asarray(

        target_idx,

        dtype=np.int64,

    )



    if target_idx.size == 0:

        return (

            np.empty(0, dtype=np.float64),

            np.empty(0, dtype=np.float64),

            np.empty(0, dtype=bool),

        )



    if closed:

        prev_idx = (

            target_idx - r

        ) % n



        next_idx = (

            target_idx + r

        ) % n



        support_ok = np.ones(

            target_idx.size,

            dtype=bool,

        )



    else:

        support_ok = (

            (target_idx - r >= 0)

            & (target_idx + r < n)

        )



        prev_idx = target_idx - r

        next_idx = target_idx + r



    theta = np.full(

        target_idx.size,

        np.nan,

        dtype=np.float64,

    )



    kappa = np.full(

        target_idx.size,

        np.nan,

        dtype=np.float64,

    )



    valid_pos = np.flatnonzero(

        support_ok

    )



    if valid_pos.size == 0:

        return theta, kappa, support_ok



    ti = target_idx[valid_pos]

    pi = prev_idx[valid_pos]

    ni = next_idx[valid_pos]



    p_prev = displaced_coords(

        points,

        pi,

        disp_field,

        delta,

    )



    p_cur = displaced_coords(

        points,

        ti,

        disp_field,

        delta,

    )



    p_next = displaced_coords(

        points,

        ni,

        disp_field,

        delta,

    )



    v_minus = p_cur - p_prev

    v_plus = p_next - p_cur



    len_minus = np.linalg.norm(

        v_minus,

        axis=1,

    )



    len_plus = np.linalg.norm(

        v_plus,

        axis=1,

    )



    ok = (

        (len_minus >= EPS)

        & (len_plus >= EPS)

    )



    if not np.any(ok):

        return theta, kappa, np.zeros_like(

            support_ok

        )



    vm = v_minus[ok]

    vp = v_plus[ok]



    nm = len_minus[ok]

    np_ = len_plus[ok]



    cosang = (

        np.sum(vm * vp, axis=1)

        / (

            nm * np_

            + EPS

        )

    )



    th = np.arccos(

        np.clip(

            cosang,

            -1.0,

            1.0,

        )

    )



    loc_len = (

        0.5

        * (nm + np_)

        + EPS

    )



    final_pos = valid_pos[ok]



    theta[final_pos] = th

    kappa[final_pos] = (

        th / loc_len

    )



    valid = np.isfinite(theta) & np.isfinite(kappa)



    return theta, kappa, valid





# ============================================================

# Fast per-contour evaluation

# ============================================================



def process_contour_fast(

    contour,

    references,

    deltas,

    support_radius,

):

    points = np.asarray(

        contour.points,

        dtype=np.float64,

    )



    n = len(points)



    if n < 4:

        return None



    # --------------------------------------------------------

    # S - same existing implementation

    # --------------------------------------------------------

    shape = contour_shape_score(

        contour,

        references,

        RADII,

    )



    S = np.asarray(

        shape["S"],

        dtype=np.float64,

    )



    quartiles = ranked_quartiles(S)



    if quartiles is None:

        return None



    # --------------------------------------------------------

    # Geometry only ONCE per contour

    # --------------------------------------------------------

    geom = precompute_geometry(

        points,

        contour.closed,

    )



    normals, valid_normals = (

        precompute_multiscale_normals(

            geom

        )

    )



    sums, counts = empty_aggregate(

        len(deltas)

    )



    for qi, centers in enumerate(

        quartiles

    ):



        for center_idx in centers:

            center_idx = int(

                center_idx

            )



            # Same condition as previous version:

            # full r=2,4,8 support at center.

            if not all(

                geom[r]["valid"][center_idx]

                for r in RADII

            ):

                continue



            if not valid_normals[

                center_idx

            ]:

                continue



            deformation = make_local_deformation(

                points,

                normals,

                valid_normals,

                center_idx,

                support_radius,

                contour.closed,

            )



            if deformation is None:

                continue



            affected_idx = deformation[

                "affected_idx"

            ]



            weights = deformation[

                "weights"

            ]



            disp_field = deformation[

                "disp_field"

            ]



            if affected_idx.size == 0:

                continue



            for di, delta in enumerate(

                deltas

            ):



                for ri, r in enumerate(

                    RADII

                ):



                    # -----------------------------------------

                    # Center diagnostics

                    # -----------------------------------------

                    after_theta_c, after_kappa_c, valid_c = (

                        after_geometry_for_indices(

                            points,

                            np.asarray(

                                [center_idx],

                                dtype=np.int64,

                            ),

                            r,

                            contour.closed,

                            disp_field,

                            delta,

                        )

                    )



                    if valid_c[0]:

                        before_theta = (

                            geom[r]["theta"][

                                center_idx

                            ]

                        )



                        before_kappa = (

                            geom[r]["kappa"][

                                center_idx

                            ]

                        )



                        dtheta_c = abs(

                            after_theta_c[0]

                            - before_theta

                        )



                        dkappa_c = abs(

                            after_kappa_c[0]

                            - before_kappa

                        )



                        add_value(

                            sums,

                            counts,

                            di,

                            ri,

                            qi,

                            0,

                            dtheta_c,

                        )



                        add_value(

                            sums,

                            counts,

                            di,

                            ri,

                            qi,

                            1,

                            dkappa_c,

                        )



                    # -----------------------------------------

                    # Local segment distortion

                    # -----------------------------------------

                    after_theta, after_kappa, after_valid = (

                        after_geometry_for_indices(

                            points,

                            affected_idx,

                            r,

                            contour.closed,

                            disp_field,

                            delta,

                        )

                    )



                    before_theta = (

                        geom[r]["theta"][

                            affected_idx

                        ]

                    )



                    before_kappa = (

                        geom[r]["kappa"][

                            affected_idx

                        ]

                    )



                    valid = (

                        after_valid

                        & np.isfinite(

                            before_theta

                        )

                        & np.isfinite(

                            before_kappa

                        )

                    )



                    if not np.any(valid):

                        continue



                    w = weights[valid]



                    w_sum = float(

                        np.sum(w)

                    )



                    if w_sum <= EPS:

                        continue



                    dtheta = np.abs(

                        after_theta[valid]

                        - before_theta[valid]

                    )



                    dkappa = np.abs(

                        after_kappa[valid]

                        - before_kappa[valid]

                    )



                    local_dtheta = float(

                        np.sum(

                            w * dtheta

                        )

                        / w_sum

                    )



                    local_dkappa = float(

                        np.sum(

                            w * dkappa

                        )

                        / w_sum

                    )



                    add_value(

                        sums,

                        counts,

                        di,

                        ri,

                        qi,

                        2,

                        local_dtheta,

                    )



                    add_value(

                        sums,

                        counts,

                        di,

                        ri,

                        qi,

                        3,

                        local_dkappa,

                    )



    return sums, counts





# ============================================================

# Per-image worker

# ============================================================



def process_one_image(args_tuple):

    (

        gt_path,

        references,

        deltas,

        support_radius,

    ) = args_tuple



    # Avoid OpenCV spawning its own thread pool inside

    # multiprocessing workers.

    cv2.setNumThreads(1)



    gt = np.asarray(

        Image.open(gt_path),

        dtype=np.int32,

    )



    total_sums, total_counts = (

        empty_aggregate(

            len(deltas)

        )

    )



    n_contours = 0



    for c in range(

        NUM_CLASSES

    ):

        contours = extract_class_contours(

            gt,

            c,

            IGNORE_LABEL,

            MIN_CONTOUR_POINTS,

        )



        for contour in contours:

            result = process_contour_fast(

                contour,

                references,

                deltas,

                support_radius,

            )



            if result is None:

                continue



            sums, counts = result



            total_sums += sums

            total_counts += counts



            n_contours += 1



    return (

        total_sums,

        total_counts,

        n_contours,

    )





# ============================================================

# Summary

# ============================================================



def safe_mean(

    sums,

    counts,

):

    out = np.full(

        sums.shape,

        np.nan,

        dtype=np.float64,

    )



    valid = counts > 0



    out[valid] = (

        sums[valid]

        / counts[valid]

    )



    return out





def increasing(values):

    return (

        all(np.isfinite(v) for v in values)

        and values[0]

        < values[1]

        < values[2]

        < values[3]

    )





def make_summary(

    means,

    counts,

    deltas,

):

    result = {}



    for di, delta in enumerate(

        deltas

    ):

        d = {}



        for ri, r in enumerate(

            RADII

        ):

            rd = {}



            for qi, q in enumerate(

                Q_NAMES

            ):

                center_dtheta = (

                    means[

                        di, ri, qi, 0

                    ]

                )



                center_dkappa = (

                    means[

                        di, ri, qi, 1

                    ]

                )



                local_dtheta = (

                    means[

                        di, ri, qi, 2

                    ]

                )



                local_dkappa = (

                    means[

                        di, ri, qi, 3

                    ]

                )



                rd[q] = {

                    "center_dtheta_mean_deg":

                        float(

                            np.degrees(

                                center_dtheta

                            )

                        )

                        if np.isfinite(

                            center_dtheta

                        )

                        else np.nan,



                    "center_dkappa_mean":

                        float(

                            center_dkappa

                        ),



                    "local_dtheta_mean_deg":

                        float(

                            np.degrees(

                                local_dtheta

                            )

                        )

                        if np.isfinite(

                            local_dtheta

                        )

                        else np.nan,



                    "local_dkappa_mean":

                        float(

                            local_dkappa

                        ),



                    "count":

                        int(

                            counts[

                                di, ri, qi, 2

                            ]

                        ),

                }



            d[str(r)] = rd



        result[str(delta)] = d



    return result





def print_summary(

    summary,

    n_images,

    n_contours,

    support_radius,

):

    print()

    print("=" * 105)

    print(

        "GT-ONLY FAST SMOOTH LOCAL-SEGMENT "

        "EQUAL-DISPLACEMENT RESULTS"

    )

    print("=" * 105)



    print(

        f"Images         : {n_images}"

    )

    print(

        f"Contours       : {n_contours}"

    )

    print(

        f"Radii          : {RADII}"

    )

    print(

        f"Support radius : {support_radius}"

    )



    print()

    print(

        "PRIMARY  : weighted local-segment |Δkappa|"

    )

    print(

        "SECONDARY: weighted local-segment |Δtheta|"

    )



    for delta, rdict in summary.items():



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



            for q in Q_NAMES:

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

                np.isfinite(local_kappa[0])

                and local_kappa[0] > 0

            ):

                print(

                    "local Δkappa Q4/Q1 : "

                    f"{local_kappa[3] / local_kappa[0]:.4f}"

                )



            if (

                np.isfinite(local_theta[0])

                and local_theta[0] > 0

            ):

                print(

                    "local Δtheta Q4/Q1 : "

                    f"{local_theta[3] / local_theta[0]:.4f}"

                )



    print("=" * 105)





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

            "gt_equal_displacement_smooth_fast.json"

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

        nargs="+",

        type=float,

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

    )



    parser.add_argument(

        "--workers",

        type=int,

        default=min(

            8,

            os.cpu_count() or 1,

        ),

        help=(

            "Image-level CPU worker processes. "

            "Start with 4 or 8."

        ),

    )



    args = parser.parse_args()



    if args.support_radius < max(

        RADII

    ):

        raise ValueError(

            "support-radius must be "

            ">= max(RADII)"

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

            "Expected one TEST dataset"

        )



    dataset_name = (

        cfg.DATASETS.TEST[0]

    )



    dataset = DatasetCatalog.get(

        dataset_name

    )



    records = list(dataset)



    if args.max_images is not None:

        records = records[

            :args.max_images

        ]



    gt_paths = []



    for record in records:

        gt_path = record.get(

            "sem_seg_file_name"

        )



        if gt_path is None:

            raise KeyError(

                "sem_seg_file_name missing"

            )



        gt_paths.append(

            gt_path

        )



    deltas = tuple(

        float(x)

        for x in args.deltas

    )



    print("=" * 105)

    print(

        "GT-only FAST Smooth Local-Segment "

        "Equal-Displacement Perturbation"

    )

    print("=" * 105)



    print(

        "Dataset         :",

        dataset_name,

    )

    print(

        "Images          :",

        len(gt_paths),

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

        "Workers         :",

        args.workers,

    )

    print(

        "Primary metric  : weighted local |Δkappa_r|"

    )

    print(

        "Secondary metric: weighted local |Δtheta_r|"

    )

    print("=" * 105)



    total_sums, total_counts = (

        empty_aggregate(

            len(deltas)

        )

    )



    total_contours = 0



    jobs = [

        (

            path,

            references,

            deltas,

            args.support_radius,

        )

        for path in gt_paths

    ]



    if args.workers <= 1:



        iterator = map(

            process_one_image,

            jobs,

        )



        for i, result in enumerate(

            iterator,

            1,

        ):

            sums, counts, n_contours = result



            total_sums += sums

            total_counts += counts

            total_contours += n_contours



            if (

                i == 1

                or (

                    args.print_period > 0

                    and i

                    % args.print_period

                    == 0

                )

            ):

                print(

                    f"[{i}/{len(jobs)}] processed"

                )



    else:



        with ProcessPoolExecutor(

            max_workers=args.workers

        ) as executor:



            # map preserves dataset order.

            iterator = executor.map(

                process_one_image,

                jobs,

                chunksize=1,

            )



            for i, result in enumerate(

                iterator,

                1,

            ):

                sums, counts, n_contours = result



                total_sums += sums

                total_counts += counts

                total_contours += n_contours



                if (

                    i == 1

                    or (

                        args.print_period > 0

                        and i

                        % args.print_period

                        == 0

                    )

                ):

                    print(

                        f"[{i}/{len(jobs)}] processed"

                    )



    means = safe_mean(

        total_sums,

        total_counts,

    )



    summary = make_summary(

        means,

        total_counts,

        deltas,

    )



    print_summary(

        summary,

        len(gt_paths),

        total_contours,

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

            len(gt_paths),



        "contours":

            total_contours,



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



        "workers":

            int(args.workers),



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

