#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from pem import add_maskformer2_config
from train_net import Trainer


IGNORE_LABEL = 255


def load_gt(x):
    if "sem_seg" in x:
        gt = x["sem_seg"]

        if torch.is_tensor(gt):
            return gt.detach().long()

        return torch.as_tensor(
            np.asarray(gt),
            dtype=torch.long,
        )

    return torch.as_tensor(
        np.asarray(
            Image.open(x["sem_seg_file_name"]),
            dtype=np.int64,
        ),
        dtype=torch.long,
    )


def semantic_boundary(gt):
    """
    gt: [H, W]
    return:
        boundary: [H, W] bool
        valid:    [H, W] bool
    """
    valid = gt != IGNORE_LABEL
    boundary = torch.zeros_like(
        valid,
        dtype=torch.bool,
    )

    # horizontal class transitions
    pair_valid = valid[:, 1:] & valid[:, :-1]
    diff = pair_valid & (gt[:, 1:] != gt[:, :-1])

    boundary[:, 1:] |= diff
    boundary[:, :-1] |= diff

    # vertical class transitions
    pair_valid = valid[1:, :] & valid[:-1, :]
    diff = pair_valid & (gt[1:, :] != gt[:-1, :])

    boundary[1:, :] |= diff
    boundary[:-1, :] |= diff

    return boundary, valid


class RunningMean:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def add(self, values):
        if values.numel() == 0:
            return

        self.sum += float(
            values.double().sum().item()
        )
        self.count += int(values.numel())

    def mean(self):
        if self.count == 0:
            return float("nan")
        return self.sum / self.count


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config-file",
        required=True,
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
    )
    ap.add_argument(
        "--output",
        required=True,
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=None,
    )

    args = ap.parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(
        args.config_file
    )

    cfg.MODEL.WEIGHTS = args.checkpoint
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    model = Trainer.build_model(cfg)

    DetectionCheckpointer(
        model
    ).load(
        args.checkpoint
    )

    model.eval()

    # ---------------------------------------------------------
    # Find B12 SCDA module
    # ---------------------------------------------------------
    matches = []

    for name, module in model.named_modules():
        if name.endswith("scda"):
            matches.append(
                (name, module)
            )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one SCDA module, "
            f"got {[x[0] for x in matches]}"
        )

    scda_name, scda = matches[0]

    print("SCDA module:", scda_name)

    cache = {}

    # B12SCDA forward:
    #
    # inputs:
    #   fine_feat
    #   coarse_up
    #
    # output:
    #   fused
    #   gate
    #   offset
    #
    def hook_fn(module, inputs, output):
        fine_feat = inputs[0]
        coarse_up = inputs[1]

        fused, gate, offset = output

        cache["fine"] = fine_feat.detach()
        cache["coarse"] = coarse_up.detach()
        cache["fused"] = fused.detach()
        cache["gate"] = gate.detach()
        cache["offset"] = offset.detach()

    handle = scda.register_forward_hook(
        hook_fn
    )

    dataset_name = cfg.DATASETS.TEST[0]

    loader = Trainer.build_test_loader(
        cfg,
        dataset_name,
    )

    gate_all = RunningMean()
    gate_boundary = RunningMean()
    gate_far = RunningMean()

    offset_component_abs = RunningMean()
    offset_mag_all = RunningMean()
    offset_mag_boundary = RunningMean()
    offset_mag_far = RunningMean()

    corr_all = RunningMean()
    corr_boundary = RunningMean()
    corr_far = RunningMean()

    high_gate_count = 0
    gate_count = 0

    saturated_offset_count = 0
    offset_component_count = 0

    correction_abs_sum = 0.0
    base_abs_sum = 0.0

    quantile_samples = []

    images_done = 0

    with torch.no_grad():
        for batch in loader:

            cache.clear()

            _ = model(batch)

            if not cache:
                raise RuntimeError(
                    "SCDA forward hook did not fire."
                )

            fine = cache["fine"].float()
            coarse = cache["coarse"].float()
            fused = cache["fused"].float()
            gate = cache["gate"].float()
            offset = cache["offset"].float()

            # Exact feature perturbation introduced by SCDA.
            #
            # fused = fine + coarse + correction
            correction = (
                fused
                - fine
                - coarse
            )

            base = fine + coarse

            corr_mag = (
                correction.abs()
                .mean(
                    dim=1,
                    keepdim=True,
                )
            )

            offset_mag = torch.sqrt(
                offset[:, 0:1].pow(2)
                + offset[:, 1:2].pow(2)
            )

            B, _, Hf, Wf = gate.shape

            for i in range(B):
                if (
                    args.max_images is not None
                    and images_done >= args.max_images
                ):
                    break

                gt = load_gt(
                    batch[i]
                ).to(
                    gate.device
                )

                boundary, valid = (
                    semantic_boundary(gt)
                )

                boundary = boundary[
                    None, None
                ].float()

                valid = valid[
                    None, None
                ].float()

                # Preserve thin GT boundaries when projecting
                # to stride-4 SCDA resolution.
                boundary_small = (
                    F.adaptive_max_pool2d(
                        boundary,
                        output_size=(Hf, Wf),
                    )
                    > 0.0
                )

                # 3x3 feature-space boundary neighborhood.
                # At stride 4 this is approximately a local
                # boundary neighborhood rather than exact contour only.
                boundary_band = (
                    F.max_pool2d(
                        boundary_small.float(),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                    > 0.0
                )

                valid_small = (
                    F.adaptive_avg_pool2d(
                        valid,
                        output_size=(Hf, Wf),
                    )
                    > 0.999
                )

                exact_mask = (
                    boundary_small
                    & valid_small
                )

                far_mask = (
                    valid_small
                    & (~boundary_band)
                )

                g = gate[i:i+1]
                om = offset_mag[i:i+1]
                cm = corr_mag[i:i+1]

                gate_all.add(
                    g[valid_small]
                )
                gate_boundary.add(
                    g[exact_mask]
                )
                gate_far.add(
                    g[far_mask]
                )

                offset_mag_all.add(
                    om[valid_small]
                )
                offset_mag_boundary.add(
                    om[exact_mask]
                )
                offset_mag_far.add(
                    om[far_mask]
                )

                corr_all.add(
                    cm[valid_small]
                )
                corr_boundary.add(
                    cm[exact_mask]
                )
                corr_far.add(
                    cm[far_mask]
                )

                off_comp = (
                    offset[i:i+1]
                    .abs()
                )

                offset_component_abs.add(
                    off_comp
                )

                saturated_offset_count += int(
                    (off_comp > 0.95)
                    .sum()
                    .item()
                )

                offset_component_count += int(
                    off_comp.numel()
                )

                high_gate_count += int(
                    (
                        g[valid_small]
                        > 0.90
                    )
                    .sum()
                    .item()
                )

                gate_count += int(
                    valid_small.sum().item()
                )

                correction_abs_sum += float(
                    correction[
                        i:i+1
                    ].abs().sum().item()
                )

                base_abs_sum += float(
                    base[
                        i:i+1
                    ].abs().sum().item()
                )

                # Deterministic subsample for quantiles.
                vals = (
                    g[valid_small]
                    .flatten()
                )

                if vals.numel() > 0:
                    step = max(
                        1,
                        vals.numel() // 4096,
                    )

                    quantile_samples.append(
                        vals[::step][
                            :4096
                        ].cpu()
                    )

                images_done += 1

            if (
                args.max_images is not None
                and images_done >= args.max_images
            ):
                break

            if images_done % 50 == 0:
                print(
                    f"[{images_done}] "
                    f"gate={gate_all.mean():.4f}, "
                    f"boundary={gate_boundary.mean():.4f}, "
                    f"far={gate_far.mean():.4f}"
                )

    handle.remove()

    samples = torch.cat(
        quantile_samples,
        dim=0,
    )

    q = torch.quantile(
        samples,
        torch.tensor(
            [0.10, 0.50, 0.90],
            dtype=samples.dtype,
        ),
    )

    result = {
        "images": images_done,

        "gate_mean": gate_all.mean(),
        "gate_q10": float(q[0]),
        "gate_q50": float(q[1]),
        "gate_q90": float(q[2]),
        "gate_frac_gt_0.90": (
            high_gate_count
            / max(gate_count, 1)
        ),

        "gate_boundary_mean":
            gate_boundary.mean(),

        "gate_far_mean":
            gate_far.mean(),

        "gate_boundary_minus_far":
            gate_boundary.mean()
            - gate_far.mean(),

        "offset_component_abs_mean":
            offset_component_abs.mean(),

        "offset_magnitude_mean":
            offset_mag_all.mean(),

        "offset_magnitude_boundary_mean":
            offset_mag_boundary.mean(),

        "offset_magnitude_far_mean":
            offset_mag_far.mean(),

        "offset_component_frac_gt_0.95": (
            saturated_offset_count
            / max(offset_component_count, 1)
        ),

        "correction_abs_mean":
            corr_all.mean(),

        "correction_boundary_mean":
            corr_boundary.mean(),

        "correction_far_mean":
            corr_far.mean(),

        "correction_boundary_minus_far":
            corr_boundary.mean()
            - corr_far.mean(),

        "correction_to_base_l1_ratio": (
            correction_abs_sum
            / max(base_abs_sum, 1e-12)
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("Saved:", out)


if __name__ == "__main__":
    main()