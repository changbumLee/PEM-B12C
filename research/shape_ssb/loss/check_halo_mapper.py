#!/usr/bin/env python3

import torch

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from pem import add_maskformer2_config
from train_net import Trainer


CONFIG = (
    "configs/kroc_curve/"
    "pem_r50_1gpu_baseline.yaml"
)


def main():

    cfg = get_cfg()

    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(
        CONFIG
    )

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.freeze()

    loader = (
        Trainer.build_train_loader(
            cfg
        )
    )

    batch = next(
        iter(loader)
    )

    assert len(batch) == 1

    x = batch[0]

    print("=" * 80)
    print("HALO MAPPER SANITY")
    print("=" * 80)

    print(
        "keys:",
        sorted(x.keys())
    )

    gt = x["sem_seg"]

    halo = x[
        "ssb_sem_seg_halo"
    ]

    halo_px = int(
        x["ssb_halo_px"]
    )

    print(
        "image:",
        tuple(
            x["image"].shape
        )
    )

    print(
        "sem_seg:",
        tuple(
            gt.shape
        )
    )

    print(
        "halo:",
        tuple(
            halo.shape
        )
    )

    print(
        "halo_px:",
        halo_px
    )

    h, w = gt.shape

    expected_shape = (
        h + 2 * halo_px,
        w + 2 * halo_px,
    )

    print(
        "expected halo:",
        expected_shape
    )

    assert tuple(
        halo.shape
    ) == expected_shape

    # --------------------------------------------------------
    # Most important check:
    #
    # center of halo must be EXACTLY the final augmented GT.
    # --------------------------------------------------------

    center = halo[
        halo_px:
        halo_px + h,

        halo_px:
        halo_px + w,
    ]

    mismatch = (
        center != gt
    )

    n_bad = int(
        mismatch.sum()
    )

    print(
        "center mismatch pixels:",
        n_bad
    )

    assert n_bad == 0, (
        "Halo center does not match final sem_seg."
    )

    print()
    print(
        "HALO MAPPER SANITY PASSED"
    )


if __name__ == "__main__":
    main()
