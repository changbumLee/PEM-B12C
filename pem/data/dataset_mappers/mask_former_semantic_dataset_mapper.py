# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
from pathlib import Path

from research.shape_ssb.geometry.shape_score import (
    load_curvature_references,
)
from research.shape_ssb.loss.ssb_halo_target import (
    build_halo_ssb_target,
)
__all__ = ["MaskFormerSemanticDatasetMapper"]


class MaskFormerSemanticDatasetMapper:
    """
    MaskFormer semantic mapper with an additional SSB halo GT tensor.

    Original augmentation order is preserved:
        Resize -> Crop -> ColorAug -> Flip

    For SSB only, semantic context is captured immediately before crop,
    expanded by a fixed halo, padded with ignore_label at image borders,
    and then transformed by the same flip as the final training crop.
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        resize_augmentation,
        crop_augmentation,
        color_augmentation,
        flip_augmentation,
        image_format,
        ignore_label,
        size_divisibility,
        ssb_halo_px=64,
        ssb_tau=0.0644756630063057,
        ssb_d_max=32.0,
        ssb_weight_lambda=2.0,
        ssb_gamma=2.0,
        ssb_mode="full",
        ssb_enabled=True,
    ):  
        self.is_train = is_train
        self.resize_augmentation = resize_augmentation
        self.crop_augmentation = crop_augmentation
        self.color_augmentation = color_augmentation
        self.flip_augmentation = flip_augmentation
        self.img_format = image_format
        self.ignore_label = int(ignore_label)
        self.size_divisibility = size_divisibility
        self.ssb_halo_px = int(ssb_halo_px)
        self.ssb_tau = float(ssb_tau)
        self.ssb_d_max = float(ssb_d_max)
        self.ssb_weight_lambda = float(ssb_weight_lambda)
        self.ssb_gamma = float(ssb_gamma)
        self.ssb_mode = str(ssb_mode).lower()
        self.ssb_enabled = bool(ssb_enabled)
        if self.ssb_mode not in ("uniform", "curvature", "full"):
            raise ValueError(
                f"Unknown SSB mode: {self.ssb_mode}"
            )
        project_root = Path(__file__).resolve().parents[3]

        self.ssb_curvature_reference_path = (
            project_root
            / "research"
            / "shape_ssb"
            / "geometry"
            / "curvature_reference_q95.json"
        )

        self.ssb_references = load_curvature_references(
            self.ssb_curvature_reference_path
        )

        if self.ssb_halo_px < 0:
            raise ValueError("ssb_halo_px must be >= 0")

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode}: "
            f"[{resize_augmentation}, {crop_augmentation}, "
            f"{color_augmentation}, {flip_augmentation}]"
        )
        logger.info(
            f"[{self.__class__.__name__}] SSB halo: {self.ssb_halo_px}px"
        )

    @classmethod
    def from_config(cls, cfg, is_train=True):
        resize_augmentation = T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )

        crop_augmentation = None
        if cfg.INPUT.CROP.ENABLED:
            crop_augmentation = T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )

        color_augmentation = None
        if cfg.INPUT.COLOR_AUG_SSD:
            color_augmentation = ColorAugSSDTransform(
                img_format=cfg.INPUT.FORMAT
            )

        flip_augmentation = T.RandomFlip()

        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        return {
            "is_train": is_train,
            "resize_augmentation": resize_augmentation,
            "crop_augmentation": crop_augmentation,
            "color_augmentation": color_augmentation,
            "flip_augmentation": flip_augmentation,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "ssb_halo_px": 64,

            # Frozen SSB geometry settings
            "ssb_tau": 0.0644756630063057,
            "ssb_d_max": 32.0,
            "ssb_weight_lambda": 2.0,
            "ssb_gamma": 2.0,
            "ssb_mode": cfg.MODEL.SSB.MODE,
            "ssb_enabled": bool(cfg.MODEL.SSB.ENABLED),
        }

    def _extract_fixed_halo(self, sem_seg_pre_crop, crop_transform):
        """
        Extract a fixed-width halo around the chosen crop from resized GT.
        Out-of-image regions are padded with ignore_label.
        """
        if crop_transform is None:
            h, w = sem_seg_pre_crop.shape[:2]
            x0 = 0
            y0 = 0
            crop_w = w
            crop_h = h
        else:
            required = ("x0", "y0", "w", "h")
            if not all(hasattr(crop_transform, k) for k in required):
                raise TypeError(
                    "Expected CropTransform with x0/y0/w/h, got "
                    f"{type(crop_transform)}"
                )
            x0 = int(crop_transform.x0)
            y0 = int(crop_transform.y0)
            crop_w = int(crop_transform.w)
            crop_h = int(crop_transform.h)

        halo = self.ssb_halo_px
        out_h = crop_h + 2 * halo
        out_w = crop_w + 2 * halo

        out = np.full(
            (out_h, out_w),
            self.ignore_label,
            dtype=sem_seg_pre_crop.dtype,
        )

        src_h, src_w = sem_seg_pre_crop.shape[:2]

        src_x0 = max(0, x0 - halo)
        src_y0 = max(0, y0 - halo)
        src_x1 = min(src_w, x0 + crop_w + halo)
        src_y1 = min(src_h, y0 + crop_h + halo)

        dst_x0 = src_x0 - (x0 - halo)
        dst_y0 = src_y0 - (y0 - halo)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        out[dst_y0:dst_y1, dst_x0:dst_x1] = (
            sem_seg_pre_crop[src_y0:src_y1, src_x0:src_x1]
        )
        return out

    def __call__(self, dataset_dict):
        assert self.is_train, (
            "MaskFormerSemanticDatasetMapper should only be used for training!"
        )

        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(
            dataset_dict["file_name"], format=self.img_format
        )
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name")
            ).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation "
                f"dataset {dataset_dict['file_name']}."
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)

        # 1) Resize
        aug_input, _ = T.apply_transform_gens(
            [self.resize_augmentation], aug_input
        )
        sem_seg_pre_crop = np.asarray(aug_input.sem_seg).copy()

        # 2) Crop, while retaining pre-crop halo context
        crop_transform = None
        if self.crop_augmentation is not None:
            aug_input, crop_transforms = T.apply_transform_gens(
                [self.crop_augmentation], aug_input
            )
            if len(crop_transforms.transforms) != 1:
                raise RuntimeError(
                    "Expected exactly one crop transform, got "
                    f"{len(crop_transforms.transforms)}"
                )
            crop_transform = crop_transforms.transforms[0]

        ssb_sem_seg_halo = self._extract_fixed_halo(
            sem_seg_pre_crop, crop_transform
        )

        # 3) Color augmentation
        if self.color_augmentation is not None:
            aug_input, _ = T.apply_transform_gens(
                [self.color_augmentation], aug_input
            )

        # 4) Flip; apply identical semantic flip to halo
        if self.flip_augmentation is not None:
            aug_input, flip_transforms = T.apply_transform_gens(
                [self.flip_augmentation], aug_input
            )
            ssb_sem_seg_halo = flip_transforms.apply_segmentation(
                ssb_sem_seg_halo
            )

        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        image = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        ssb_sem_seg_halo = torch.as_tensor(
            np.ascontiguousarray(ssb_sem_seg_halo.astype("long"))
        )
        # ------------------------------------------------------------
        # SSB GT-side target generation
        # CPU / DataLoader worker only.
        # ------------------------------------------------------------

        crop_h = int(sem_seg_gt.shape[-2])
        crop_w = int(sem_seg_gt.shape[-1])

        ssb_target = build_halo_ssb_target(
            halo_gt=ssb_sem_seg_halo.numpy(),
            halo_px=self.ssb_halo_px,
            crop_shape=(crop_h, crop_w),
            references=self.ssb_references,
            num_classes=19,
            ignore_label=self.ignore_label,
            min_contour_points=16,
            radii=(2, 4, 8),
            d_max=self.ssb_d_max,
            weight_lambda=self.ssb_weight_lambda,
            gamma=self.ssb_gamma,
            tau=self.ssb_tau,
            mode=self.ssb_mode,
            compute_distance_fields=self.ssb_enabled,
        )

        # ssb_phi = torch.from_numpy(
        #     np.ascontiguousarray(
        #         ssb_target.phi.astype(np.float32)
        #     )
        # )
        ssb_phi = None

        if self.ssb_enabled:
            ssb_phi = torch.from_numpy(
                np.ascontiguousarray(
                    ssb_target.phi.astype(np.float32)
                )
            )
        ssb_weight = torch.from_numpy(
            np.ascontiguousarray(
                ssb_target.weight.astype(np.float32)
            )
        )
        ssb_valid_classes = torch.from_numpy(
            np.ascontiguousarray(
                ssb_target.valid_classes.astype(np.bool_)
            )
        )
        ssb_boundary = torch.from_numpy(
            np.ascontiguousarray(
                ssb_target.boundary.astype(np.float32)
            )
        )
        # ------------------------------------------------------------
        # CSM shape-critical boundary target
        # ------------------------------------------------------------

        # Remove classes that are not valid supervision classes.
        class_valid = (
            ssb_valid_classes[:, None, None]
            .to(dtype=torch.float32)
        )

        ssb_boundary_for_csm = (
            ssb_boundary
            * class_valid
        )

        # Recover normalized shape sensitivity on [0, 1].
        # Current frozen geometry uses:
        #   W = 1 + lambda_w * R
        # therefore:
        #   R = (W - 1) / lambda_w
        # shape_sensitivity = (
        #     (ssb_weight - 1.0)
        #     / float(self.ssb_weight_lambda)
        # ).clamp(
        #     min=0.0,
        #     max=1.0,
        # )

        # # Keep shape sensitivity only on actual GT contours.
        # class_shape_target = (
        #     ssb_boundary_for_csm
        #     * shape_sensitivity
        # )

        # # Class-agnostic target for pre-semantic CSM feature.
        # csm_shape_target = class_shape_target.amax(
        #     dim=0,
        #     keepdim=True,
        # )

        # # Binary GT boundary used to construct the local supervision band.
        # csm_boundary = (
        #     ssb_boundary_for_csm.amax(
        #         dim=0,
        #         keepdim=True,
        #     )
        #     > 0.0
        # )
        # ============================================================
        # CSM-B3:
        # Shape-Weighted Distance Profile
        #
        # Target:
        #   all valid GT boundaries receive a positive proximity target.
        #
        # Weight:
        #   shape-sensitive regions receive stronger supervision.
        #
        # Important:
        #   target meaning and shape importance are deliberately separated.
        # ============================================================

        # ============================================================
        # CSM-B4:
        # Exact-Contour Shape-Weighted Supervision
        #
        # Target:
        #   every valid GT contour pixel has target = 1.
        #
        # Weight:
        #   shape-sensitive GT contour pixels receive stronger loss.
        #
        # Important:
        #   no positive distance halo is used.
        #   nearby non-boundary pixels remain negatives.
        # ============================================================

        # ============================================================
        # CSM-B5:
        # Floor-Calibrated Exact-Contour Target
        #
        # Motivation:
        #   B2: T = B * S
        #       preserves shape sensitivity but can make a real
        #       GT contour target arbitrarily close to zero.
        #
        #   B4: T = B
        #       restores all contour targets to one, but removes
        #       shape sensitivity from the target itself.
        #
        # B5:
        #   T = B * [alpha + (1 - alpha) * S]
        #
        # With alpha = 0.5:
        #   low-shape contour  -> target >= 0.5
        #   high-shape contour -> target -> 1.0
        #
        # Loss weighting is uniform.
        # No positive distance halo is used.
        # ============================================================

        if self.ssb_weight_lambda <= 0:
            raise ValueError(
                "ssb_weight_lambda must be > 0 for CSM shape supervision"
            )

        # ------------------------------------------------------------
        # Recover normalized shape sensitivity S in [0, 1].
        #
        # Frozen geometry:
        #   W_ssb = 1 + lambda_w * S
        #
        # therefore:
        #   S = (W_ssb - 1) / lambda_w
        # ------------------------------------------------------------
        shape_sensitivity = (
            (ssb_weight - 1.0)
            / float(self.ssb_weight_lambda)
        ).clamp(
            min=0.0,
            max=1.0,
        )

        # ------------------------------------------------------------
        # Binary union of all valid GT contours.
        # [C,H,W] -> [1,H,W]
        # ------------------------------------------------------------
        csm_boundary = (
            torch.amax(
                ssb_boundary_for_csm,
                dim=0,
                keepdim=True,
            )
            > 0.0
        )

        # ------------------------------------------------------------
        # B5 target floor.
        #
        # Development setting for the first B5 run.
        # Keep fixed at 0.5; do NOT sweep yet.
        # ------------------------------------------------------------
         # ------------------------------------------------------------
        # B6 target: EXACT B2 target semantics
        #
        #   T_c = B_c * S_c
        #
        # B6 does NOT modify the target.
        # The only B6 change is in the auxiliary loss:
        # contour responses already >= T are not pushed downward.
        # ------------------------------------------------------------
        class_shape_target = (
            ssb_boundary_for_csm
            * shape_sensitivity
        )

        # ------------------------------------------------------------
        # CSM feature is class-agnostic:
        # merge class-wise targets by maximum.
        #
        # [C,H,W] -> [1,H,W]
        # ------------------------------------------------------------
        csm_shape_target = torch.amax(
            class_shape_target,
            dim=0,
            keepdim=True,
        )

        # B2/B6 use uniform pixel weighting.
        csm_shape_weight = torch.ones_like(
            csm_shape_target,
            dtype=torch.float32,
        )

        # ------------------------------------------------------------
        # B6 / B2-target sanity checks
        # ------------------------------------------------------------
        if not torch.isfinite(
            csm_shape_target
        ).all():
            raise RuntimeError(
                "Non-finite CSM-B6 target"
            )

        if (
            csm_shape_target.min() < 0.0
            or csm_shape_target.max() > 1.0
        ):
            raise RuntimeError(
                "CSM-B6 target outside [0,1]"
            )

        if not torch.all(
            csm_shape_weight == 1.0
        ):
            raise RuntimeError(
                "CSM-B6 pixel weight must be uniformly 1"
            )

        # Target support must remain exact GT contours only.
        non_boundary = ~csm_boundary

        if torch.any(
            csm_shape_target[non_boundary] != 0.0
        ):
            raise RuntimeError(
                "CSM-B6 target is non-zero outside GT contour"
            )

        # Since T = B*S and S in [0,1], target cannot exceed B.
        if torch.any(
            csm_shape_target
            > csm_boundary.to(
                dtype=csm_shape_target.dtype
            )
        ):
            raise RuntimeError(
                "CSM-B6 target exceeds binary contour support"
            )
#---------------------------------_____**************************_________________*(_______)
        # # b4 version
        # if self.ssb_weight_lambda <= 0:
        #     raise ValueError(
        #         "ssb_weight_lambda must be > 0 for CSM shape supervision"
        #     )

        # # ------------------------------------------------------------
        # # Recover normalized shape sensitivity S in [0,1].
        # #
        # # Frozen geometry:
        # #   W_ssb = 1 + lambda_w * S
        # #
        # # therefore:
        # #   S = (W_ssb - 1) / lambda_w
        # # ------------------------------------------------------------
        # shape_sensitivity = (
        #     (ssb_weight - 1.0)
        #     / float(self.ssb_weight_lambda)
        # ).clamp(
        #     min=0.0,
        #     max=1.0,
        # )

        # # ------------------------------------------------------------
        # # Binary union of all valid GT contours.
        # #
        # # [C,H,W] -> [1,H,W]
        # # ------------------------------------------------------------
        # csm_boundary = torch.amax(
        #     ssb_boundary_for_csm,
        #     dim=0,
        #     keepdim=True,
        # ) > 0.0

        # # ------------------------------------------------------------
        # # B4 target:
        # #
        # # exact contour only.
        # #
        # # contour     -> 1
        # # nearby band -> 0
        # # ------------------------------------------------------------
        # csm_shape_target = (
        #     csm_boundary
        #     .to(dtype=torch.float32)
        # )

        # # ------------------------------------------------------------
        # # Shape-sensitive contour importance.
        # #
        # # Only REAL contour pixels are shape-weighted.
        # #
        # # low-shape contour:
        # #   weight ~= 1
        # #
        # # high-shape contour:
        # #   weight -> 2
        # #
        # # non-contour band pixels:
        # #   weight = 1
        # # ------------------------------------------------------------
        # class_shape_emphasis = (
        #     ssb_boundary_for_csm
        #     * shape_sensitivity
        # )

        # shape_emphasis = torch.amax(
        #     class_shape_emphasis,
        #     dim=0,
        #     keepdim=True,
        # )

        # csm_shape_weight = (
        #     1.0
        #     + shape_emphasis
        # )
#_______________________________________________________________________________________________________________________________


        # ------------------------------------------------------------
        # Safety checks
        # ------------------------------------------------------------
        # if not torch.isfinite(csm_shape_target).all():
        #     raise RuntimeError(
        #         "Non-finite CSM-B4 target"
        #     )

        # if not torch.isfinite(csm_shape_weight).all():
        #     raise RuntimeError(
        #         "Non-finite CSM-B4 weight"
        #     )

        # if (
        #     csm_shape_target.min() < 0.0
        #     or csm_shape_target.max() > 1.0
        # ):
        #     raise RuntimeError(
        #         "CSM-B4 target outside [0,1]"
        #     )

        # if (
        #     csm_shape_weight.min() < 1.0
        #     or csm_shape_weight.max() > 2.0
        # ):
        #     raise RuntimeError(
        #         "CSM-B4 weight outside [1,2]"
        #     )
        ssb_valid_mask = torch.from_numpy(
            np.ascontiguousarray(
                ssb_target.valid_mask.astype(np.bool_)
            )
        )
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            sem_seg_gt = F.pad(
                sem_seg_gt,
                padding_size,
                value=self.ignore_label,
            ).contiguous()
            # ssb_phi = F.pad(
            #     ssb_phi,
            #     padding_size,
            #     value=0.0,
            # ).contiguous()
            # ssb_weight = F.pad(
            #     ssb_weight,
            #     padding_size,
            #     value=0.0,
            # ).contiguous()
            if self.ssb_enabled:
                ssb_phi = F.pad(
                    ssb_phi,
                    padding_size,
                    value=0.0,
                ).contiguous()

                ssb_weight = F.pad(
                    ssb_weight,
                    padding_size,
                    value=0.0,
                ).contiguous()
            ssb_valid_mask = F.pad(
                ssb_valid_mask,
                padding_size,
                value=False,
            ).contiguous()
            csm_shape_target = F.pad(
                csm_shape_target,
                padding_size,
                value=0.0,
            ).contiguous()

            csm_shape_weight = F.pad(
                csm_shape_weight,
                padding_size,
                value=1.0,
            ).contiguous()

            csm_boundary = F.pad(
                csm_boundary,
                padding_size,
                value=False,
            ).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])

        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = sem_seg_gt.long()
        dataset_dict["ssb_sem_seg_halo"] = ssb_sem_seg_halo.long()
        dataset_dict["ssb_halo_px"] = self.ssb_halo_px
        # dataset_dict["ssb_phi"] = ssb_phi.float()
        # dataset_dict["ssb_weight"] = ssb_weight.float()

        # dataset_dict["ssb_valid_classes"] = ssb_valid_classes.bool()
        # dataset_dict["ssb_valid_mask"] = ssb_valid_mask.bool()
        if self.ssb_enabled:
            dataset_dict["ssb_phi"] = ssb_phi.float()
            dataset_dict["ssb_weight"] = ssb_weight.float()
            dataset_dict["ssb_valid_classes"] = ssb_valid_classes.bool()

        dataset_dict["ssb_valid_mask"] = ssb_valid_mask.bool()
        dataset_dict["csm_shape_target"] = (
            csm_shape_target.float()
        )
        dataset_dict["csm_shape_weight"] = (
            csm_shape_weight.float()
        )
        dataset_dict["csm_boundary"] = (
            csm_boundary.bool()
        )
        if "annotations" in dataset_dict:
            raise ValueError(
                "Semantic segmentation dataset should not have 'annotations'."
            )

        sem_seg_np = sem_seg_gt.numpy()
        instances = Instances(image_shape)
        classes = np.unique(sem_seg_np)
        classes = classes[classes != self.ignore_label]
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

        masks = []
        for class_id in classes:
            masks.append(sem_seg_np == class_id)

        if len(masks) == 0:
            instances.gt_masks = torch.zeros(
                (0, sem_seg_np.shape[-2], sem_seg_np.shape[-1])
            )
        else:
            masks = BitMasks(
                torch.stack(
                    [
                        torch.from_numpy(
                            np.ascontiguousarray(x.copy())
                        )
                        for x in masks
                    ]
                )
            )
            instances.gt_masks = masks.tensor

        dataset_dict["instances"] = instances
        return dataset_dict
