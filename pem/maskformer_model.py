# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from research.shape_ssb.loss.csm_shape_loss import (
    csm_shape_aux_loss,
    csm_boundary_presence_loss,
)
from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from detectron2.utils.events import get_event_storage

from research.shape_ssb.loss.ssb_loss import (
    SSBLoss,
    ssb_warmup_weight,
)

@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # SSB
        ssb_enabled: bool,
        ssb_max_weight: float,
        ssb_warmup_start: int,
        ssb_warmup_end: int,
        # CSM shape supervision
        csm_shape_enabled: bool,
        csm_shape_max_weight: float,
        csm_shape_warmup_start: int,
        csm_shape_warmup_end: int,
        csm_shape_band_kernel: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        # ------------------------------------------------------------
        # Shape-Sensitive Boundary auxiliary loss
        # ------------------------------------------------------------
        self.ssb_enabled = bool(ssb_enabled)
        self.ssb_max_weight = float(ssb_max_weight)
        self.ssb_warmup_start = int(ssb_warmup_start)
        self.ssb_warmup_end = int(ssb_warmup_end)

        self.ssb_loss = SSBLoss()
        self.csm_shape_enabled = bool(
            csm_shape_enabled
        )

        self.csm_shape_max_weight = float(
            csm_shape_max_weight
        )

        self.csm_shape_warmup_start = int(
            csm_shape_warmup_start
        )

        self.csm_shape_warmup_end = int(
            csm_shape_warmup_end
        )

        self.csm_shape_band_kernel = int(
            csm_shape_band_kernel
        )
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # ------------------------------------------------------------
            # SSB
            # max_weight=0.1 is TEMPORARY for integration / smoke testing.
            # Final lambda_B will be selected in Phase 5-5.
            # ------------------------------------------------------------
            # "ssb_enabled": True,
            # "ssb_max_weight": 0.1,
            # "ssb_warmup_start": 1000,
            # "ssb_warmup_end": 5000,
            #short 
            "ssb_enabled": cfg.MODEL.SSB.ENABLED,
            "ssb_max_weight": cfg.MODEL.SSB.MAX_WEIGHT,
            "ssb_warmup_start": cfg.MODEL.SSB.WARMUP_START,
            "ssb_warmup_end": cfg.MODEL.SSB.WARMUP_END,
            "csm_shape_enabled":
                cfg.MODEL.CSM_SHAPE.ENABLED,

            "csm_shape_max_weight":
                cfg.MODEL.CSM_SHAPE.MAX_WEIGHT,

            "csm_shape_warmup_start":
                cfg.MODEL.CSM_SHAPE.WARMUP_START,

            "csm_shape_warmup_end":
                cfg.MODEL.CSM_SHAPE.WARMUP_END,

            "csm_shape_band_kernel":
                cfg.MODEL.CSM_SHAPE.BAND_KERNEL,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            # ------------------------------------------------------------
            # Existing PEM losses
            # ------------------------------------------------------------
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in weight_dict
                    losses.pop(k)

            # ------------------------------------------------------------
            # Shape-Sensitive Boundary auxiliary loss
            # ------------------------------------------------------------
            if self.ssb_enabled:

                required_ssb_keys = (
                    "ssb_phi",
                    "ssb_weight",
                    "ssb_valid_classes",
                    "ssb_valid_mask",
                )

                for key in required_ssb_keys:
                    if key not in batched_inputs[0]:
                        raise KeyError(
                            f"SSB is enabled but '{key}' "
                            "is missing from mapper output."
                        )

                # GT-side targets were generated on CPU workers.
                # Move only the tensors needed by the differentiable GPU loss.
                ssb_phi = torch.stack(
                    [
                        x["ssb_phi"]
                        for x in batched_inputs
                    ],
                    dim=0,
                ).to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                ssb_weight = torch.stack(
                    [
                        x["ssb_weight"]
                        for x in batched_inputs
                    ],
                    dim=0,
                ).to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                ssb_valid_classes = torch.stack(
                    [
                        x["ssb_valid_classes"]
                        for x in batched_inputs
                    ],
                    dim=0,
                ).to(
                    device=self.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )

                ssb_valid_mask = torch.stack(
                    [
                        x["ssb_valid_mask"]
                        for x in batched_inputs
                    ],
                    dim=0,
                ).to(
                    device=self.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )

                # ------------------------------------------------------------
                # Deep SSB supervision
                #
                # Use:
                #   - final decoder prediction
                #   - last two auxiliary decoder predictions
                #
                # Therefore SSB is applied to the last 3 prediction stages.
                # ------------------------------------------------------------

                aux_outputs = outputs.get("aux_outputs", None)

                if aux_outputs is None:
                    raise RuntimeError(
                        "Deep SSB requires outputs['aux_outputs'], "
                        "but it is missing. Check DEEP_SUPERVISION."
                    )

                if len(aux_outputs) < 1:
                    raise RuntimeError(
                        "Deep SSB requires at least one auxiliary output, "
                        f"but got {len(aux_outputs)}."
                    )

                ssb_stage_outputs = [
                    aux_outputs[-1],
                    {
                        "pred_logits": outputs["pred_logits"],
                        "pred_masks": outputs["pred_masks"],
                    },
                ]
                ssb_stage_losses = []

                for stage_output in ssb_stage_outputs:
                    stage_loss = self.ssb_loss(
                        pred_logits=stage_output["pred_logits"],
                        pred_masks=stage_output["pred_masks"],
                        phi=ssb_phi,
                        weight=ssb_weight,
                        valid_classes=ssb_valid_classes,
                        valid_mask=ssb_valid_mask,
                        # valid_mask=csm_valid_mask,
                    )

                    ssb_stage_losses.append(stage_loss)

                raw_loss_ssb = torch.stack(
                    ssb_stage_losses
                ).mean()
                
                iteration = int(
                    get_event_storage().iter
                )

                lambda_b = ssb_warmup_weight(
                    iteration=iteration,
                    max_weight=self.ssb_max_weight,
                    warmup_start=self.ssb_warmup_start,
                    warmup_end=self.ssb_warmup_end,
                )

                # Important:
                # Add AFTER the PEM criterion weight/pop loop.
                # losses["loss_ssb"] = (
                #     raw_loss_ssb
                #     * float(lambda_b)
                # )
                weighted_loss_ssb = (
                    raw_loss_ssb
                    * float(lambda_b)
                )
                # ---------------------------------------------------------
                # Periodic gradient diagnostics
                #
                # Compare the original PEM objective against raw SSB.
                # Diagnostic only: autograd.grad() does not populate p.grad.
                # ---------------------------------------------------------
                grad_diag_period = 200

                if (
                    iteration > 0
                    and (iteration+1) % grad_diag_period == 0
                ):
                    # At this point `losses` contains only the weighted
                    # original PEM criterion losses.
                    pem_loss_for_grad = sum(
                        losses.values()
                    )

                    grad_params = [
                        p
                        for p in self.parameters()
                        if p.requires_grad
                    ]

                    grad_pem = torch.autograd.grad(
                        pem_loss_for_grad,
                        grad_params,
                        retain_graph=True,
                        allow_unused=True,
                    )

                    grad_ssb_raw = torch.autograd.grad(
                        raw_loss_ssb,
                        grad_params,
                        retain_graph=True,
                        allow_unused=True,
                    )

                    pem_sq = raw_loss_ssb.new_zeros(())
                    ssb_sq = raw_loss_ssb.new_zeros(())
                    dot = raw_loss_ssb.new_zeros(())

                    for g_pem, g_ssb in zip(
                        grad_pem,
                        grad_ssb_raw,
                    ):
                        if g_pem is not None:
                            pem_sq = (
                                pem_sq
                                + g_pem.detach().float().square().sum()
                            )

                        if g_ssb is not None:
                            ssb_sq = (
                                ssb_sq
                                + g_ssb.detach().float().square().sum()
                            )

                        if (
                            g_pem is not None
                            and g_ssb is not None
                        ):
                            dot = (
                                dot
                                + (
                                    g_pem.detach().float()
                                    * g_ssb.detach().float()
                                ).sum()
                            )

                    grad_pem_norm = torch.sqrt(
                        pem_sq
                    )

                    grad_ssb_raw_norm = torch.sqrt(
                        ssb_sq
                    )

                    grad_ssb_weighted_norm = (
                        grad_ssb_raw_norm
                        * abs(float(lambda_b))
                    )

                    denom = (
                        grad_pem_norm
                        * grad_ssb_raw_norm
                    )

                    if float(denom.item()) > 0.0:
                        grad_cosine = (
                            dot / denom
                        )
                    else:
                        grad_cosine = raw_loss_ssb.new_tensor(
                            float("nan")
                        )

                    storage = get_event_storage()

                    storage.put_scalar(
                        "grad_pem_norm",
                        float(
                            grad_pem_norm.item()
                        ),
                    )

                    storage.put_scalar(
                        "grad_ssb_raw_norm",
                        float(
                            grad_ssb_raw_norm.item()
                        ),
                    )

                    storage.put_scalar(
                        "grad_ssb_weighted_norm",
                        float(
                            grad_ssb_weighted_norm.item()
                        ),
                    )

                    storage.put_scalar(
                        "grad_pem_ssb_cosine",
                        float(
                            grad_cosine.item()
                        ),
                    )
                losses["loss_ssb"] = weighted_loss_ssb

                storage = get_event_storage()

                storage.put_scalar(
                    "loss_ssb_raw",
                    float(
                        raw_loss_ssb.detach().item()
                    ),
                )

                storage.put_scalar(
                    "lambda_ssb",
                    float(lambda_b),
                )

                storage.put_scalar(
                    "loss_ssb_weighted",
                    float(
                        weighted_loss_ssb.detach().item()
                    ),
                )
            # ------------------------------------------------------------
            # CSM feature-level shape supervision
            # ------------------------------------------------------------
            if self.csm_shape_enabled:
                required_csm_keys = (
                    "csm_shape_target",
                    "csm_shape_weight",
                    "csm_boundary",
                    "ssb_valid_mask",
                )
                for key in required_csm_keys:
                    if key not in batched_inputs[0]:
                        raise KeyError(
                            "CSM shape supervision is enabled but "
                            f"'{key}' is missing from mapper output."
                        )
                if "csm_shape_pred" not in outputs:
                    raise KeyError(
                        "B7 CSM supervision is enabled but "
                        "outputs['csm_shape_pred'] is missing."
                    )

                if "csm_boundary_pred" not in outputs:
                    raise KeyError(
                        "B7 CSM supervision is enabled but "
                        "outputs['csm_boundary_pred'] is missing."
                    )
                csm_shape_target = torch.stack(
                    [
                        x["csm_shape_target"]
                        for x in batched_inputs
                    ],
                    dim=0,
                ).to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                csm_boundary = torch.stack(
                    [
                        x["csm_boundary"]
                        for x in batched_inputs
                    ],
                    dim=0,
                ).to(
                    device=self.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
                # csm_shape_weight = torch.stack(
                #     [
                #         x["csm_shape_weight"]
                #         for x in batched_inputs
                #     ],
                #     dim=0,
                # ).to(self.device)
                csm_shape_weight = torch.stack(
                    [
                        x["csm_shape_weight"]
                        for x in batched_inputs
                    ],
                    dim=0,
                ).to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                csm_valid_mask = torch.stack(
                    [
                        x["ssb_valid_mask"]
                        for x in batched_inputs
                    ],
                    dim=0,
                ).to(
                    device=self.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
                raw_loss_csm_shape = csm_shape_aux_loss(
                    pred_logits=outputs["csm_shape_pred"],
                    target=csm_shape_target,
                    boundary=csm_boundary,
                    valid_mask=csm_valid_mask,
                    pixel_weight=csm_shape_weight,
                    band_kernel=self.csm_shape_band_kernel,
                )
                raw_loss_csm_presence = (
                    csm_boundary_presence_loss(
                        pred_logits=outputs[
                            "csm_boundary_pred"
                        ],
                        boundary=csm_boundary,
                        valid_mask=csm_valid_mask,
                        band_kernel=self.csm_shape_band_kernel,
                    )
                )

                # --------------------------------------------------
                # B7 factorized CSM supervision
                #
                # presence branch:
                #     target = B
                #
                # shape branch:
                #     target = B*S
                #     B6 one-sided loss
                #
                # Keep total auxiliary scale comparable to B2-B6.
                # --------------------------------------------------
                raw_loss_csm_factorized = (
                    0.5 * raw_loss_csm_presence
                    + 0.5 * raw_loss_csm_shape
                )

                iteration = int(
                    get_event_storage().iter
                )

                lambda_c = ssb_warmup_weight(
                    iteration=iteration,
                    max_weight=self.csm_shape_max_weight,
                    warmup_start=self.csm_shape_warmup_start,
                    warmup_end=self.csm_shape_warmup_end,
                )

                losses["loss_csm_shape"] = (
                    raw_loss_csm_factorized
                    * float(lambda_c)
                )

                storage = get_event_storage()
                storage.put_scalar(
                    "loss_csm_shape_raw",
                    float(
                        raw_loss_csm_shape.detach()
                    ),
                )

                storage.put_scalar(
                    "loss_csm_presence_raw",
                    float(
                        raw_loss_csm_presence.detach()
                    ),
                )

                storage.put_scalar(
                    "loss_csm_factorized_raw",
                    float(
                        raw_loss_csm_factorized.detach()
                    ),
                )

                storage.put_scalar(
                    "lambda_csm_shape",
                    float(lambda_c),
                )
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
