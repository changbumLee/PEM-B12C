import math
from detectron2.layers import DeformConv, ModulatedDeformConv

import logging
from typing import Callable, Dict, Optional, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.layers import Conv2d, DeformConv, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY


def build_pixel_decoder(cfg, in_channels):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    #if not cfg.MODEL.SEM_SEG_HEAD.USE_BISENET:
    #    return None

    # name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    name = "PEM_Pixel_Decoder"
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, in_channels)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm='BN', groups=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False,
                              groups=groups)
        self.bn = get_norm(norm, out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class DeformLayer(nn.Module):
    """
    Deformable Convolution Layer module.

    This module implements deformable convolutional operation followed by upsampling.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        deconv_kernel (int): Kernel size for the transposed convolution operation (default: 4).
        deconv_stride (int): Stride for the transposed convolution operation (default: 2).
        deconv_pad (int): Padding for the transposed convolution operation (default: 1).
        deconv_out_pad (int): Output padding for the transposed convolution operation (default: 0).
        num_groups (int): Number of groups for convolution operation (default: 1).
        deform_num_groups (int): Number of deformable groups for deformable convolution (default: 1).
        dilation (int): Dilation factor for deformable convolution operation (default: 1).
        norm (str): Type of normalization layer to use (default: 'BN').
    """
    def __init__(self, in_planes, out_planes, deconv_kernel=4, deconv_stride=2, deconv_pad=1, deconv_out_pad=0,
                 num_groups=1, deform_num_groups=1, dilation=1, norm="BN"):
        super(DeformLayer, self).__init__()

        deform_conv_op = ModulatedDeformConv
        # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
        offset_channels = 27

        self.dcn_offset = nn.Conv2d(in_planes,
                                    offset_channels * deform_num_groups,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1 * dilation,
                                    dilation=dilation)
        self.dcn = deform_conv_op(in_planes,
                                  out_planes,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1 * dilation,
                                  bias=False,
                                  groups=num_groups,
                                  dilation=dilation,
                                  deformable_groups=deform_num_groups)
        for layer in [self.dcn]:
            weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.dcn_offset.weight, 0)
        nn.init.constant_(self.dcn_offset.bias, 0)

        self.dcn_bn = get_norm(norm, out_planes)

        self.up_sample = nn.ConvTranspose2d(in_channels=out_planes,
                                            out_channels=out_planes,
                                            kernel_size=deconv_kernel,
                                            stride=deconv_stride, padding=deconv_pad,
                                            output_padding=deconv_out_pad,
                                            bias=False)
        self._deconv_init()
        self.up_bn = get_norm(norm, out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x

        # DCN
        offset_mask = self.dcn_offset(out)
        offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()
        x = self.dcn(out.to(self.dcn.weight.dtype), offset.to(self.dcn.weight.dtype), mask.to(self.dcn.weight.dtype))
        x = self.dcn_bn(x)
        x = self.relu(x)

        # Upsampling
        x_up = self.up_sample(x)
        x_up = self.up_bn(x_up)
        x_up = self.relu(x_up)
        return x, x_up

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module for feature transformation.

    This module applies two convolutional layers with batch normalization and ReLU activation
    to transform input feature maps.

    Args:
        dim (int): The input feature dimensionality.
        factor (int): The reduction factor for the intermediate feature dimensionality.
    """
    def __init__(self, dim, factor, norm):
        super().__init__()

        self.mlp = nn.Sequential(nn.Conv2d(dim, dim // factor, 1, bias=False),
                                 get_norm(norm, dim // factor),
                                 nn.ReLU(),
                                 nn.Conv2d(dim // factor, dim, 1, bias=False),
                                 )

    def forward(self, x):
        return self.mlp(x)

class CSM(nn.Module):
    """
    Context-based Self-Modulation (CSM) module for enhancing feature representations.

    This module applies an MLP on global averaged pooled input feature to dynamically reweight
    its input in a residual fashion.

    Args:
        hidden_dim (int): The hidden dimensionality of the MLP.
        factor (int): The reduction factor for the intermediate feature dimensionality in the MLP.
    """
    def __init__(self, hidden_dim, factor, norm):
        super().__init__()
        self.mlp = nn.Sequential(MLP(hidden_dim, factor, norm),
                                 nn.Sigmoid())

    def forward(self, x):
        x_avg = x.mean(dim=[2, 3], keepdim=True)
        x_w = self.mlp(x_avg).sigmoid()
        return x + x * x_w

class B8BoundaryExpert(nn.Module):
    """
    B8 boundary-presence expert.

    Purpose:
        Learn explicit GT boundary presence B while keeping
        the auxiliary branch lightweight.

    Input:
        csm_4: [B, C, H, W]

    Output:
        boundary logits: [B, 1, H, W]
    """

    def __init__(self, in_channels: int):
        super().__init__()

        hidden = in_channels // 2
        num_groups = 8 if hidden % 8 == 0 else 1

        self.reduce = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups,
                hidden,
            ),
            nn.GELU(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups,
                hidden,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups,
                hidden,
            ),
        )

        self.act = nn.GELU()

        self.out = nn.Conv2d(
            hidden,
            1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        base = self.reduce(x)

        refined = self.refine(base)

        feat = self.act(
            base + refined
        )

        return self.out(feat)

class B8ShapeExpert(nn.Module):
    """
    B8 multi-scale shape-sensitive expert.

    The shape branch receives multiple local receptive fields
    while preserving the B7 B*S target and one-sided loss.

    Input:
        csm_4: [B, C, H, W]

    Output:
        shape logits: [B, 1, H, W]
    """

    def __init__(self, in_channels: int):
        super().__init__()

        hidden = in_channels // 2
        num_groups = 8 if hidden % 8 == 0 else 1

        self.reduce = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups,
                hidden,
            ),
            nn.GELU(),
        )

        # Lightweight multi-scale spatial experts.
        self.branch_d1 = nn.Sequential(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                dilation=1,
                groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups,
                hidden,
            ),
            nn.GELU(),
        )

        self.branch_d2 = nn.Sequential(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups,
                hidden,
            ),
            nn.GELU(),
        )

        self.branch_d4 = nn.Sequential(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=4,
                dilation=4,
                groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups,
                hidden,
            ),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                hidden * 3,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups,
                hidden,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups,
                hidden,
            ),
        )

        self.act = nn.GELU()

        self.out = nn.Conv2d(
            hidden,
            1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        base = self.reduce(x)

        feat_d1 = self.branch_d1(base)
        feat_d2 = self.branch_d2(base)
        feat_d4 = self.branch_d4(base)

        multi_scale = torch.cat(
            [
                feat_d1,
                feat_d2,
                feat_d4,
            ],
            dim=1,
        )

        refined = self.fuse(
            multi_scale
        )

        feat = self.act(
            base + refined
        )

        return self.out(feat)
class B12SCDA(nn.Module):
    """
    B12 Shape-Critical Dynamic Alignment.

    Align the final coarse stride-8 -> stride-4 feature
    before the final PEM fusion.

    The residual formulation guarantees exact B10 behavior
    when the predicted offset is zero:

        fine + coarse
        + gate * (aligned_coarse - coarse)

    Offset is zero-initialized.
    """

    def __init__(
        self,
        channels: int,
        max_offset: float = 1.0,
    ):
        super().__init__()

        hidden = channels // 2
        num_groups = (
            8 if hidden % 8 == 0 else 1
        )

        self.max_offset = float(max_offset)

        self.context = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups,
                hidden,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups,
                hidden,
            ),
            nn.GELU(),
        )

        # 2-D sampling displacement:
        # channel 0 = dx
        # channel 1 = dy
        self.offset_head = nn.Conv2d(
            hidden,
            2,
            kernel_size=1,
            bias=True,
        )

        # Spatial gate deciding how much aligned
        # coarse feature replaces the original one.
        self.gate_head = nn.Conv2d(
            hidden,
            1,
            kernel_size=1,
            bias=True,
        )

        # --------------------------------------------------
        # Important initialization:
        #
        # offset == 0
        # => aligned_coarse == coarse
        # => SCDA output == original B10 fusion
        # --------------------------------------------------
        nn.init.zeros_(
            self.offset_head.weight
        )
        nn.init.zeros_(
            self.offset_head.bias
        )

        # gate starts at 0.5.
        # Since offset starts at zero, this still gives
        # exactly the original B10 feature.
        nn.init.zeros_(
            self.gate_head.weight
        )
        nn.init.zeros_(
            self.gate_head.bias
        )

    def _make_base_grid(
        self,
        batch_size,
        height,
        width,
        device,
        dtype,
    ):
        ys = torch.linspace(
            -1.0,
            1.0,
            height,
            device=device,
            dtype=dtype,
        )

        xs = torch.linspace(
            -1.0,
            1.0,
            width,
            device=device,
            dtype=dtype,
        )

        grid_y, grid_x = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )

        grid = torch.stack(
            [
                grid_x,
                grid_y,
            ],
            dim=-1,
        )

        grid = grid.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
            -1,
        )

        return grid

    def forward(
        self,
        fine_feat,
        coarse_up,
    ):
        """
        fine_feat:
            stride-4 local CSM feature

        coarse_up:
            stride-8 feature already upsampled
            to stride-4 by the original PEM DeformLayer
        """

        if (
            fine_feat.shape[-2:]
            != coarse_up.shape[-2:]
        ):
            raise RuntimeError(
                "B12SCDA expects fine_feat and "
                "coarse_up to have identical spatial size, "
                f"got {fine_feat.shape[-2:]} and "
                f"{coarse_up.shape[-2:]}"
            )

        context = torch.cat(
            [
                fine_feat,
                coarse_up,
            ],
            dim=1,
        )

        context = self.context(
            context
        )

        # Offset in stride-4 feature-pixel units.
        offset = (
            torch.tanh(
                self.offset_head(
                    context
                )
            )
            * self.max_offset
        )

        gate = torch.sigmoid(
            self.gate_head(
                context
            )
        )

        batch_size, _, height, width = (
            coarse_up.shape
        )

        base_grid = self._make_base_grid(
            batch_size,
            height,
            width,
            coarse_up.device,
            coarse_up.dtype,
        )

        # Convert feature-pixel displacement
        # to normalized grid_sample coordinates.
        if width > 1:
            offset_x = (
                offset[:, 0]
                * (2.0 / (width - 1))
            )
        else:
            offset_x = (
                offset[:, 0] * 0.0
            )

        if height > 1:
            offset_y = (
                offset[:, 1]
                * (2.0 / (height - 1))
            )
        else:
            offset_y = (
                offset[:, 1] * 0.0
            )

        offset_grid = torch.stack(
            [
                offset_x,
                offset_y,
            ],
            dim=-1,
        )

        sample_grid = (
            base_grid
            + offset_grid
        )

        aligned_coarse = F.grid_sample(
            coarse_up,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # --------------------------------------------------
        # Residual replacement.
        #
        # offset == 0:
        # aligned_coarse == coarse_up
        #
        # therefore:
        # fused == fine_feat + coarse_up
        #
        # exactly reproducing B10.
        # --------------------------------------------------
        alignment_residual = (
            aligned_coarse
            - coarse_up
        )

        fused = (
            fine_feat
            + coarse_up
            + gate * alignment_residual
        )

        if self.training:
            storage = get_event_storage()

            storage.put_scalar(
                "scda_gate_mean",
                float(
                    gate.detach()
                    .mean()
                    .cpu()
                ),
            )

            storage.put_scalar(
                "scda_offset_abs_mean",
                float(
                    offset.detach()
                    .abs()
                    .mean()
                    .cpu()
                ),
            )

            storage.put_scalar(
                "scda_offset_abs_max",
                float(
                    offset.detach()
                    .abs()
                    .max()
                    .cpu()
                ),
            )

        return (
            fused,
            gate,
            offset,
        )
@SEM_SEG_HEADS_REGISTRY.register()
class PEM_Pixel_Decoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        csm_shape_enabled: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        self.norm = norm
        self.in_channels = [spec.channels for spec in input_shape.values()]

        self.in_projections = nn.ModuleList([ConvBNReLU(in_dim, conv_dim, 1, norm=norm, padding=0)
                                             for in_dim in self.in_channels])

        self.conv_avg = ConvBNReLU(self.in_channels[-1], conv_dim, 1, padding=0, norm=norm)

        self.csm = nn.ModuleList([CSM(conv_dim, factor=2, norm=norm) for _ in range(4)])
        self.dcn = nn.ModuleList([DeformLayer(conv_dim, conv_dim, norm=norm) for _ in range(3)])
        self.csm_shape_enabled = bool(csm_shape_enabled)
        # B12-SCDA:
        # final stride-8 -> stride-4 dynamic alignment.
        #
        # This is part of the main inference path,
        # unlike the B10 training-only auxiliary experts.
        self.scda = B12SCDA(
            conv_dim,
            max_offset=1.0,
        )

        self.out = ConvBNReLU(
            conv_dim,
            mask_dim,
            1,
            padding=0,
            norm=norm,
        )

        if self.csm_shape_enabled:

            # ------------------------------------------------------------
            # B8: task-specialized dual experts
            #
            # boundary expert:
            #     explicit boundary presence B
            #
            # shape expert:
            #     multi-scale local geometry for B*S
            #
            # Losses / targets / lambda remain identical to B7.
            # ------------------------------------------------------------
            self.csm_boundary_head = B8BoundaryExpert(
                conv_dim
            )

            self.csm_shape_head = B8ShapeExpert(
                conv_dim
            )

        else:
            self.csm_shape_head = None
            self.csm_boundary_head = None

        # if self.csm_shape_enabled:
        #     self.csm_shape_head = nn.Sequential(
        #         nn.Conv2d(
        #             conv_dim,
        #             conv_dim // 2,
        #             kernel_size=3,
        #             padding=1,
        #             bias=True,
        #         ),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(
        #             conv_dim // 2,
        #             1,
        #             kernel_size=1,
        #             bias=True,
        #         ),
        #     )

        #     # B7:
        #     # independent boundary-presence head.
        #     #
        #     # Shape semantics and boundary existence are intentionally
        #     # factorized into separate training-only heads.
        #     self.csm_boundary_head = nn.Sequential(
        #         nn.Conv2d(
        #             conv_dim,
        #             conv_dim // 2,
        #             kernel_size=3,
        #             padding=1,
        #             bias=True,
        #         ),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(
        #             conv_dim // 2,
        #             1,
        #             kernel_size=1,
        #             bias=True,
        #         ),
        #     )

        # else:
        #     self.csm_shape_head = None
        #     self.csm_boundary_head = None

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["csm_shape_enabled"] = cfg.MODEL.CSM_SHAPE.ENABLED
        return ret

    def forward_features(self, features):
        in_features = []
        for feature, projection in zip(features.values(), self.in_projections):
            in_features.append(projection(feature))

        conv_avg = self.conv_avg(features["res5"].mean(dim=[2, 3], keepdim=True))

        x_32 = self.csm[0](in_features[3]) + conv_avg
        x_32, x_up = self.dcn[0](x_32)

        x_16 = self.csm[1](in_features[2]) + x_up
        x_16, x_up = self.dcn[1](x_16)

        x_8 = self.csm[2](in_features[1]) + x_up
        x_8, x_up = self.dcn[2](x_8)

        csm_4 = self.csm[3](in_features[0])

        x_4, scda_gate, scda_offset = self.scda(
            csm_4,
            x_up,
        )
        csm_shape_pred = None
        csm_boundary_pred = None

        # if self.training and self.csm_shape_enabled:
        #     csm_shape_pred = self.csm_shape_head(
        #         csm_4
        #     )

        #     csm_boundary_pred = self.csm_boundary_head(
        #         csm_4
        #     )
        if self.training and self.csm_shape_enabled:
            csm_shape_pred = self.csm_shape_head(
                x_4
            )

            csm_boundary_pred = self.csm_boundary_head(
                x_4
            )

        return (
            self.out(x_4),
            None,
            [x_32, x_16, x_8],
            csm_shape_pred,
            csm_boundary_pred,
        )

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)
