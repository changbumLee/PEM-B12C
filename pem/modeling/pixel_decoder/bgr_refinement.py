"""Boundary-Guided Residual Refinement for PEM B12-c (Cityscapes 60k).

Main semantic/SCDA feature is never attenuated:
    refined = scda_output + gain * sigmoid(boundary_logit).detach() * residual

The B8 boundary head produces boundary_logit and is supervised by the existing
B12-c boundary-presence loss. No GT boundary information is used at inference.
"""
import torch
from torch import nn
import torch.nn.functional as F


class BoundaryGuidedResidualRefinement(nn.Module):
    def __init__(self, channels=128, hidden=64, gain=0.25):
        super().__init__()
        if channels < 1 or hidden < 8 or hidden % 8:
            raise ValueError('channels must be positive; hidden must be a multiple of 8')
        if not (0.0 < float(gain) <= 1.0):
            raise ValueError('gain must be within (0,1]')
        self.gain = float(gain)
        self.detail = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1,
                      groups=hidden, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        # Initial features are exactly the unchanged SCDA output.
        nn.init.zeros_(self.detail[-1].weight)
        nn.init.zeros_(self.detail[-1].bias)

    def forward(self, scda_feature, fine_feature, boundary_logits, activation=1.0):
        if not 0.0 <= float(activation) <= 1.0:
            raise ValueError('activation must be within [0,1]')
        if scda_feature.shape != fine_feature.shape:
            raise ValueError('SCDA and fine features must share [B,C,H,W] shape')
        if boundary_logits.shape != (scda_feature.shape[0], 1,
                                     scda_feature.shape[2], scda_feature.shape[3]):
            raise ValueError('boundary logits must be [B,1,H,W]')
        # Do not feed segmentation gradients backward into the supervised
        # B8 boundary expert through this gate; it retains its own GT loss.
        gate = boundary_logits.detach().sigmoid()
        residual = self.detail(torch.cat((fine_feature, scda_feature), dim=1))
        # FP32 is not forced; autocast AMP controls precision as usual.
        return scda_feature + (self.gain * float(activation) * gate) * residual
