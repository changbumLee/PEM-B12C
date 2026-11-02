import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patch_b12c import patch_config, patch_pixel_decoder


PIXEL = '''from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
class PEM_Pixel_Decoder:
    def __init__(self, 
        csm_shape_enabled: bool = False,
    ):
        if self.csm_shape_enabled:
            self.csm_boundary_head = 1
        else:
            self.csm_shape_head = None
            self.csm_boundary_head = None
    def from_config(self, cfg, ret):
        ret["csm_shape_enabled"] = cfg.MODEL.CSM_SHAPE.ENABLED
        return ret
    def forward_features(self):
        csm_shape_pred = None
        csm_boundary_pred = None
        if self.training and self.csm_shape_enabled:
            csm_shape_pred = self.csm_shape_head(
                x_4
            )
            csm_boundary_pred = self.csm_boundary_head(
                x_4
            )
'''


class PatcherTests(unittest.TestCase):
    def test_patch_residual_exact_once(self):
        changed = patch_pixel_decoder(PIXEL)
        self.assertEqual(changed.count('x_4 = self.bgr_refiner(x_4, csm_4, csm_boundary_pred,'), 1)
        self.assertEqual(changed.count('csm_boundary_pred = self.csm_boundary_head(x_4)'), 1)
        self.assertIn('if csm_boundary_pred is None:', changed)
        self.assertIn('"bgr_enabled"', changed)
        self.assertIn('BoundaryGuidedResidualRefinement', changed)

    def test_patch_no_silent_reapply(self):
        changed = patch_pixel_decoder(PIXEL)
        with self.assertRaises(RuntimeError):
            patch_pixel_decoder(changed)

    def test_patch_config(self):
        src = '    cfg.MODEL.CSM_SHAPE.SCDA_RELATIONAL_MAX_WEIGHT = 0.01'
        dst = patch_config(src)
        self.assertIn('cfg.MODEL.BGR.GAIN = 0.25', dst)
        self.assertIn('cfg.MODEL.BGR.ENABLED = False', dst)
        with self.assertRaises(RuntimeError):
            patch_config(dst)

    def test_invalid_base_fails(self):
        with self.assertRaises(RuntimeError):
            patch_pixel_decoder('not baseline')


if __name__ == '__main__':
    unittest.main()
