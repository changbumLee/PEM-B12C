"""Conservative patcher for a pinned B12-c base. Raises on any anchor drift."""

def once(src, old, new, label):
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected exactly one anchor, found {n}')
    return src.replace(old, new, 1)


def patch_config(src):
    if 'cfg.MODEL.BGR = CN()' in src:
        raise RuntimeError('BGR config already patched')
    target = '    cfg.MODEL.CSM_SHAPE.SCDA_RELATIONAL_MAX_WEIGHT = 0.01'
    added = target + '''

    # Boundary-guided residual, uses existing B8 boundary expert for the gate.
    cfg.MODEL.BGR = CN()
    cfg.MODEL.BGR.ENABLED = False
    cfg.MODEL.BGR.HIDDEN_DIM = 64
    cfg.MODEL.BGR.GAIN = 0.25
    cfg.MODEL.BGR.WARMUP_START = 1000
    cfg.MODEL.BGR.WARMUP_END = 5000'''
    return once(src, target, added, 'config: B12-c relational weight')


def patch_pixel_decoder(src):
    src = once(src, 'from detectron2.modeling import SEM_SEG_HEADS_REGISTRY',
               'from detectron2.modeling import SEM_SEG_HEADS_REGISTRY\nfrom .bgr_refinement import BoundaryGuidedResidualRefinement',
               'pixel decoder import')
    src = once(src, '        csm_shape_enabled: bool = False,',
               '''        csm_shape_enabled: bool = False,
        bgr_enabled: bool = False,
        bgr_hidden_dim: int = 64,
        bgr_gain: float = 0.25,
        bgr_warmup_start: int = 1000,
        bgr_warmup_end: int = 5000,''', 'constructor signature')
    src = once(src, '''        else:
            self.csm_shape_head = None
            self.csm_boundary_head = None''',
               '''        else:
            self.csm_shape_head = None
            self.csm_boundary_head = None

        # BGR uses the existing supervised B8 boundary prediction at inference.
        if bgr_enabled and not self.csm_shape_enabled:
            raise ValueError('MODEL.BGR.ENABLED requires MODEL.CSM_SHAPE.ENABLED=True')
        if bgr_warmup_end <= bgr_warmup_start:
            raise ValueError('BGR warmup end must be greater than start')
        self.bgr_warmup_start = int(bgr_warmup_start)
        self.bgr_warmup_end = int(bgr_warmup_end)
        self.bgr_refiner = (
            BoundaryGuidedResidualRefinement(conv_dim, bgr_hidden_dim, bgr_gain)
            if bgr_enabled else None
        )''', 'B8 heads and refiner')
    src = once(src, '''        ret["csm_shape_enabled"] = cfg.MODEL.CSM_SHAPE.ENABLED
        return ret''',
               '''        ret["csm_shape_enabled"] = cfg.MODEL.CSM_SHAPE.ENABLED
        ret["bgr_enabled"] = cfg.MODEL.BGR.ENABLED
        ret["bgr_hidden_dim"] = cfg.MODEL.BGR.HIDDEN_DIM
        ret["bgr_gain"] = cfg.MODEL.BGR.GAIN
        ret["bgr_warmup_start"] = cfg.MODEL.BGR.WARMUP_START
        ret["bgr_warmup_end"] = cfg.MODEL.BGR.WARMUP_END
        return ret''', 'from_config')
    src = once(src, '''        csm_shape_pred = None
        csm_boundary_pred = None
''', '''        csm_shape_pred = None
        csm_boundary_pred = None
        if self.bgr_refiner is not None:
            # Predict boundary on unmodified SCDA features; reuse its GT
            # supervision during training and its prediction at inference.
            csm_boundary_pred = self.csm_boundary_head(x_4)
            bgr_activation = 1.0
            if self.training:
                bgr_iter = get_event_storage().iter
                bgr_activation = max(0.0, min(1.0,
                    (bgr_iter - self.bgr_warmup_start) /
                    (self.bgr_warmup_end - self.bgr_warmup_start)))
            x_4 = self.bgr_refiner(x_4, csm_4, csm_boundary_pred,
                                   activation=bgr_activation)
''', 'forward BGR insertion')
    src = once(src, '''            csm_boundary_pred = self.csm_boundary_head(
                x_4
            )''', '''            if csm_boundary_pred is None:
                csm_boundary_pred = self.csm_boundary_head(
                    x_4
                )''', 'reuse boundary supervision')
    return src
