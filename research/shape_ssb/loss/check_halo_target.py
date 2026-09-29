#!/usr/bin/env python3
import argparse, numpy as np, torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from pem import add_maskformer2_config
from train_net import Trainer
from research.shape_ssb.geometry.shape_score import load_curvature_references
from research.shape_ssb.loss.ssb_halo_target import build_halo_ssb_target

def npy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--curvature-reference", default="research/shape_ssb/geometry/curvature_reference_q95.json")
    ap.add_argument("--tau", type=float, default=0.25)
    a=ap.parse_args()

    cfg=get_cfg(); add_deeplab_config(cfg); add_maskformer2_config(cfg)
    cfg.merge_from_file(a.config_file)
    cfg.SOLVER.IMS_PER_BATCH=1; cfg.DATALOADER.NUM_WORKERS=0; cfg.freeze()

    x=next(iter(Trainer.build_train_loader(cfg)))[0]
    gt=npy(x["sem_seg"]).astype(np.int32,copy=False)
    halo=npy(x["ssb_sem_seg_halo"]).astype(np.int32,copy=False)
    hp=int(x["ssb_halo_px"])
    h,w=gt.shape

    center=halo[hp:hp+h,hp:hp+w]
    mismatch=int((center!=gt).sum())

    print("GT:",gt.shape,"halo:",halo.shape,"halo_px:",hp,"center mismatch:",mismatch)
    assert mismatch==0

    refs=load_curvature_references(a.curvature_reference)
    t=build_halo_ssb_target(halo,hp,(h,w),refs,tau=a.tau)

    print("phi:",t.phi.shape,"valid classes:",np.flatnonzero(t.valid_classes).tolist())
    assert t.phi.shape==(19,h,w)
    assert t.valid_mask.shape==(h,w)
    assert np.isfinite(t.phi).all()
    assert float(t.distance.max())<=32.0+1e-6

    b=t.boundary & t.valid_mask[None]
    if b.any():
        assert float(t.distance[b].max())<=1e-6
        assert float(t.phi[b].max())<=1e-6

    print("HALO SSB TARGET SANITY PASSED")

if __name__=="__main__":
    main()
