#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config
from pem import add_maskformer2_config
from train_net import Trainer
from research.shape_ssb.geometry.contour import extract_class_contours
from research.shape_ssb.geometry.shape_score import load_curvature_references
from research.shape_ssb.geometry.stability import contour_shape_score

NUM_CLASSES=19
IGNORE_LABEL=255
RADII=(2,4,8)
MIN_CONTOUR_POINTS=16

def npy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def collect_center_scores(halo_gt, halo_px, crop_shape, refs):
    ch,cw=map(int,crop_shape); h=int(halo_px)
    y0,y1=h,h+ch; x0,x1=h,h+cw
    chunks=[]
    for c in range(NUM_CLASSES):
        contours=extract_class_contours(
            halo_gt,c,IGNORE_LABEL,MIN_CONTOUR_POINTS
        )
        for contour in contours:
            S=np.asarray(
                contour_shape_score(contour,refs,RADII)["S"],
                dtype=np.float32
            )
            xy=np.rint(contour.points).astype(np.int64)
            x,y=xy[:,0],xy[:,1]
            keep=(np.isfinite(S)&(x>=x0)&(x<x1)&(y>=y0)&(y<y1))
            if keep.any():
                chunks.append(S[keep])
    return np.concatenate(chunks) if chunks else np.empty(0,np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config-file",required=True)
    ap.add_argument("--curvature-reference",default="research/shape_ssb/geometry/curvature_reference_q95.json")
    ap.add_argument("--output",default="research/shape_ssb/geometry/shape_score_tau.json")
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--max-images",type=int,default=None)
    ap.add_argument("--print-period",type=int,default=100)
    a=ap.parse_args()

    cfg=get_cfg(); add_deeplab_config(cfg); add_maskformer2_config(cfg)
    cfg.merge_from_file(a.config_file)
    cfg.SEED=a.seed; cfg.SOLVER.IMS_PER_BATCH=1; cfg.DATALOADER.NUM_WORKERS=0
    cfg.freeze()

    refs=load_curvature_references(a.curvature_reference)
    ds=cfg.DATASETS.TRAIN[0]
    n_total=len(DatasetCatalog.get(ds))
    target=n_total if a.max_images is None else min(a.max_images,n_total)

    vals=[]; npts=0; done=0
    for batch in Trainer.build_train_loader(cfg):
        for x in batch:
            if done>=target: break
            gt=npy(x["sem_seg"])
            halo=npy(x["ssb_sem_seg_halo"])
            hp=int(x["ssb_halo_px"])
            s=collect_center_scores(halo,hp,gt.shape,refs)
            if s.size:
                vals.append(s.astype(np.float32,copy=False)); npts+=s.size
            done+=1
            if done%a.print_period==0 or done==target:
                print(f"[{done}/{target}] valid S={npts:,}")
        if done>=target: break

    if not vals: raise RuntimeError("No finite S collected")
    v=np.concatenate(vals).astype(np.float32,copy=False)
    q=np.quantile(v,[.05,.25,.50,.75,.95])
    out={
      "dataset":ds,"num_images":done,"num_valid_S_points":int(v.size),
      "seed":a.seed,"tau_rule":"Q50(valid S)","tau":float(q[2]),
      "quantiles":{"q05":float(q[0]),"q25":float(q[1]),"q50":float(q[2]),"q75":float(q[3]),"q95":float(q[4])},
      "radii":[2,4,8],"min_contour_points":16,"ignore_label":255,
      "curvature_reference":a.curvature_reference,
      "calibration_distribution":"PEM TRAIN augmentation; S computed with halo context; keep contour points inside final center crop"
    }
    op=Path(a.output); op.parent.mkdir(parents=True,exist_ok=True)
    op.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print("FINAL tau =",float(q[2]))
    print("saved to =",op)

if __name__=="__main__":
    main()
