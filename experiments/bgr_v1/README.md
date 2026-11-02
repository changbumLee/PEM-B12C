# BGR v1 — Cityscapes 60k (2026-11-02)

This branch begins at B12-c pinned commit `ddce1e5a8bf5d162ce4241c3c166114de1b4b6e3`.
It contains the applied BGR source changes and the original BGR installer,
runner, and tests in `B12C_BGR_V1_CITYSCAPES_60k/`.

## Final validation results (Cityscapes val 500)

| Metric | B12-c 60k | BGR v1 60k | Difference (BGR - B12-c) |
|---|---:|---:|---:|
| mIoU (%) | 78.1869 | **78.5147** | +0.3278 pp |
| iIoU (%) | 60.9162 | **61.1501** | +0.2339 pp |
| BF1 (%) | 60.6362 | **60.7111** | +0.0749 pp |
| Boundary IoU (%) | 71.8535 | **71.9756** | +0.1221 pp |
| SC-BF1 (%) | **48.8827** | 48.7506 | -0.1321 pp |
| SCLE (px; lower better) | 5.9081 | **5.825158** | -0.082942 px |

BGR gate: supervised B8 boundary prediction (detached). BGR residual is
zero-initialized and warmed up from 1k to 5k iterations. No Query Mask
Stability (QMS) nor Alignment Reliability (ALR) changes are included.

The evaluation output files from the user's local run are stored under
`experiments/bgr_v1/raw_evaluation/`. The raw `metrics.json` training log
and `model_final.pth` are intentionally **not in GitHub** because the full
checkpoint is approximately 412 MB. They are preserved in a local archive
outside the Git worktree by `save_bgr_v1.py`.

## Evaluating a locally available checkpoint

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
BGR_DIR="$PWD/output/kroc_curve/bgr_b12c_v1/train60k_bgr_seed2026_gain0p25_20261102_130446_533462_20261102_130448"
python train_net.py --num-gpus 1 --eval-only --config-file "$BGR_DIR/config.yaml" MODEL.WEIGHTS "$BGR_DIR/model_final.pth"
python research/shape_ssb/evaluation/evaluate_ssb.py --config-file "$BGR_DIR/config.yaml" --checkpoint "$BGR_DIR/model_final.pth" --output-dir "$BGR_DIR/evaluation/boundary" --bf-tolerance 2 --boundary-iou-ratio .02 --l-min 16
python research/shape_ssb/evaluation/evaluate_shape_baseline.py --config-file "$BGR_DIR/config.yaml" --checkpoint "$BGR_DIR/model_final.pth" --curvature-reference research/shape_ssb/geometry/curvature_reference_q95.json --output "$BGR_DIR/evaluation/shape_metrics.json" --print-period 50
```

If you want to move the checkpoint to GitHub, use Git LFS or a release/remote
artifact store with appropriate file size and repository quotas, not ordinary Git.
