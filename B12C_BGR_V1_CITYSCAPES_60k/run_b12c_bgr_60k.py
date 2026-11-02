#!/usr/bin/env python3
"""Cityscapes-only 60k B12-c + Boundary-Guided Residual Refinement runner."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
REPO_DEFAULT = PACKAGE.parent
BASE_CONFIG_REL = ('output/kroc_curve/'
                   'b12c_relation_l001_seed2026_20261005_162211_20261005_162212/config.yaml')
WEIGHTS = 'detectron2://ImageNetPretrained/torchvision/R-50.pkl'


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def preflight(root, base_cfg):
    record = root / 'B12C_BGR_V1_INSTALLED.json'
    if not record.is_file():
        raise RuntimeError('BGR is not installed: run install_bgr_b12c.py --install')
    data = json.loads(record.read_text())
    for relative, expected in data['installed_sha256'].items():
        file = root / relative
        if not file.is_file() or sha(file) != expected:
            raise RuntimeError('Installed BGR source changed: ' + str(file))
    if not base_cfg.is_file():
        raise RuntimeError('Reference B12-c config not found: ' + str(base_cfg))
    try:
        import yaml
        settings = yaml.safe_load(base_cfg.read_text())
    except Exception as exc:
        raise RuntimeError('Cannot parse the saved B12-c YAML config: ' + str(exc)) from exc
    model = settings.get('MODEL', {})
    csm = model.get('CSM_SHAPE', {})
    tests = {
        'MODEL.CSM_SHAPE.ENABLED': csm.get('ENABLED') is True,
        'MODEL.CSM_SHAPE.SCDA_SEMANTIC_PRESERVE_ENABLED': csm.get('SCDA_SEMANTIC_PRESERVE_ENABLED') is True,
        'MODEL.CSM_SHAPE.SCDA_RELATIONAL_ENABLED': csm.get('SCDA_RELATIONAL_ENABLED') is True,
        'MODEL.SSB.ENABLED=False': model.get('SSB', {}).get('ENABLED') is False,
        'SOLVER.MAX_ITER=60000': settings.get('SOLVER', {}).get('MAX_ITER') == 60000,
        'SOLVER.IMS_PER_BATCH=8': settings.get('SOLVER', {}).get('IMS_PER_BATCH') == 8,
        'MODEL.SEM_SEG_HEAD.NUM_CLASSES=19': model.get('SEM_SEG_HEAD', {}).get('NUM_CLASSES') == 19,
        'Cityscapes training': 'cityscapes_fine_sem_seg_train' in settings.get('DATASETS', {}).get('TRAIN', []),
        'Cityscapes val': 'cityscapes_fine_sem_seg_val' in settings.get('DATASETS', {}).get('TEST', []),
    }
    bad = [k for k, v in tests.items() if not v]
    if bad:
        raise RuntimeError('Reference config not full B12-c Cityscapes 60k: ' + ', '.join(bad))
    return data


def make_command(root, config, args, output):
    b = 8 if args.smoke else args.batch_size
    cmd = [sys.executable, str(root / 'train_net.py'), '--num-gpus', '1',
           '--config-file', str(config),
           'MODEL.WEIGHTS', WEIGHTS,
           'MODEL.BGR.ENABLED', 'True',
           'MODEL.BGR.HIDDEN_DIM', str(args.hidden),
           'MODEL.BGR.GAIN', str(args.gain),
           'SEED', str(args.seed),
           'SOLVER.MAX_ITER', str(3 if args.smoke else 60000),
           'SOLVER.IMS_PER_BATCH', str(b),
           'DATALOADER.NUM_WORKERS', str(0 if args.smoke else args.workers),
           'TEST.EVAL_PERIOD', str(0 if args.smoke else 5000),
           'SOLVER.CHECKPOINT_PERIOD', str(3 if args.smoke else 10000),
           'OUTPUT_DIR', str(output)]
    if args.smoke:
        cmd += ['DATASETS.TEST', '()',
                'MODEL.CSM_SHAPE.WARMUP_START', '0',
                'MODEL.CSM_SHAPE.WARMUP_END', '1',
                'MODEL.BGR.WARMUP_START', '0',
                'MODEL.BGR.WARMUP_END', '1']
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=REPO_DEFAULT)
    ap.add_argument('--config-file', type=Path, default=None)
    ap.add_argument('--smoke', action='store_true', help='3 iterations with real batch=8 and backward')
    ap.add_argument('--print-only', action='store_true')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--gain', type=float, default=0.25)
    args = ap.parse_args()
    root = args.repo.expanduser().resolve()
    cfg = (args.config_file or (root / BASE_CONFIG_REL)).expanduser().resolve()
    if args.batch_size < 2 or args.workers < 0:
        raise RuntimeError('Expected global batch>=2, workers>=0')
    if args.hidden < 8 or args.hidden % 8 or not (0 < args.gain <= 1):
        raise RuntimeError('Expected hidden multiple of 8 and 0<gain<=1')
    rec = preflight(root, cfg)
    tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = root / 'output/kroc_curve/bgr_b12c_v1' / (
        ('smoke3' if args.smoke else 'train60k') + '_bgr_seed' + str(args.seed)
        + '_gain' + str(args.gain).replace('.', 'p') + '_' + tag)
    cmd = make_command(root, cfg, args, out)
    print('[PREFLIGHT PASS] pinned B12-c + BGR installed; no QMS/Support in tracked source', flush=True)
    print('[TRAIN]' if not args.smoke else '[SMOKE batch=8]')
    print(shlex.join(cmd), flush=True)
    if args.print_only:
        return
    records = root / 'output/kroc_curve/bgr_b12c_v1/records'
    records.mkdir(parents=True, exist_ok=True)
    record_file = records / (tag + ('_smoke' if args.smoke else '_train') + '.json')
    info = {'base_commit': rec['base_commit'], 'method': 'BGR',
            'data': 'Cityscapes', 'iteration_count': 3 if args.smoke else 60000,
            'batch': 8 if args.smoke else args.batch_size, 'workers': 0 if args.smoke else args.workers,
            'seed': args.seed, 'gain': args.gain, 'hidden': args.hidden,
            'initial_weights': WEIGHTS, 'command': cmd, 'output_prefix': str(out), 'status': 'started'}
    record_file.write_text(json.dumps(info, indent=2) + '\n')
    env = os.environ.copy()
    # Previous QMS benchmark uses environment-driven switches; never inherit them.
    env = {k:v for k,v in env.items() if not k.startswith('QMS_')}
    p = subprocess.run(cmd, cwd=root, env=env)
    dirs = sorted(out.parent.glob(out.name + '_*'), key=lambda f: f.stat().st_mtime)
    actual = dirs[-1] if dirs else None
    info.update(returncode=p.returncode, output_dir=str(actual) if actual else None,
                status='completed' if p.returncode == 0 else 'failed')
    record_file.write_text(json.dumps(info, indent=2) + '\n')
    if p.returncode:
        raise RuntimeError('Training failed. Check ' + str(record_file))
    if not actual or not (actual / 'model_final.pth').exists():
        raise RuntimeError('Training returned without model_final.pth. Check ' + str(record_file))
    if args.smoke:
        print('[SMOKE PASS] batch8 CUDA forward/backward and checkpoint completed', flush=True)
    print('[SAVED]', actual, flush=True)
    print('[RECORD]', record_file, flush=True)

if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError) as e:
        print('[ERROR]', e, file=sys.stderr)
        raise SystemExit(1)
