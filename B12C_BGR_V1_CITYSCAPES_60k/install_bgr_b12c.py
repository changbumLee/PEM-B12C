#!/usr/bin/env python3
"""Back up only whitelisted B12-c sources, restore pinned Git blobs and apply BGR.

Does not reset Git HEAD/index or delete model checkpoints.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = 'ddce1e5a8bf5d162ce4241c3c166114de1b4b6e3'
CORE = (
    'pem/config.py',
    'pem/maskformer_model.py',
    'pem/data/dataset_mappers/mask_former_semantic_dataset_mapper.py',
    'pem/modeling/pixel_decoder/pem_pixel_decoder.py',
    'pem/modeling/meta_arch/mask_former_head.py',
    'pem/modeling/transformer_decoder/pem_transformer_decoder.py',
)
ADDON = 'pem/modeling/pixel_decoder/bgr_refinement.py'
RECORD = 'B12C_BGR_V1_INSTALLED.json'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(repo, *args):
    p = subprocess.run(['git', '-C', str(repo), *args], capture_output=True)
    if p.returncode:
        raise RuntimeError(f'Git failed: {" ".join(args)}: {p.stderr.decode(errors="replace").strip()}')
    return p.stdout


def read_pinned(repo):
    git(repo, 'cat-file', '-e', BASE + '^{commit}')
    blobs = {p: git(repo, 'show', f'{BASE}:{p}') for p in CORE}
    for rel, b in blobs.items():
        if not b:
            raise RuntimeError('Pinned Git blob is empty: ' + rel)
    return blobs


def current_hashes(root, files):
    return {p: sha((root / p).read_bytes()) if (root / p).is_file() else None for p in files}


def apply_bytes(blobs):
    from patch_b12c import patch_config, patch_pixel_decoder
    desired = dict(blobs)
    desired['pem/config.py'] = patch_config(blobs['pem/config.py'].decode('utf-8')).encode('utf-8')
    desired['pem/modeling/pixel_decoder/pem_pixel_decoder.py'] = patch_pixel_decoder(
        blobs['pem/modeling/pixel_decoder/pem_pixel_decoder.py'].decode('utf-8')).encode('utf-8')
    desired[ADDON] = (HERE / 'payload' / ADDON).read_bytes()
    # Ensure no stale QMS or Support-v1 reference survives in the tracked core.
    for p in CORE:
        if b'from ..scda_support' in desired[p] or b'qms_token_selector' in desired[p]:
            raise RuntimeError('Unexpected Support/QMS reference in pinned core: ' + p)
    return desired


def running_repo_training(root):
    """Best-effort guard. User must still ensure external processes finished."""
    try:
        proc = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    found = []
    mypid = os.getpid()
    for line in proc.stdout.splitlines():
        pieces = line.strip().split(maxsplit=1)
        if len(pieces) < 2 or not pieces[0].isdigit():
            continue
        pid, cmd = int(pieces[0]), pieces[1]
        if pid == mypid or 'ps -eo' in cmd:
            continue
        if ('train_net.py' in cmd and (str(root) in cmd or 'kroc_curve' in cmd)):
            found.append((pid, cmd[:180]))
    return found


def tests():
    r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE / 'tests'),
                        '-p', 'test_*.py', '-v'], cwd=HERE)
    if r.returncode:
        raise RuntimeError('CPU tests failed')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--install', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    root = args.repo.expanduser().resolve()
    if not (root / 'train_net.py').is_file():
        raise RuntimeError('Not a PEM repo: train_net.py missing')
    if args.self_test:
        tests()
    blobs = read_pinned(root)
    desired = apply_bytes(blobs)
    before = current_hashes(root, desired)
    after = {p: sha(data) for p, data in desired.items()}
    changed = [p for p in desired if before[p] != after[p]]
    print('Pinned B12-c:', BASE)
    print('Current HEAD (unchanged):', git(root, 'rev-parse', '--short', 'HEAD').decode().strip())
    print('Restore tracked core:', ', '.join(CORE))
    print('BGR overlay:', ADDON, '+ config / pixel decoder')
    print('Files differing:', ', '.join(changed) if changed else '(none)')
    if args.check:
        print('[CHECK PASS] pinned Git blobs and BGR patch anchors validated. No files changed.')
        return
    live = running_repo_training(root)
    if live:
        raise RuntimeError('PEM training appears active. Finish it before replacing source: ' + str(live))
    if not changed and (root / RECORD).is_file():
        print('[ALREADY INSTALLED] no source changes')
        return
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = root.parent / 'PEM_source_backups' / ('bgr_v1_' + timestamp)
    backup.mkdir(parents=True, exist_ok=False)
    # Back up ALL targets, not just those that differ. No destructive git reset.
    manifest = {}
    for rel in desired:
        src = root / rel
        manifest[rel] = {'existed': src.is_file(), 'sha256': before[rel]}
        if src.exists():
            dest = backup / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    (backup / 'backup_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Installed package uses original Git contents for untouched files.
    try:
        for rel, data in desired.items():
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + '.bgr_temp')
            tmp.write_bytes(data)
            os.replace(tmp, dst)
        for rel, h in after.items():
            if sha((root / rel).read_bytes()) != h:
                raise RuntimeError('Post-install hash mismatch: ' + rel)
    except Exception:
        for rel, state in manifest.items():
            dst = root / rel
            if state['existed']:
                shutil.copy2(backup / rel, dst)
            elif dst.exists():
                dst.unlink()
        raise
    record = {'base_commit': BASE, 'backup': str(backup), 'installed_sha256': after,
              'recipe': 'B12-c + boundary-guided residual refinement (no Support/QMS)',
              'created_at': timestamp}
    (root / RECORD).write_text(json.dumps(record, indent=2) + '\n')
    print('[INSTALLED] Pinned B12-c core + Boundary-Guided Residual Refinement v1')
    print('Backup:', backup)
    print('Record:', root / RECORD)
    print('HEAD/index, checkpoints and existing outputs unchanged.')

if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, UnicodeError) as e:
        print('[ERROR]', e, file=sys.stderr)
        raise SystemExit(1)
