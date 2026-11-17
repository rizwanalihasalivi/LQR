"""Run complete fresh campaigns with checkpoint and output preflight checks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('corrected_evaluation', ROOT / 'code/evaluate.py')
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)


def jobs():
    learned = [(kind, seed) for kind in ('corrected_lans', 'corrected_ppo', 'new_bc_lans', 'new_bc_ppo')
               for seed in (82, 83, 84)]
    learned += [(kind, seed) for kind in ('previous_lans', 'previous_ppo') for seed in (72, 73, 74)]
    return learned + [(kind, -1) for kind in ('matched_mpc', 'mpc', 'fast_lqr')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_checkpoints(root=ROOT):
    names = [os.path.relpath(evaluation.checkpoint(kind, seed), evaluation.ROOT)
             for kind, seed in jobs() if evaluation.checkpoint(kind, seed) is not None]
    return {name: root / name for name in names}


def verify_manifest(root=ROOT):
    manifest_path = root / 'results/checkpoint_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    declared = manifest['checkpoints']
    expected = expected_checkpoints(root)
    if set(declared) != set(expected):
        raise ValueError('Checkpoint manifest must cover exactly all 18 new and historical checkpoints.')
    for name, path in expected.items():
        if declared[name] != digest(path):
            raise ValueError(f'Frozen checkpoint hash mismatch: {name}')
    return digest(manifest_path)


def preflight(stage, root=ROOT):
    if stage not in evaluation.PLANS:
        raise ValueError(f'Unsupported stage: {stage}')
    protocol = root / 'results/protocol.json'
    evaluation.validate_protocol(stage, json.loads(protocol.read_text()))
    # Missing BC checkpoints or reference policies must fail before any work.
    for name, path in expected_checkpoints(root).items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint: {name}')
    out = root / 'results' / stage
    if (out / 'campaign.json').exists():
        raise FileExistsError(f'{stage} campaign already started; preserve it and use a fresh study.')
    for kind, seed in jobs():
        if ((out / f'{kind}_{seed}.csv').exists() or (out / f'{kind}_{seed}.run.json').exists()
                or (out / f'{kind}_{seed}.log').exists()
                or any((out / 'traces').glob(f'{kind}_{seed}_*.npz'))):
            raise FileExistsError(f'Existing or partial output: {kind}, seed {seed}')
    manifest_hash = verify_manifest(root) if stage == 'test' else None
    return dict(stage=stage, status='running', started=time.time(), jobs=jobs(),
                protocol_sha256=digest(protocol), checkpoint_manifest_sha256=manifest_hash)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=evaluation.PLANS, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    campaign = preflight(args.stage)
    out = ROOT / 'results' / args.stage
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'campaign.json'
    with marker.open('x') as f:
        json.dump(campaign, f, indent=2)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def run(pair):
        kind, seed = pair
        command = [sys.executable, str(ROOT / 'code/evaluate.py'), '--stage', args.stage,
                   '--kind', kind, '--seed', str(seed)]
        with (out / f'{kind}_{seed}.log').open('x') as f:
            subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        print('DONE', args.stage, kind, seed, flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, jobs()))
    if digest(ROOT / 'results/protocol.json') != campaign['protocol_sha256']:
        raise RuntimeError('Protocol changed during campaign.')
    if args.stage == 'test' and verify_manifest() != campaign['checkpoint_manifest_sha256']:
        raise RuntimeError('Checkpoint manifest changed during campaign.')
    campaign.update(status='completed', finished=time.time())
    marker.write_text(json.dumps(campaign, indent=2))


if __name__ == '__main__':
    main()
