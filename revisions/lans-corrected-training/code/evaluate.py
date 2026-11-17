"""Fresh scalar evaluations; historical policies keep their original observations.

Examples:
  python code/evaluate.py --stage development --kind corrected_lans --seed 82
  python code/evaluate.py --stage test --kind previous_lans --seed 72
Each kind/seed/stage is write-once. Preserve interrupted runs and use a fresh
study directory rather than replacing traces or silently reusing old scores.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / 'lans-tracking-study'
sys.path.insert(1, str(LEGACY / 'code'))
from features import NetworkEnv as LegacyEnv
from dynamic_env import design
from mpc import MPC
from matched_mpc import MatchedMPC
from precise_features import PrecisionEnv

# Reuse the historical checkpoint loader without importing this file as itself.
_spec = importlib.util.spec_from_file_location('tracking_evaluate', LEGACY / 'code/evaluate.py')
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

KINDS = ('corrected_lans', 'corrected_ppo', 'new_bc_lans', 'new_bc_ppo',
         'previous_lans', 'previous_ppo', 'matched_mpc', 'mpc', 'fast_lqr')
PLANS = {
    'development': (range(31000, 31003), ((.94, 1.03), (1.06, .97))),
    'test': (range(33000, 33005), ((.92, 1.06), (1.08, .94))),
}


def cases(stage):
    seeds, scales = PLANS[stage]
    for amplitude, time_scale in scales:
        for path in ('dlc', 'step', 'sine', 'slalom'):
            for delay in ('none', 'mixed', 'fixed200'):
                for seed in seeds:
                    yield dict(path=path, delay=delay, mass=1650., amplitude=amplitude,
                               time_scale=time_scale, physical_seed=seed)
        for mass in (1100., 2200.):
            for seed in seeds:
                yield dict(path='dlc', delay='none', mass=mass, amplitude=amplitude,
                           time_scale=time_scale, physical_seed=seed)


def validate_protocol(stage, spec):
    seeds, scales = PLANS[stage]
    if spec[stage + '_seeds'] != list(seeds) or spec[stage + '_scales'] != [list(x) for x in scales]:
        raise ValueError('Evaluator cases do not match the frozen protocol.')


def checkpoint(kind, seed):
    if kind not in KINDS:
        raise ValueError(f'Unsupported controller: {kind}')
    if kind in ('previous_lans', 'previous_ppo'):
        family = kind.removeprefix('previous_')
        return LEGACY / 'models' / f'limited_{family}_{seed}' / 'policy.zip'
    if kind.startswith('corrected_') or kind.startswith('new_bc_'):
        family = kind.rsplit('_', 1)[1]
        name = 'bc.zip' if kind.startswith('new_bc_') else 'policy.zip'
        return ROOT / 'models' / f'corrected_{family}_{seed}' / name
    return None


def load(kind, seed):
    path = checkpoint(kind, seed)
    if kind.startswith('previous_'):
        return _legacy.load('limited_' + kind.removeprefix('previous_'), seed)
    return PPO.load(path, device='cpu') if path is not None else None


def environment(kind, case):
    kwargs = dict(path=case['path'], mass=case['mass'], delay_mode=case['delay'],
                  amplitude=case['amplitude'], time_scale=case['time_scale'])
    if kind.startswith('corrected_') or kind.startswith('new_bc_'):
        return PrecisionEnv(kind=kind.rsplit('_', 1)[1], **kwargs)
    family = 'lans' if kind == 'previous_lans' else ('ppo' if kind == 'previous_ppo'
             else ('lqr' if kind == 'fast_lqr' else 'mpc'))
    env = LegacyEnv(kind=family, **kwargs)
    if kind == 'fast_lqr':
        env.K = design()
    return env


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(stage, kind, seed):
    # Direct calls get the same validation as argparse; typos must not run LQR.
    policy_path = checkpoint(kind, seed)
    if stage not in PLANS:
        raise ValueError(f'Unsupported stage: {stage}')
    protocol = ROOT / 'results/protocol.json'
    if stage == 'test' and not protocol.is_file():
        raise FileNotFoundError('Freeze results/protocol.json before final testing.')
    if protocol.is_file():
        validate_protocol(stage, json.loads(protocol.read_text()))
    if stage == 'test':
        # Direct single-controller runs obey the same freeze as the campaign runner.
        from run_eval import verify_manifest
        verify_manifest(ROOT)
    out = ROOT / 'results' / stage
    trace_dir = out / 'traces'
    csv_path = out / f'{kind}_{seed}.csv'
    marker = out / f'{kind}_{seed}.run.json'
    if csv_path.exists() or marker.exists() or any(trace_dir.glob(f'{kind}_{seed}_*.npz')):
        raise FileExistsError(f'{stage} results already exist for {kind}, seed {seed}; preserve this run.')
    policy = load(kind, seed)
    out.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(exist_ok=True)
    metadata = dict(stage=stage, kind=kind, seed=seed, status='running', started=time.time(),
                    protocol_sha256=file_hash(protocol) if protocol.is_file() else None,
                    checkpoint_sha256=file_hash(policy_path) if policy_path else None,
                    cases=list(cases(stage)), trace_columns=[
                        'time', 'error_next', 'heading_error_next', 'command', 'applied',
                        'residual', 'gain', 'delay_ms', 'queue', 'ey_before',
                        'ey_dot_before', 'heading_before', 'heading_dot_before'])
    with marker.open('x') as f:
        json.dump(metadata, f, indent=2)
    rows = []
    for case in metadata['cases']:
        env = environment(kind, case)
        obs, _ = env.reset(seed=case['physical_seed'])
        initial = float(env.x[0])
        mpc = MatchedMPC() if kind == 'matched_mpc' else (MPC() if kind == 'mpc' else None)
        trace = []
        for tick in range(1000):
            before = env.x.copy()
            action = policy.predict(obs, deterministic=True)[0] if policy else (
                [mpc.action(env) / .5] if mpc else [0.])
            obs, _, terminated, truncated, info = env.step(action)
            trace.append([tick * .01, info['error'], info['heading_error'], info['u'],
                          info['applied'], info['residual'], info['gain'],
                          info['delay_ms'], info['queue'], *before])
            if terminated or truncated:
                break
        arr = np.asarray(trace)
        count = len(arr)
        sse = initial ** 2 + np.square(arr[:, 1]).sum()
        rows.append(dict(kind=kind, train_seed=seed, **case, steps=count,
                         completed=not terminated and truncated,
                         rmse_penalized=float(np.sqrt((sse + (1000 - count) *
                             max(2.5, abs(arr[-1, 1])) ** 2) / 1001)),
                         rmse_observed=float(np.sqrt(sse / (count + 1))),
                         optimizer_failures=mpc.failures if mpc else 0))
        key = (f"{kind}_{seed}_{case['physical_seed']}_{case['path']}_{case['delay']}"
               f"_m{case['mass']:g}_a{case['amplitude']:g}_t{case['time_scale']:g}")
        np.savez_compressed(trace_dir / f'{key}.npz', trace=arr, initial_error=initial)
        env.close()
    with csv_path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Detect accidental checkpoint/protocol mutation during evaluation.
    if policy_path and metadata['checkpoint_sha256'] != file_hash(policy_path):
        raise RuntimeError('Checkpoint changed during evaluation.')
    if protocol.is_file() and metadata['protocol_sha256'] != file_hash(protocol):
        raise RuntimeError('Protocol changed during evaluation.')
    metadata.update(status='completed', finished=time.time(), episodes=len(rows),
                    completed=sum(row['completed'] for row in rows))
    marker.write_text(json.dumps(metadata, indent=2))
    print(stage, kind, seed, len(rows), metadata['completed'],
          np.mean([row['rmse_penalized'] for row in rows]), flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=PLANS, required=True)
    parser.add_argument('--kind', choices=KINDS, required=True)
    parser.add_argument('--seed', type=int, default=-1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    evaluate(args.stage, args.kind, args.seed)
