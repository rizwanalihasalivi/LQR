"""Reproducible separate campaign; never overwrite the historical policies."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from precise_features import PrecisionEnv
from features import unwhiten
from core import teacher, KR, XS
from training_utils import ParameterBoxProjection

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['lans', 'ppo'], required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    protocol = json.loads((ROOT / 'results/protocol.json').read_text())
    params = protocol['teacher']
    dest = ROOT / 'models' / f'corrected_{a.kind}_{a.seed}'
    dest.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    steps = protocol['PPO_steps']
    env = Monitor(PrecisionEnv(kind=a.kind, training=True, delay_mode='none',
                              training_budget=steps, training_delay_curriculum=False),
                  str(dest / 'monitor'))
    model = PPO('MlpPolicy', env, seed=a.seed, device='cpu', learning_rate=1e-5,
                n_steps=1024, batch_size=64, n_epochs=10, gamma=.99,
                gae_lambda=.95, clip_range=.2, ent_coef=0., vf_coef=.5,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                                   activation_fn=torch.nn.ReLU,
                                   log_std_init=-5.7 if a.kind == 'lans' else -5.), verbose=0)
    weights = np.array([teacher(unwhiten(np.eye(11)[j] * .01), params) / .01
                        for j in range(11)])
    weights[9:] = 0
    if a.kind == 'lans':
        weights[:4] += KR * XS
    else:
        weights /= .5
    with torch.no_grad():
        l1 = model.policy.mlp_extractor.policy_net[0]
        l2 = model.policy.mlp_extractor.policy_net[2]
        l1.weight[:18].zero_(); l1.bias[:18].zero_()
        l2.weight[:18].zero_(); l2.bias[:18].zero_()
        for j in range(9):
            l1.weight[j, j] = 1; l1.weight[j + 9, j] = -1
        for j in range(18):
            l2.weight[j, j] = 1
        model.policy.action_net.weight.zero_(); model.policy.action_net.bias.zero_()
        model.policy.action_net.weight[0, :9] = torch.tensor(weights[:9], dtype=torch.float32)
        model.policy.action_net.weight[0, 9:18] = torch.tensor(-weights[:9], dtype=torch.float32)
    with np.load(ROOT / 'results/corpus.npz') as data:
        X = torch.tensor(data['X'])
        Y = torch.tensor(data[a.kind + '_target'])[:, None]
    optimizer = torch.optim.Adam(list(model.policy.mlp_extractor.policy_net.parameters())
                                 + list(model.policy.action_net.parameters()), lr=3e-4)
    generator = torch.Generator().manual_seed(a.seed)
    logs = []; start = time.time()
    for update in range(protocol['BC_updates']):
        idx = torch.randint(len(X), (512,), generator=generator)
        output = model.policy.action_net(model.policy.mlp_extractor.policy_net(X[idx]))
        loss = ((output - Y[idx]) * (1. if a.kind == 'lans' else .5)).square().mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.)
        optimizer.step()
        if update % 500 == 0:
            logs.append(dict(update=update,physical_steering_mse=float(loss.detach())))
    model.save(dest / 'bc')
    for parameter in model.policy.mlp_extractor.policy_net.parameters():
        parameter.requires_grad_(False)
    model.policy.log_std.requires_grad_(False)
    bound = 1e-5 / (1. if a.kind == 'lans' else .5)
    projection = ParameterBoxProjection(model.policy.action_net.parameters(), bound)
    hook = model.policy.optimizer.register_step_post_hook(projection.after_step)
    try:
        model.learn(steps)
    finally:
        hook.remove()
    projection.project(); model.save(dest / 'policy')
    meta = dict(kind=a.kind, seed=a.seed, PPO_steps=model.num_timesteps,
                BC_updates=protocol['BC_updates'], BC_log=logs,
                projection_optimizer_steps=projection.optimizer_steps,
                max_output_parameter_change=projection.max_change(),
                training_delay_coverage=env.unwrapped.training_delay_steps,
                seconds=time.time()-start,features=11,architecture='64x64 ReLU',
                teacher_is_offline_only=True,precision_before_float32=True)
    (dest / 'training.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
