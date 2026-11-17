"""Prospective corrective trial after unrestricted PPO regressed on validation.
Existing BC, unrestricted checkpoints and failures remain intact."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from features import NetworkEnv
from training_utils import ParameterBoxProjection
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['lans','ppo'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
torch.set_num_threads(1);torch.set_num_interop_threads(1)
dest=ROOT/'models'/f'limited_{a.kind}_{a.seed}';dest.mkdir(exist_ok=True)
training_steps=100352
# BC has already supplied a competent policy. Fine-tune on the complete delay
# distribution immediately, instead of restarting a no-delay curriculum.
e=Monitor(NetworkEnv(kind=a.kind,training=True,delay_mode='none',training_budget=training_steps,training_delay_curriculum=False),str(dest/'monitor'))
m=PPO.load(ROOT/'models'/f'{a.kind}_{a.seed}'/'bc.zip',env=e,device='cpu')
m.learning_rate=1e-5;m.lr_schedule=lambda _:1e-5
for parameter in m.policy.mlp_extractor.policy_net.parameters():parameter.requires_grad_(False)
m.policy.log_std.requires_grad_(False)
bound=1e-5/(1. if a.kind=='lans' else .5)
projection=ParameterBoxProjection(m.policy.action_net.parameters(),bound)
hook=m.policy.optimizer.register_step_post_hook(projection.after_step)
start=time.time()
try:m.learn(training_steps)
finally:hook.remove()
projection.project();m.save(dest/'policy')
delta=projection.max_change()
meta=dict(kind=a.kind,seed=a.seed,PPO_steps=m.num_timesteps,additional_PPO_steps=training_steps,seconds=time.time()-start,hidden_layers_frozen=True,exploration_logstd_frozen=True,output_parameter_bound=bound,max_output_parameter_change=delta,projection_optimizer_steps=projection.optimizer_steps,projection_frequency='after_every_optimizer_step',training_delay_curriculum=False,training_budget=e.unwrapped.training_budget,training_delay_coverage=e.unwrapped.training_delay_steps,scope='Projected output-layer PPO fine-tuning; pretraining dominates. Not unmodified vanilla PPO, and not proof of additional PPO benefit.')
(dest/'training.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
