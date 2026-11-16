"""Prospective corrective trial after unrestricted PPO regressed on validation.
Existing BC, unrestricted checkpoints and failures remain intact."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from features import NetworkEnv
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['lans','ppo'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
torch.set_num_threads(1);torch.set_num_interop_threads(1)
dest=ROOT/'models'/f'limited_{a.kind}_{a.seed}';dest.mkdir(exist_ok=True)
e=Monitor(NetworkEnv(kind=a.kind,training=True,delay_mode='none'),str(dest/'monitor'))
m=PPO.load(ROOT/'models'/f'{a.kind}_{a.seed}'/'bc.zip',env=e,device='cpu')
m.learning_rate=1e-5;m.lr_schedule=lambda _:1e-5
for parameter in m.policy.mlp_extractor.policy_net.parameters():parameter.requires_grad_(False)
m.policy.log_std.requires_grad_(False)
reference=[p.detach().clone() for p in m.policy.action_net.parameters()]
bound=1e-5/(1. if a.kind=='lans' else .5)
def project():
 with torch.no_grad():
  for p,r in zip(m.policy.action_net.parameters(),reference):p.copy_(torch.maximum(torch.minimum(p,r+bound),r-bound))
class Projection(BaseCallback):
 def _on_rollout_start(self):project()
 def _on_step(self):return True
start=time.time();m.learn(100352,callback=Projection());project();m.save(dest/'policy')
delta=max(float(torch.max(abs(p-r)).detach()) for p,r in zip(m.policy.action_net.parameters(),reference))
meta=dict(kind=a.kind,seed=a.seed,PPO_steps=m.num_timesteps,additional_PPO_steps=100352,seconds=time.time()-start,hidden_layers_frozen=True,exploration_logstd_frozen=True,output_parameter_bound=bound,max_output_parameter_change=delta,scope='Projected output-layer PPO fine-tuning; pretraining dominates. Not unmodified vanilla PPO, and not proof of additional PPO benefit.')
(dest/'training.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
