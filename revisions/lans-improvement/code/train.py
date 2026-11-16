import argparse,json,time
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env import ImprovedEnv
from protocol import P
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
torch.set_num_threads(1);torch.set_num_interop_threads(1)
c=P['candidates'][a.name];dest=ROOT/'models'/f'{a.name}_{a.seed}';dest.mkdir(exist_ok=True)
e=Monitor(ImprovedEnv(kind=c['kind'],residual_scale=c['residual_scale'],beta_decay=c['beta_decay'],training=True,delay_mode='none'),str(dest/'monitor'))
m=PPO('MlpPolicy',e,seed=a.seed,device='cpu',learning_rate=3e-4,n_steps=1024,batch_size=64,n_epochs=10,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.0001,vf_coef=.5,policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.ReLU,log_std_init=c['log_std']),verbose=0)
start=time.time();m.learn(P['training_steps_per_policy']);m.save(dest/'policy')
meta=dict(candidate=a.name,seed=a.seed,actual_steps=m.num_timesteps,seconds=time.time()-start,configuration=c)
(dest/'training.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
