from pathlib import Path
import argparse,time,json
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from robust_env import RobustEnv,robust_gain
from dynamic_env import design
from study_plan import PLAN
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['lans','ppo'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
torch.set_num_threads(1);torch.set_num_interop_threads(1)
dest=ROOT/'models'/f'{a.kind}_{a.seed}';dest.mkdir(exist_ok=True)
env=Monitor(RobustEnv(kind=a.kind,training=True,delay_mode='none'),str(dest/'monitor'))
model=PPO('MlpPolicy',env,seed=a.seed,device='cpu',learning_rate=3e-4,n_steps=1024,batch_size=64,n_epochs=10,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.0001,vf_coef=.5,policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.ReLU,log_std_init=-4.),verbose=0)
scales=np.array([.1,.5,.05,.2]);fast=design()
coeff=(robust_gain()-fast)*scales/.2 if a.kind=='lans' else -fast*scales/.5
with torch.no_grad():
 first=model.policy.mlp_extractor.policy_net[0];second=model.policy.mlp_extractor.policy_net[2]
 first.weight[:8].zero_();first.bias[:8].zero_();second.weight[:8].zero_();second.bias[:8].zero_()
 for j in range(4):
  first.weight[j,j]=1;first.weight[j+4,j]=-1
 for j in range(8):second.weight[j,j]=1
 model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
 model.policy.action_net.weight[0,:4]=torch.tensor(coeff,dtype=torch.float32)
 model.policy.action_net.weight[0,4:8]=torch.tensor(-coeff,dtype=torch.float32)
 # Unused units retain their random features; zero output weights can learn.
 tests=[]
 for x in [[.01,.02,.001,.002],[-.01,-.02,-.001,-.002]]:
  obs=np.zeros(11,dtype=np.float32);obs[:4]=np.array(x)/scales
  pred=model.predict(obs,deterministic=True)[0][0]
  total=-robust_gain()@x+.2*pred if a.kind=='lans' else .5*pred
  assert abs(total-(-fast@x))<1e-6
  tests.append(float(total))
start=time.time();model.learn(PLAN['steps_per_policy']);model.save(dest/'policy')
meta=dict(kind=a.kind,seed=a.seed,actual_steps=model.num_timesteps,seconds=time.time()-start,initialization='exact fast-LQR prior encoded in existing ReLU actor; same initial total deterministic control for both families; no extra deployed block',initialization_test_commands=tests,robust_gain=robust_gain().tolist(),source='same physical dynamics and11 observations as preceding study')
(dest/'training.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
