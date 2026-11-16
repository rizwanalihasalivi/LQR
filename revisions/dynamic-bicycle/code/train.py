import argparse,json,time
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from dynamic_env import DynamicEnv
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--kind',choices=['full','no_reward','ppo'],required=True);p.add_argument('--steps',type=int,default=100352);a=p.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 out=ROOT/'models'/f'{a.kind}_{a.seed}';out.mkdir(exist_ok=True)
 env=Monitor(DynamicEnv(variant='ppo' if a.kind=='ppo' else 'lans',delay_mode='mixed',training=True,reward_latency=a.kind!='no_reward'),str(out/'monitor'))
 model=PPO('MlpPolicy',env,seed=a.seed,device='cpu',learning_rate=3e-4,n_steps=1024,batch_size=64,n_epochs=10,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.01,vf_coef=.5,policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.ReLU,log_std_init=-2.),verbose=0)
 begin=time.time();model.learn(a.steps);model.save(out/'policy')
 meta=dict(seed=a.seed,kind=a.kind,requested_steps=a.steps,actual_steps=model.num_timesteps,seconds=time.time()-begin,device='cpu',policy_distribution='diagonal Gaussian with learned state-independent log_std; action clipped to [-1,1] by SB3 before env; deterministic evaluation uses clipped mean; no tanh output claim',curriculum='none: fixed 25 m/s DLC; new matched reconstruction campaign, not original curriculum replication')
 (out/'training.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
