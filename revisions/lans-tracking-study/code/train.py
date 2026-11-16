from pathlib import Path
import argparse,json,time
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from core import teacher,KR,XS
from features import NetworkEnv,unwhiten
from protocol import P
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['lans','ppo'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
torch.set_num_threads(1);torch.set_num_interop_threads(1)
params=json.loads((ROOT/'results/teacher_search.json').read_text())['selected']['parameters']
dest=ROOT/'models'/f'{a.kind}_{a.seed}';dest.mkdir(exist_ok=True)
env=Monitor(NetworkEnv(kind=a.kind,training=True,delay_mode='none'),str(dest/'monitor'))
m=PPO('MlpPolicy',env,seed=a.seed,device='cpu',learning_rate=P['training']['PPO_learning_rate'],n_steps=1024,batch_size=64,n_epochs=10,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.ReLU,log_std_init=-5.7 if a.kind=='lans' else -5.),verbose=0)
# Encode exact linear zero-delay teacher in the existing ReLU network.
w=np.array([teacher(unwhiten(np.eye(11)[j]*.01),params)/.01 for j in range(11)])
w[9:]=0
if a.kind=='lans':w[:4]+=KR*XS
else:w/=.5
with torch.no_grad():
 l1=m.policy.mlp_extractor.policy_net[0];l2=m.policy.mlp_extractor.policy_net[2]
 l1.weight[:18].zero_();l1.bias[:18].zero_();l2.weight[:18].zero_();l2.bias[:18].zero_()
 for j in range(9):l1.weight[j,j]=1;l1.weight[j+9,j]=-1
 for j in range(18):l2.weight[j,j]=1
 m.policy.action_net.weight.zero_();m.policy.action_net.bias.zero_()
 m.policy.action_net.weight[0,:9]=torch.tensor(w[:9],dtype=torch.float32);m.policy.action_net.weight[0,9:18]=torch.tensor(-w[:9],dtype=torch.float32)
with np.load(ROOT/'results/corpus.npz') as data:
 X=torch.tensor(data['X']);Y=torch.tensor(data[a.kind+'_target'])[:,None]
optimizer=torch.optim.Adam(list(m.policy.mlp_extractor.policy_net.parameters())+list(m.policy.action_net.parameters()),lr=P['training']['BC_learning_rate'])
generator=torch.Generator().manual_seed(a.seed);logs=[];start=time.time()
for step in range(P['training']['BC_updates']):
 idx=torch.randint(len(X),(P['training']['batch_size'],),generator=generator)
 pred=m.policy.action_net(m.policy.mlp_extractor.policy_net(X[idx]))
 loss=((pred-Y[idx])*(1. if a.kind=='lans' else .5)).square().mean()
 optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.policy.parameters(),1.);optimizer.step()
 if step%500==0:logs.append(dict(step=step,physical_steering_mse=float(loss.detach())))
m.save(dest/'bc')
(dest/'bc_log.json').write_text(json.dumps(logs,indent=2));bc_seconds=time.time()-start
m.learn(P['training']['PPO_steps']);m.save(dest/'policy')
meta=dict(kind=a.kind,seed=a.seed,BC_updates=P['training']['BC_updates'],BC_seconds=bc_seconds,last_BC_physical_MSE=logs[-1]['physical_steering_mse'],PPO_steps=m.num_timesteps,total_seconds=time.time()-start,teacher_is_offline_only=True,features=11,policy_architecture='64x64ReLU',corpus_rows=len(X),PPO_learning_rate=P['training']['PPO_learning_rate'])
(dest/'training.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
