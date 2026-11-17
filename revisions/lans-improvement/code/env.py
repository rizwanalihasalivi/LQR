import sys
from pathlib import Path
from functools import lru_cache
import numpy as np
SOURCE=Path(__file__).resolve().parents[2]/'dynamic-bicycle/code'
sys.path.insert(1,str(SOURCE))
from dynamic_env import DynamicEnv,HORIZON,DT
class ImprovedEnv(DynamicEnv):
 def __init__(self,*,kind='lans',residual_scale=.02,beta_decay=.8,training=False,amplitude=1.,time_scale=1.,training_budget=200704,training_delay_curriculum=True,**kwargs):
  self.residual_scale=residual_scale;self.beta_decay=beta_decay;self.amplitude=amplitude;self.time_scale=time_scale
  if training_budget<=0:raise ValueError('training_budget must be positive')
  self.training_episode=0;self.total_steps=0;self.training_budget=int(training_budget)
  self.training_delay_curriculum=bool(training_delay_curriculum)
  self.training_delay_steps=dict(total=0,zero=0,above_10ms=0,at_200ms=0,maximum_ms=0.)
  super().__init__(variant=kind,training=training,**kwargs)
  if not training:self._reference()
 def _reference(self):
  from dynamic_env import reference
  arr=np.asarray(reference(self.path,np.arange(HORIZON+6)*DT*self.time_scale))
  arr*=self.amplitude*np.array([1,self.time_scale,self.time_scale**2,self.time_scale**3])[:,None]
  self.ref=arr
 def reset(self,*,seed=None,options=None):
  obs,info=super().reset(seed=seed,options=options)
  if self.training:
   self.path=str(self.np_random.choice(['dlc','step','sine','slalom']))
   self.amplitude=float(self.np_random.uniform(.8,1.2));self.time_scale=float(self.np_random.uniform(.85,1.15))
   self.mass=float(self.np_random.uniform(1100,2200))
   from dynamic_env import plant
   self.Ad,self.Bd,self.Gd,self.Hd=plant(self.mass,self.vx)
   phase=self.total_steps/self.training_budget if self.training_delay_curriculum else 1.
   if phase<.25:self.delays[:]=0
   elif phase<.5:self.delays[:]=float(self.np_random.choice([0,10]))
   elif self.np_random.random()<.3:
    self.delays[:]=0;mask=self.np_random.random(len(self.delays))<.1;self.delays[mask]=self.np_random.uniform(5,130,mask.sum())
   else:self.delays[:]=float(self.np_random.choice([0,10,20,50,100,200]))
  self._reference()
  return self.obs(),info
 def obs(self):
  raw=super().obs().astype(float)
  # Same information, invertible scaling of the four straight-heading previews:
  # remove duplicate present error/heading terms and normalize by horizon squared.
  horizons=DT*np.arange(1,5)
  raw[5:9]=(raw[5:9]-self.x[0]-self.vx*self.x[2]*horizons)/(10*horizons**2)
  raw[:4]=self.x/np.array([.1,.5,.05,.2])
  return raw.astype(np.float32)
 def compose(self,action,delay):
  res=self.residual_scale*float(np.clip(np.asarray(action).ravel()[0],-1,1))
  base=-float(self.K@self.x)
  if self.variant=='ppo':base=0.;gain=1.
  elif self.variant=='lqr':gain=0.
  elif self.variant=='mpc':base=0.;gain=1.
  else:gain=float(np.exp(-self.beta_decay*max(0.,delay-5)))
  return float(np.clip(base+gain*res,-.5,.5)),res,gain,base
 def step(self,action):
  if self.training:
   delay=float(self.delays[self.k]);coverage=self.training_delay_steps
   coverage['total']+=1;coverage['zero']+=int(delay==0.)
   coverage['above_10ms']+=int(delay>10.);coverage['at_200ms']+=int(delay==200.)
   coverage['maximum_ms']=max(coverage['maximum_ms'],delay)
  previous=self.prev_u
  obs,old_r,t,tr,info=super().step(action)
  # Reward no longer credits an external latency value as controlled computation.
  reward=-100.*(self.x[0]**2+4*self.x[2]**2)-.01*info['u']**2-.1*(info['u']-previous)**2
  if t:reward-=250.
  self.total_steps+=1
  return obs,float(reward),t,tr,info
