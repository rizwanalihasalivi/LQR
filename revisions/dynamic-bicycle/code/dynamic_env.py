"""Dynamic bicycle LANS reproduction; no archived kinematic simulator is used."""
from collections import deque
from functools import lru_cache
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from model_audit import model,gains
DT=.01;HORIZON=1000;DELTA_MAX=.5
@lru_cache(None)
def plant(mass=1650.,vx=25.):return model('signed',mass,vx)[2:]
@lru_cache(None)
def design(vx=25.):return gains('signed',vx)[0]
def beta(delay_ms):return float(np.exp(-.8*max(0.,delay_ms-5.)))
def reference(path,t):
 t=np.asarray(t)
 if path=='dlc':
  # Smooth lane change out and back; not claimed ISO 3888-2 compliant.
  a=np.tanh((t-2.)/.6);b=np.tanh((t-6.)/.6)
  y=1.75*(a-b);dy=1.75*((1-a*a)-(1-b*b))/.6
  ddy=1.75*(-2*a*(1-a*a)+2*b*(1-b*b))/.36
  dddy=1.75*((-2+6*a*a)*(1-a*a)-(-2+6*b*b)*(1-b*b))/.216
 elif path=='step':
  a=np.tanh((t-3.)/.8);y=1.75*(1+a);dy=1.75*(1-a*a)/.8;ddy=-3.5*a*(1-a*a)/.64;dddy=1.75*(-2+6*a*a)*(1-a*a)/.512
 else:
  amp,freq=(3.,.1) if path=='sine' else (2.,.5)
  w=2*np.pi*freq;y=amp*np.sin(w*t);dy=amp*w*np.cos(w*t);ddy=-amp*w*w*np.sin(w*t);dddy=-amp*w**3*np.cos(w*t)
 return y,dy,ddy,dddy
class CommandQueue:
 """FIFO release: no command can execute before its transport deadline."""
 def __init__(self):self.q=deque();self.last_release=0;self.applied=0.;self.applied_generation=-1
 def push(self,tick,u,delay_ms):
  release=max(tick+int(np.ceil(max(0.,delay_ms)/(1000*DT))),self.last_release)
  self.last_release=release;self.q.append((release,tick,float(u)))
  while self.q and self.q[0][0]<=tick:
   _,self.applied_generation,self.applied=self.q.popleft()
  return self.applied
class DynamicEnv(gym.Env):
 metadata={}
 def __init__(self,variant='lans',delay_mode='mixed',path='dlc',mass=1650.,vx=25.,training=False,reward_latency=True,delay_scope='full'):
  self.variant=variant;self.delay_mode=delay_mode;self.path=path;self.mass=mass;self.vx=vx;self.training=training;self.reward_latency=reward_latency;self.delay_scope=delay_scope
  self.K=design(vx);self.Ad,self.Bd,self.Gd,self.Hd=plant(mass,vx)
  self.observation_space=spaces.Box(-np.inf,np.inf,(11,),np.float32);self.action_space=spaces.Box(-1.,1.,(1,),np.float32)
  ts=np.arange(HORIZON+6)*DT
  self.ref=np.asarray(reference(path,ts))
 def reset(self,*,seed=None,options=None):
  super().reset(seed=seed)
  self.k=0;self.x=self.np_random.normal(0,[.05,.02,.005,.002]);self.queue=CommandQueue();self.prev_u=0.;self.last_delay=0.;self.history=deque(maxlen=10)
  self.delays=np.zeros(HORIZON+1)
  if self.delay_mode=='mixed':
   trigger=self.np_random.random(HORIZON+1)<.1;self.delays[trigger]=self.np_random.uniform(5,130,trigger.sum())
  elif self.delay_mode=='burst130':self.delays[500:510]=130.
  elif self.delay_mode.startswith('fixed'):self.delays[:]=float(self.delay_mode[5:])
  elif self.delay_mode!='none':raise ValueError(self.delay_mode)
  return self.obs(),{}
 def obs(self):
  # Four current errors, speed, four straight-heading preview lateral errors,
  # previous observed transport delay, causal mean of last ten delays.
  y=self.ref[0,self.k]+self.x[0];heading=self.ref[1,self.k]/self.vx+self.x[2]
  preview=y+self.vx*heading*DT*np.arange(1,5)-self.ref[0,self.k+np.arange(1,5)]
  return np.asarray([self.x[0],self.x[1]/5,self.x[2]*10,self.x[3],self.vx/25,*preview,self.last_delay/200,(np.mean(self.history) if self.history else 0)/200],np.float32)
 def compose(self,action,delay):
  raw=float(np.asarray(action).ravel()[0]);res=DELTA_MAX*float(np.clip(raw,-1,1));base=-float(self.K@self.x)
  if self.variant in ('ppo','mpc'):base=0.;gain=1.
  elif self.variant=='no_gain':gain=1.
  elif self.variant=='lqr':gain=0.
  else:gain=beta(delay)
  return float(np.clip(base+gain*res,-DELTA_MAX,DELTA_MAX)),res,gain,base
 def step(self,action):
  delay=float(self.delays[self.k]);u,res,gain,base=self.compose(action,delay)
  if self.delay_scope=='full':applied=self.queue.push(self.k,u,delay)
  elif self.delay_scope=='residual':applied=float(np.clip(base+self.queue.push(self.k,gain*res,delay),-.5,.5))
  else:raise ValueError(self.delay_scope)
  y,d,dd,ddd=self.ref[:,self.k]
  # Small-angle desired yaw rate and derivative: r_ref=y_ref''/vx.
  self.x=self.Ad@self.x+self.Bd*applied+self.Gd*(dd/self.vx)+self.Hd*(ddd/self.vx)
  # Reward coefficients specified here; historical unspecified weights are not inferred.
  tracking=-(self.x[0]**2+4*self.x[2]**2)
  control=-(.01*u*u+.1*(u-self.prev_u)**2)
  latency=-(delay/200.+max(0.,(delay-5)/200.)**2)
  reward=tracking+control+(.02*latency if self.reward_latency else 0.)
  self.prev_u=u;self.last_delay=delay;self.history.append(delay);self.k+=1
  failed=bool(abs(self.x[0])>2.5 or abs(self.x[2])>.5 or not np.isfinite(self.x).all())
  if failed:reward-=25.
  info=dict(error=float(self.x[0]),heading_error=float(self.x[2]),u=u,applied=applied,residual=res,gain=gain,delay_ms=delay,saturation=abs(u)>=.5,queue=len(self.queue.q),generation=self.queue.applied_generation,latency_term=.02*latency,tracking_reward=tracking,control_reward=control)
  return self.obs(),float(reward),failed,self.k>=HORIZON,info
