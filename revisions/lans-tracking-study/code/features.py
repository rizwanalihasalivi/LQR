import numpy as np
from core import INV,ImprovedEnv,KR
from scipy.linalg import null_space
# Full-rank linear conditioning of the same four existing preview channels.
null=null_space(INV).ravel()
T=np.column_stack([-INV[0],-.6*INV[1],-.48*INV[2],1000*null])
TI=np.linalg.inv(T)
def whiten(obs):
 y=np.array(obs,copy=True);y[...,5:9]=y[...,5:9]@T
 return y.astype(np.float32)
def unwhiten(obs):
 y=np.asarray(obs,dtype=float).copy();y[...,5:9]=y[...,5:9]@TI
 return y
class NetworkEnv(ImprovedEnv):
 def __init__(self,kind='lans',**kwargs):
  super().__init__(kind=kind,residual_scale=1. if kind=='lans' else .5,beta_decay=.002,**kwargs)
  self.K=KR.copy()
 def obs(self):return whiten(super().obs())
 def step(self,action):
  prev=self.prev_u
  obs,_,t,tr,info=super().step(action)
  reward=-1000*self.x[0]**2-.01*self.x[2]**2-.001*info['u']**2-.01*(info['u']-prev)**2
  if t:reward-=250.
  return obs,float(reward),t,tr,info
