"""Training constraints shared by projected PPO and its regression checks."""
import torch

class ParameterBoxProjection:
 """Project after every optimizer step, before the next PPO minibatch.

 A rollout callback runs before optimization, so it cannot enforce a bound
 across the multiple epochs/minibatches in one PPO update.
 """
 def __init__(self,parameters,bound):
  if bound<0:raise ValueError('projection bound must be nonnegative')
  self.parameters=list(parameters)
  self.reference=[p.detach().clone() for p in self.parameters]
  self.bound=float(bound);self.optimizer_steps=0
 def project(self):
  with torch.no_grad():
   for p,r in zip(self.parameters,self.reference):
    p.copy_(torch.maximum(torch.minimum(p,r+self.bound),r-self.bound))
 def after_step(self,optimizer,args,kwargs):
  self.project();self.optimizer_steps+=1
 def max_change(self):
  return max(float(torch.max(abs(p-r)).detach()) for p,r in zip(self.parameters,self.reference))
