"""Supplementary MPC with the new tracking/control objective, not deployed LANS."""
import numpy as np
from scipy import sparse
import osqp
from dynamic_env import plant
class MatchedMPC:
 def __init__(self,N=20):
  self.N=N;self.A,self.B,self.G,self.H=plant()
  self.T=np.zeros((4*N,4));S=np.zeros((4*N,N))
  for k in range(N):
   self.T[4*k:4*k+4]=np.linalg.matrix_power(self.A,k+1)
   for j in range(k+1):S[4*k:4*k+4,j]=np.linalg.matrix_power(self.A,k-j)@self.B
  W=np.kron(np.eye(N),np.diag([1000.,0.,.01,0.]))
  D=np.eye(N)-np.eye(N,k=-1)
  self.SW=S.T@W;self.Hessian=S.T@W@S+.001*np.eye(N)+.01*D.T@D
  self.inv=np.linalg.inv(self.Hessian);self.last=np.zeros(N);self.failures=0
  self.solver=osqp.OSQP()
  self.solver.setup(P=sparse.csc_matrix(2*self.Hessian),q=np.zeros(N),A=sparse.eye(N,format='csc'),l=np.full(N,-.5),u=np.full(N,.5),eps_abs=1e-8,eps_rel=1e-8,max_iter=50000,polishing=True,verbose=False)
 def action(self,e):
  x=np.zeros(4);stack=[]
  for j in range(self.N):
   k=min(e.k+j,e.ref.shape[1]-1);x=self.A@x+self.G*(e.ref[2,k]/25)+self.H*(e.ref[3,k]/25);stack.extend(x)
  c=self.SW@(self.T@e.x+np.asarray(stack));c[0]-=.01*e.prev_u
  u=-self.inv@c
  if np.max(abs(u))>.5:
   self.solver.update(q=2*c);self.solver.warm_start(x=self.last)
   result=self.solver.solve(raise_error=False)
   if result.info.status_val!=1:
    self.failures+=1
    raise RuntimeError('Matched MPC solve failed: '+result.info.status)
   u=result.x
  self.last=np.r_[u[1:],u[-1]];return float(u[0])
