"""Constrained linear MPC; same ZOH dynamic bicycle model and Q/R as LQR."""
import numpy as np
from scipy.optimize import minimize
from dynamic_env import plant
from model_audit import Q,R
class MPC:
 def __init__(self,N=20):
  self.N=N;A,B,G,H=plant();self.A=A;self.B=B;self.G=G;self.H=H
  S=np.zeros((4*N,N));T=np.zeros((4*N,4))
  for k in range(N):
   T[k*4:k*4+4]=np.linalg.matrix_power(A,k+1)
   for j in range(k+1):S[k*4:k*4+4,j]=np.linalg.matrix_power(A,k-j)@B
  W=np.kron(np.eye(N),Q);self.S=S;self.T=T;self.W=W;self.Hessian=S.T@W@S+np.eye(N)*R.item();self.inv=np.linalg.inv(self.Hessian);self.last=np.zeros(N);self.failures=0
 def action(self,e):
  offset=np.zeros(4);stack=[]
  for j in range(self.N):
   k=min(e.k+j,e.ref.shape[1]-1)
   offset=self.A@offset+self.G*(e.ref[2,k]/e.vx)+self.H*(e.ref[3,k]/e.vx);stack.extend(offset)
  c=self.S.T@self.W@(self.T@e.x+np.array(stack));uncon=-self.inv@c
  if np.max(abs(uncon))<=.5:u=uncon
  else:
   result=minimize(lambda u:(float(u@self.Hessian@u+2*c@u),2*(self.Hessian@u+c)),self.last,jac=True,method='L-BFGS-B',bounds=[(-.5,.5)]*self.N,options=dict(maxiter=100,ftol=1e-10,gtol=1e-8))
   if not result.success:self.failures+=1
   u=result.x
  self.last=np.r_[u[1:],u[-1]]
  return float(u[0])
