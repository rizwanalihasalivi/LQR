"""Same dynamic plant and same11 inputs. Analytic teacher is offline only."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(1,str(ROOT.parent/'lans-improvement/code'))
from env import ImprovedEnv
from model_audit import model
from dynamic_env import design
KR=np.array(json.loads((ROOT.parent/'lans-robust-fallback/results/robust_gain.json').read_text())['chosen']['K'])
A,B,_,_,_,_=model('signed')
v=25.;cf=cr=80000.;lf=1.2;lr=1.4;m=1650.;I=4224.
G=np.array([0,2*(lr*cr-lf*cf)/(m*v)-v,0,-2*(lf*lf*cf+lr*lr*cr)/(I*v)])
H=np.array([0,0,0,-1.])
f2=-G[1]/(B[1,0]*v)
Z=np.array([[0.,1.],[A[3,2]-B[3,0]*A[1,2]/B[1,0],A[3,3]-B[3,0]*A[1,3]/B[1,0]]])
b2=np.array([0.,B[3,0]*f2+G[3]/v]);b3=np.array([0.,H[3]/v])
zi=np.linalg.inv(Z)
p0=(-zi@b2)[0];p1=(-zi@zi@b2-zi@b3)[0];p2=(-zi@zi@zi@b2-zi@zi@b3)[0]
D=np.array([f2-A[1,2]/B[1,0]*p0,-(A[1,2]*p1+A[1,3]*p0)/B[1,0],-(A[1,2]*p2+A[1,3]*p1)/B[1,0]])
h=.01*np.arange(1,5);V=np.stack([np.ones(4),h,h*h],1);INV=np.linalg.pinv(V)
XS=np.array([.1,.5,.05,.2])
def beta(ms):return np.exp(-.002*np.maximum(0,np.asarray(ms)-5))
def teacher(obs,params):
 obs=np.asarray(obs,dtype=float);one=obs.ndim==1
 if one:obs=obs[None,:]
 x=obs[:,:4]*XS
 poly=obs[:,5:9]@INV.T
 deriv=np.stack([-20*poly[:,0],-60*poly[:,1],-240*poly[:,2]],1)
 deriv=np.clip(deriv,[-60,-500,-3000],[60,500,3000])
 delay=np.minimum(200,np.maximum(obs[:,9]*200,params['proxy']*obs[:,10]*200));d=delay/1000
 kf=(A[1]+np.array([params['kp'],params['kd'],0,0]))/B[1,0]
 blend=np.exp(-params['decay']*d)
 K=KR[None,:]+blend[:,None]*(kf-KR)[None,:]
 ff=np.stack([np.full(len(obs),D[0]),D[1]+d*D[0],D[2]+d*D[1]+.5*d*d*D[0]],1)
 ff+=K[:,2,None]*np.array([p0,p1,p2])[None,:]+K[:,3,None]*np.array([0,p0,p1])[None,:]
 ff[:,2]*=params['fourth']
 target=-np.sum(K*x,1)+np.sum(ff*deriv,1)
 return target[0] if one else target
class StudyEnv(ImprovedEnv):
 def __init__(self,kind='lans',**kwargs):
  super().__init__(kind=kind,residual_scale=1.0 if kind=='lans' else .5,beta_decay=.002,**kwargs)
  self.K=KR.copy()
