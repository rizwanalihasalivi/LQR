import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from dynamic_env import DynamicEnv,CommandQueue,beta
from model_audit import model,gains,Q,R
ROOT=Path(__file__).resolve().parents[1];checks={}
q=CommandQueue();out=[q.push(k,.2 if k==0 else .3,30) for k in range(5)]
assert out[:3]==[0.,0.,0.] and out[3]==.2 and out[4]==.3;checks['queue_30ms_exact_execution']=out
q=CommandQueue();out=[q.push(k,k,30 if k==0 else 0) for k in range(5)]
assert out[:3]==[0.,0.,0.] and q.applied_generation==4;checks['fifo_no_premature_execution']=True
assert beta(0)==beta(5)==1 and beta(200)<1e-60;checks['original_beta_preserved']=True
A,B,Ad,Bd,G,H=model('signed');M=np.zeros((5,5));M[:4,:4]=A;M[:4,4]=B[:,0]
x=np.array([.01,.02,.003,.004]);u=.02;expected=(expm(M*.01)@np.r_[x,u])[:4]
assert np.allclose(Ad@x+Bd*u,expected,atol=1e-13);checks['exact_zoh_matches_continuous_augmentation']=True
# Independent integration of the two physical lateral/yaw equations, then
# transformation to error coordinates, guards against a mislabeled plant.
from scipy.integrate import solve_ivp
maxerr=0.
rng=np.random.default_rng(937)
for mass in [1100.,1650.,2200.]:
 for case in range(3):
  x0=rng.normal(0,[.02,.02,.002,.002]);u=float(rng.uniform(-.02,.02))
  z0=[x0[0],x0[1]-25*x0[2],x0[2],x0[3]]
  def physical(t,z):
   ey,vy,psi,r=z;cf=cr=80000.;lf=1.2;lr=1.4;iz=4224.;v=25.
   vydot=-2*(cf+cr)/(mass*v)*vy+(2*(lr*cr-lf*cf)/(mass*v)-v)*r+2*cf/mass*u
   rdot=2*(lr*cr-lf*cf)/(iz*v)*vy-2*(lf*lf*cf+lr*lr*cr)/(iz*v)*r+2*lf*cf/iz*u
   return [vy+v*psi,vydot,r,rdot]
  z=solve_ivp(physical,[0,.01],z0,rtol=1e-11,atol=1e-13).y[:,-1]
  physical_x=np.array([z[0],z[1]+25*z[2],z[2],z[3]])
  _,_,aa,bb,_,_=model('signed',mass)
  maxerr=max(maxerr,float(np.max(abs(aa@x0+bb*u-physical_x))))
assert maxerr<1e-9
checks['independent_physical_bicycle_integration_max_error']=maxerr
checks['independent_physical_bicycle_cases']=9
# Same exogenous schedules: reward on/off changes reward only, never transition.
a=DynamicEnv(reward_latency=True);b=DynamicEnv(reward_latency=False);a.reset(seed=711);b.reset(seed=711)
for k in range(1000):
 action=[.01*np.sin(k)];oa,ra,ta,tra,ia=a.step(action);ob,rb,tb,trb,ib=b.step(action)
 np.testing.assert_array_equal(oa,ob);assert np.isclose(ra-rb,ia['latency_term']);assert ta==tb
 if ta or tra:break
checks['reward_intervention_identical_trajectory_steps']=k+1
K,P=gains('signed');F=Ad-Bd[:,None]*K;W=Q+K[:,None]@K[None,:]*R.item();res=F.T@P@F-P+W
assert max(abs(res.ravel()))<1e-8 and np.linalg.eigvalsh(P).min()>0 and max(abs(np.linalg.eigvals(F)))<1
checks['discrete_lyapunov_max_residual']=float(max(abs(res.ravel())))
# Certified local unsaturated invariant ellipsoid for residual=0, reference=0.
c=.5**2/float(K@np.linalg.solve(P,K));checks['local_lqr_invariant_ellipsoid_c']=c
checks['limitation']='Ellipsoid certificate applies only to zero-reference zero-residual no-delay local LQR; delayed or saturated hybrid globally is not certified.'
(ROOT/'results/checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
# Exact sampled fixed-delay LQR eigenvalues, all-command channel.
rows=[]
for d in range(31):
 if d==0:F=Ad-Bd[:,None]*K
 else:
  F=np.zeros((4+d,4+d));F[:4,:4]=Ad;F[:4,4]=Bd
  if d>1:F[4:4+d-1,5:4+d]=np.eye(d-1)
  F[-1,:4]=-K
 rows.append(dict(delay_steps=d,delay_ms=d*10,spectral_radius=float(max(abs(np.linalg.eigvals(F)))),schur=bool(max(abs(np.linalg.eigvals(F)))<1)))
(ROOT/'results/fixed_delay_stability.json').write_text(json.dumps(rows,indent=2))
print('FIXED-DELAY SCHUR CASES',[x['delay_ms'] for x in rows if x['schur']])
