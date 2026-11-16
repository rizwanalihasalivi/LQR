from pathlib import Path
import sys,json,time,csv
import numpy as np
from scipy.linalg import solve_discrete_are
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'lans-improvement/code'))
from env import ImprovedEnv
from model_audit import model
A,B,Ad,Bd,_,_=model('signed')
def radius(K,d):
 if d==0:return float(max(abs(np.linalg.eigvals(Ad-Bd[:,None]*K))))
 F=np.zeros((4+d,4+d));F[:4,:4]=Ad;F[:4,4]=Bd
 if d>1:F[4:-1,5:]=np.eye(d-1)
 F[-1,:4]=-K
 return float(max(abs(np.linalg.eigvals(F))))
candidates=[]
for heading in [.01,1.,100.]:
 for r in [4000.,40000.,400000.]:
  Q=np.diag([100.,4.,heading,1.]);R=np.array([[r]])
  P=solve_discrete_are(Ad,Bd[:,None],Q,R);K=np.linalg.solve(R+Bd[None,:]@P@Bd[:,None],Bd[None,:]@P@Ad).ravel()
  rho=max(radius(K,d) for d in range(21))
  record=dict(heading_weight=heading,R=r,Q=Q.tolist(),K=K.tolist(),maximum_fixed_delay_radius=rho,eligible=rho<1.)
  if record['eligible']:
   rows=[]
   for path in ['dlc','step','sine','slalom']:
    for delay in ['none','mixed','fixed200']:
     for seed in [8000,8001]:
      e=ImprovedEnv(kind='lqr',path=path,delay_mode=delay);e.K=K;_,_=e.reset(seed=seed);initial=float(e.x[0]);errors=[]
      for k in range(1000):
       _,_,t,tr,i=e.step([0]);errors.append(i['error'])
       if t or tr:break
      n=len(errors);sse=initial**2+np.square(errors).sum()
      rows.append(dict(path=path,delay=delay,seed=seed,completed=not t and tr,penalized_rmse=float(np.sqrt((sse+(1000-n)*max(2.5,abs(errors[-1]))**2)/1001))))
   record.update(completion=float(np.mean([x['completed'] for x in rows])),mean_penalized_rmse=float(np.mean([x['penalized_rmse'] for x in rows])),validation_rows=rows)
  candidates.append(record);print(heading,r,record.get('completion'),record.get('mean_penalized_rmse'),rho,flush=True)
eligible=[c for c in candidates if c['eligible']];assert eligible
chosen=min(eligible,key=lambda c:(-c['completion'],c['mean_penalized_rmse'],c['R'],c['heading_weight']))
output=dict(scope='Validation-only tuning of LQR Q/R, retaining the same controller family and vehicle. Eligibility requires Schur stability for each constant delay0..200ms in10ms steps; not a proof for arbitrary varying delay or saturation.',candidates=candidates,chosen=chosen)
(ROOT/'results/fallback_selection.json').write_text(json.dumps(output,indent=2))
print('SELECTED',chosen['K'],chosen['completion'],chosen['mean_penalized_rmse'],flush=True)
