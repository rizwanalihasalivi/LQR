from pathlib import Path
import sys,json
import numpy as np
from scipy.linalg import solve_discrete_are
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'lans-improvement/code'))
from env import ImprovedEnv
from model_audit import model
_,_,Ad,Bd,_,_=model('signed')
def rho(K,d):
 if d==0:return max(abs(np.linalg.eigvals(Ad-Bd[:,None]*K)))
 F=np.zeros((4+d,4+d));F[:4,:4]=Ad;F[:4,4]=Bd
 if d>1:F[4:-1,5:]=np.eye(d-1)
 F[-1,:4]=-K
 return max(abs(np.linalg.eigvals(F)))
original=json.loads((ROOT/'results/fallback_selection.json').read_text())
records=list(original['candidates'])
for dy in [100.,1000.]:
 for yaw in [1.,100.]:
  for heading in [1.,100.]:
   for rv in [40000.,400000.,4000000.]:
    Q=np.diag([100.,dy,heading,yaw]);R=np.array([[rv]]);P=solve_discrete_are(Ad,Bd[:,None],Q,R)
    K=np.linalg.solve(R+Bd[None,:]@P@Bd[:,None],Bd[None,:]@P@Ad).ravel()
    maximum=float(max(rho(K,d) for d in range(21)))
    record=dict(Q=Q.tolist(),R=rv,K=K.tolist(),maximum_fixed_delay_radius=maximum,eligible=maximum<1.)
    if record['eligible']:
     rows=[]
     for path in ['dlc','step','sine','slalom']:
      for delay in ['none','mixed','fixed200']:
       for seed in [8000,8001]:
        e=ImprovedEnv(kind='lqr',path=path,delay_mode=delay);e.K=K;e.reset(seed=seed);initial=float(e.x[0]);errors=[]
        for k in range(1000):
         _,_,t,tr,i=e.step([0]);errors.append(i['error'])
         if t or tr:break
        n=len(errors);sse=initial**2+np.square(errors).sum()
        rows.append(dict(path=path,delay=delay,seed=seed,completed=not t and tr,penalized_rmse=float(np.sqrt((sse+(1000-n)*max(2.5,abs(errors[-1]))**2)/1001))))
     record.update(completion=float(np.mean([r['completed'] for r in rows])),mean_penalized_rmse=float(np.mean([r['penalized_rmse'] for r in rows])),validation_rows=rows)
    records.append(record)
    print(dy,yaw,heading,rv,record.get('completion'),record.get('mean_penalized_rmse'),maximum,flush=True)
eligible=[r for r in records if r['eligible']]
chosen=min(eligible,key=lambda r:(-r['completion'],r['mean_penalized_rmse'],r['R']))
report=dict(scope=original['scope'],search='Original9 Q/R trials plus24 additional damping-weight trials. All tuning uses validation only; no final test for this stage has run.',candidates=records,chosen=chosen)
(ROOT/'results/robust_gain.json').write_text(json.dumps(report,indent=2));print('CHOSEN',chosen,flush=True)
