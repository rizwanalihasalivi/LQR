from pathlib import Path
import json,itertools,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from core import A,B,KR,model
from batch_sim import prepare,build_cases,rollout
ROOT=Path(__file__).resolve().parents[1]
_,_,Ad,Bd,_,_=model('signed')
def stable(p):
 kf=(A[1]+[p['kp'],p['kd'],0,0])/B[1,0]
 for d in range(21):
  k=KR+np.exp(-p['decay']*d*.01)*(kf-KR)
  if d==0:F=Ad-Bd[:,None]*k
  else:
   F=np.zeros((4+d,4+d));F[:4,:4]=Ad;F[:4,4]=Bd
   if d>1:F[4:-1,5:]=np.eye(d-1)
   F[-1,:4]=-k
  if max(abs(np.linalg.eigvals(F)))>=1:return False
 return True
def run(p):
 data=prepare(build_cases([22000,22001]))
 score,ok=rollout(data,p)
 return dict(parameters=p,completion=float(ok.mean()),mean_rmse=float(score.mean()),maximum_rmse=float(score.max()),condition_scores=score.tolist())
if __name__=='__main__':
 grid=[dict(kp=k,kd=2*np.sqrt(k),decay=d,proxy=q,fourth=f) for k,d,q,f in itertools.product([25.,100.,400.],[40.,80.,160.],[1.,5.],[0.,1.])]
 eligible=[p for p in grid if stable(p)];print('eligible',len(eligible),'of',len(grid),flush=True)
 with ProcessPoolExecutor(max_workers=4) as pool:
  results=[]
  for r in pool.map(run,eligible):
   results.append(r);print(r['parameters'],r['completion'],r['mean_rmse'],flush=True)
 best=min(results,key=lambda r:(-r['completion'],r['mean_rmse']))
 (ROOT/'results/teacher_search.json').write_text(json.dumps(dict(scope='Development/validation-only static11-input teacher search. This analytic teacher is offline and is NOT a deployed LANS controller or a learned PPO result.',all_results=results,selected=best),indent=2))
 print('SELECTED',best,flush=True)
