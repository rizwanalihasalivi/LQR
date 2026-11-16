from pathlib import Path
import json,sys
import numpy as np
from scipy.linalg import solve_discrete_are
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[0]/'dynamic-bicycle/code'))
from model_audit import model,gains,Q
A,B,Ad,Bd,_,_=model('signed')
K,_=gains('signed')
def radius(k,d):
 if d==0:return float(max(abs(np.linalg.eigvals(Ad-Bd[:,None]*k))))
 m=np.zeros((4+d,4+d));m[:4,:4]=Ad;m[:4,4]=Bd
 if d>1:m[4:-1,5:]=np.eye(d-1)
 m[-1,:4]=-k
 return float(max(abs(np.linalg.eigvals(m))))
rows=[]
for scale in [1,10,100,1000,10000,100000,1000000]:
 R=np.array([[4.*scale]]);P=solve_discrete_are(Ad,Bd[:,None],Q,R)
 k=np.linalg.solve(R+Bd[None,:]@P@Bd[:,None],Bd[None,:]@P@Ad).ravel()
 rows.append(dict(R_multiplier=scale,gain=k.tolist(),rho_no_delay=radius(k,0),rho_fixed200=radius(k,20)))
report=dict(original_beta_at200=float(np.exp(-.8*195)),maximum_original_residual_authority_at200=float(.5*np.exp(-.8*195)),original_fixed200_radius=radius(K,20),candidate_decay_002_beta_at200=float(np.exp(-.002*195)),scope='Gain-feasibility diagnostic only, not a trained PPO result. Alternative gains are not inserted into deployed LQR: the original K remains fixed in all new LANS experiments.',gain_scan=rows)
(ROOT/'results/feasibility.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
