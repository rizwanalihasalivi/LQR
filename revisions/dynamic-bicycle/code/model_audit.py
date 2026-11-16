"""Literal manuscript Eq. (9), Table I and Bryson limits, versus signed-stiffness convention."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm, solve_continuous_are, solve_discrete_are
ROOT=Path(__file__).resolve().parents[1]
PARAMS=dict(m=1650.,Iz=4224.,lf=1.2,lr=1.4,Cf=80000.,Cr=80000.,vx=25.,dt=.01,delta_max=.5)
Q=np.diag([100.,4.,100.,100.]);R=np.array([[4.]])
def model(convention='printed',mass=1650.,vx=25.):
 p=PARAMS;m=mass;I=p['Iz'];a=p['lf'];b=p['lr'];f=p['Cf'];r=p['Cr']
 if convention=='signed': f=-f;r=-r
 A=np.array([[0,1,0,0],[0,2*(f+r)/(m*vx),-2*(f+r)/m,2*(a*f-b*r)/(m*vx)],[0,0,0,1],[0,2*(a*f-b*r)/(I*vx),-2*(a*f-b*r)/I,2*(a*a*f+b*b*r)/(I*vx)]],float)
 B=np.array([[0],[-2*f/m],[0],[-2*a*f/I]])
 # Curvature is an exogenous reference input, not a change to A/B.
 # From Eq (2),(8): g*r_ref + h*r_ref_dot.
 G=np.array([0,2*(a*f-b*r)/(m*vx)-vx,0,2*(a*a*f+b*b*r)/(I*vx)])
 H=np.array([0,0,0,-1.])
 M=np.zeros((7,7));M[:4,:4]=A;M[:4,4:5]=B;M[:4,5]=G;M[:4,6]=H
 D=expm(M*p['dt'])
 return A,B,D[:4,:4],D[:4,4],D[:4,5],D[:4,6]
def gains(convention='printed',vx=25.):
 A,B,Ad,Bd,_,_=model(convention,vx=vx)
 P=solve_discrete_are(Ad,Bd[:,None],Q,R)
 K=np.linalg.solve(R+Bd[None,:]@P@Bd[:,None],Bd[None,:]@P@Ad).ravel()
 return K,P
if __name__=='__main__':
 out={'parameters':PARAMS,'Q':Q.tolist(),'R':R.tolist(),'cases':[]}
 for c in ['printed','signed']:
  A,B,Ad,Bd,G,H=model(c);P=solve_continuous_are(A,B,Q,R);K=np.linalg.solve(R,B.T@P)
  kd,pd=gains(c)
  out['cases'].append(dict(convention=c,A=A.tolist(),B=B.tolist(),continuous_K=K.tolist(),discrete_K=kd.tolist(),open_loop_real_parts=np.linalg.eigvals(A).real.tolist(),continuous_closed_real_parts=np.linalg.eigvals(A-B@K).real.tolist(),care_gain_zoh_radius=float(max(abs(np.linalg.eigvals(Ad-Bd[:,None]@K)))),dare_gain_zoh_radius=float(max(abs(np.linalg.eigvals(Ad-Bd[:,None]*kd)))),P_min_eigenvalue=float(np.linalg.eigvalsh(pd).min())))
 (ROOT/'results/model_audit.json').write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2))
