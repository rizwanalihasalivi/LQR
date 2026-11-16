from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from robust_env import robust_gain
from model_audit import model
ROOT=Path(__file__).resolve().parents[1]
K=robust_gain();_,_,A,B,_,_=model('signed')
rows=[]
for d in range(21):
 if d==0:F=A-B[:,None]*K
 else:
  F=np.zeros((4+d,4+d));F[:4,:4]=A;F[:4,4]=B
  if d>1:F[4:-1,5:]=np.eye(d-1)
  F[-1,:4]=-K
 rho=float(max(abs(np.linalg.eigvals(F))));assert rho<1
 P=solve_discrete_lyapunov(F.T,np.eye(len(F)))
 error=float(np.max(abs(F.T@P@F-P+np.eye(len(F)))))
 assert np.linalg.eigvalsh(P).min()>0 and error<1e-6
 selector=np.r_[K,np.zeros(d)]
 c=.25/float(selector@np.linalg.solve(P,selector))
 rows.append(dict(delay_steps=d,delay_ms=10*d,spectral_radius=rho,lyapunov_residual=error,minimum_P_eigenvalue=float(np.linalg.eigvalsh(P).min()),unsaturated_invariant_c=c))
report=dict(gain=K.tolist(),cases=rows,scope='Local nominal linear fallback, fixed integer delay0..200ms, zero reference forcing and no PPO residual. Each delay has its own augmented-state Lyapunov certificate. This is not a common certificate for varying delay, not global saturated stability and not guaranteed tracking of forced slalom references.')
(ROOT/'results/certificate.json').write_text(json.dumps(report,indent=2));print('Certified fixed-delay local cases:',len(rows),'max residual:',max(r['lyapunov_residual'] for r in rows))
