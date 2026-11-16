"""Validation-only admissible-action diagnostic, never a new deployed controller."""
from pathlib import Path
import json
import numpy as np
from env import ImprovedEnv
from protocol import P
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
selection=json.loads((R/'selection.json').read_text());chosen=selection['ppo']
K=ImprovedEnv().K
rows=[]
for candidate in ['precision','tolerant']:
 c=P['candidates'][candidate];outside=total=0;by_delay={}
 for path in (R/'validation/traces').glob(chosen+'_*.npz'):
  with np.load(path) as z:a=z['trace']
  x=a[:,9:13];base=-x@K;gain=np.exp(-c['beta_decay']*np.maximum(0,a[:,7]-5))
  low=np.clip(base-gain*c['residual_scale'],-.5,.5)
  high=np.clip(base+gain*c['residual_scale'],-.5,.5)
  desired=a[:,3]
  bad=(desired<low-1e-7)|(desired>high+1e-7)
  outside+=int(bad.sum());total+=len(a)
  delay='fixed200' if '_fixed200_' in path.name else ('mixed' if '_mixed_' in path.name else 'none')
  q=by_delay.setdefault(delay,dict(outside=0,total=0));q['outside']+=int(bad.sum());q['total']+=len(a)
 rows.append(dict(candidate=candidate,reference_policy=chosen,steps=total,outside_admissible_actions=outside,fraction=outside/total,by_delay=by_delay,scope='Computed on validation PPO trajectories. This diagnoses bounded residual authority, not proof that no alternative LANS trajectory could improve tracking. It is not used to change selection or tune final test.'))
(R/'authority_diagnostic.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
