"""Constructed observability diagnostic, not a performance benchmark."""
from pathlib import Path
import json
import numpy as np
from env import ImprovedEnv
ROOT=Path(__file__).resolve().parents[1]
a=ImprovedEnv(kind='lans',delay_mode='fixed200')
b=ImprovedEnv(kind='lans',delay_mode='fixed200')
a.reset(seed=719);b.reset(seed=719)
for e,sign in [(a,1),(b,-1)]:
 e.x=np.zeros(4);e.k=100;e.last_delay=200.;e.history.extend([200.]*10)
 # Two admissible pending-command histories, held plant state identical.
 e.queue.q.extend((e.k+j,e.k+j-20,sign*.05) for j in range(20))
 e.queue.last_release=e.k+19
np.testing.assert_array_equal(a.obs(),b.obs())
pred=[]
for e in [a,b]:
 x=e.x.copy()
 for j,packet in enumerate(e.queue.q):
  _,_,command=packet
  k=e.k+j
  x=e.Ad@x+e.Bd*command+e.Gd*(e.ref[2,k]/e.vx)+e.Hd*(e.ref[3,k]/e.vx)
 pred.append(x)
targets=[float(np.clip(-a.K@x,-.5,.5)) for x in pred]
assert not np.allclose(pred[0],pred[1])
report=dict(policy_observations_identical=True,feature_count=11,pending_commands_per_case=20,pending_steering_values=[.05,-.05],predicted_state_at20ticks=[x.tolist() for x in pred],nominal_corrective_targets_at20ticks=targets,interpretation='Distinct pending queues are aliased by the current 11 observations, even though future plant states and corrective decisions differ. This demonstrates partial observability; it is not a proof that LANS cannot win a particular finite benchmark, and is not used to tune final test.',scope='Constructed diagnostic histories; not claimed to be observed benchmark trajectories or a new deployed predictor.')
(ROOT/'results/aliasing.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
