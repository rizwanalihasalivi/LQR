from pathlib import Path
import json
import numpy as np
from env import ImprovedEnv
from dynamic_env import DynamicEnv
ROOT=Path(__file__).resolve().parents[1];checks={}
for delay in ['none','mixed','fixed200']:
 a=DynamicEnv(variant='lans',delay_mode=delay)
 b=ImprovedEnv(kind='lans',residual_scale=.02,beta_decay=.8,delay_mode=delay)
 a.reset(seed=927);b.reset(seed=927)
 for tick in range(1000):
  _,_,ta,tra,ia=a.step([0]);_,_,tb,trb,ib=b.step([0])
  np.testing.assert_array_equal(a.x,b.x);assert ta==tb and tra==trb and ia['applied']==ib['applied']
  if ta or tra:break
 checks['unchanged_plant_zero_residual_'+delay]=tick+1
# Original gain at 200 ms makes all bounded residual interventions numerically inert.
a=DynamicEnv(variant='lans',delay_mode='fixed200')
b=DynamicEnv(variant='lans',delay_mode='fixed200')
a.reset(seed=927);b.reset(seed=927)
for tick in range(1000):
 _,_,ta,tra,ia=a.step([-1]);_,_,tb,trb,ib=b.step([1])
 np.testing.assert_array_equal(a.x,b.x);assert ia['u']==ib['u']
 if ta or tra:break
checks['original_fixed200_extreme_actions_identical_steps']=tick+1
# Same 11 observations with an invertible feature transform, not additional state.
e=ImprovedEnv(delay_mode='none');raw_initial,_=e.reset(seed=471)
raw=DynamicEnv.obs(e).astype(float);new=e.obs().astype(float)
x=new[:4]*np.array([.1,.5,.05,.2])
recovered=new[5:9]*10*(.01*np.arange(1,5))**2+x[0]+e.vx*x[2]*(.01*np.arange(1,5))
assert np.max(abs(recovered-raw[5:9]))<1e-7
checks['preview_transform_reconstruction_error']=float(np.max(abs(recovered-raw[5:9])))
assert len(new)==11;checks['policy_feature_count']=11
(ROOT/'results/checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
