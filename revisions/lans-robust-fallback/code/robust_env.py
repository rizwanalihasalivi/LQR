import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(1,str(ROOT.parent/'lans-improvement/code'))
from env import ImprovedEnv
def robust_gain():return np.array(json.loads((ROOT/'results/robust_gain.json').read_text())['chosen']['K'])
class RobustEnv(ImprovedEnv):
 def __init__(self,kind='lans',**kwargs):
  super().__init__(kind=kind,residual_scale=.2 if kind=='lans' else .5,beta_decay=.8,**kwargs)
  self.K=robust_gain()
