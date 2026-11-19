import itertools

import numpy as np

from candidate import ORIGINAL_ENV as NetworkEnv
from dynamic_env import plant
from features import T

A, B, G, H = map(np.ascontiguousarray, plant())
B = B.reshape(4)

TIME = np.arange(1000)
HORIZON = .01*np.arange(1,5)

def cases(scales,seeds):
    disturbance, preview, initial = [],[],[]
    for (amplitude,time_scale),path,seed in itertools.product(scales,
            ['dlc','step','sine','slalom'],seeds):
        env = NetworkEnv(kind='lans',path=path,amplitude=amplitude,time_scale=time_scale)
        env.reset(seed=seed)
        r = env.ref
        initial.append(env.x.copy())
        disturbance.append((r[2,:1000,None]*G+r[3,:1000,None]*H)/25)
        q = (r[0,:1000,None]+r[1,:1000,None]*HORIZON-
             r[0,TIME[:,None]+np.arange(1,5)])/(10*HORIZON**2)
        preview.append(q@T)
    return tuple(np.ascontiguousarray(x) for x in (np.swapaxes(disturbance,0,1),
                                                   np.swapaxes(preview,0,1),initial))
