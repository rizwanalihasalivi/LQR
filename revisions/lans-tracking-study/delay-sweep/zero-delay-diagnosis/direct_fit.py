import ctypes
import json
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adjoint import HERE,FITS,DATA,VALID,CHECK,VEC,A,B,XS,BASE
from phase_fit import EQ

LIB=ctypes.CDLL(str(HERE/'forward.so'))
LIB.score.argtypes=[ctypes.c_int,*[VEC]*7,ctypes.c_int,ctypes.c_double,VEC]
LIB.score.restype=ctypes.c_double

def score(p,ds,phase,w=1.):
    return LIB.score(len(ds[-1]),p,A,B,*ds,np.empty(len(p)),1000,-w,phase)

def start(kind):
    if kind==0:
        return BASE.copy(),np.zeros((4,4))
    top=json.loads((HERE/'phase_validation.json').read_text())['selected']
    return np.array(top['p']),np.ascontiguousarray(EQ*top['spec'][0])

def fit(spec):
    kind,seed,w=spec
    p,m=start(kind);rng=np.random.default_rng(seed)
    scales=np.r_[[.022,.024,.024,.015],[.002,.005,.009,.012],
                 np.tile([.045,.035,.025,.025,.028,.040],14)]
    def metrics(v):
        return [score(v,ds,m) for ds in (VALID,CHECK)]
    best=(metrics(p),p.copy())
    training=score(p,DATA,m,w)
    for stage,decrease in enumerate([1.,.55,.30,.18,.10,.06,.03]):
        for _ in range(134):
            d=np.zeros(len(p))
            q=rng.random()
            if q<.41:
                idx=rng.choice(8,size=rng.integers(1,5),replace=False)
            elif q<.83:
                if rng.random()<.7:
                    idx=np.r_[rng.choice(8,size=2,replace=False),8+6*rng.integers(14)+np.arange(6)]
                else:
                    idx=rng.choice(np.arange(8,len(p)),size=5,replace=False)
            else:
                idx=np.r_[np.arange(8),8+6*rng.choice(14,size=3,replace=False)+5]
            d[idx]=rng.normal(size=len(idx))*scales[idx]*decrease
            forward=p+d;backward=p-d
            forward[12::6]=np.maximum(forward[12::6],.009)
            backward[12::6]=np.maximum(backward[12::6],.009)
            values=[score(x,DATA,m,w) for x in (forward,backward)]
            j=int(np.argmin(values))
            if values[j]<training:
                training=values[j]
                p=(forward,backward)[j]
        v=metrics(p)
        if sum(v)<sum(best[0]):best=(v,p.copy())
        print(spec,stage,training,v,best[0],flush=True)
    return dict(spec=spec,metrics=best[0],p=best[1].tolist(),phase=m.tolist())

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(fit,[(0,232,1.),(0,233,1.015),(1,234,1.),(1,235,.995)]))
    best=min(rows,key=lambda r:sum(r['metrics']))
    p=np.array(best['p']);m=np.array(best['phase']);f=FITS[0]
    f['weights']=[*(p[:4]/XS),*p[4:7]]
    f['extra_preview_weight']=float(p[7])
    f['hinges']=[dict(direction=v[:4].tolist(),threshold=v[4],weight=v[5],
                      preview_shift=(-v[:4]@m).tolist()) for v in p[8:].reshape(-1,6)]
    (HERE/'direct_fit.json').write_text(json.dumps(FITS,indent=2))
    (HERE/'direct_validation.json').write_text(json.dumps(dict(selected=best,trials=rows),indent=2))
    print('Selected',best['spec'],best['metrics'],flush=True)
