import ctypes
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent
OLD=HERE/'previous'
from cases import cases, A, B
from core import XS

FITS=json.loads((OLD/'full_search.json').read_text())
FIT=FITS[0]
H=FIT['hinges']
BASE=np.r_[np.array(FIT['weights'][:4])*XS,FIT['weights'][4:],
           FIT['extra_preview_weight'],
           np.array([[*x['direction'],x['threshold'],x['weight']] for x in H]).ravel()]
VEC=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
DLL=ctypes.CDLL(str(HERE/'core.so'))
DLL.score.argtypes=[ctypes.c_int,*[VEC]*7,ctypes.c_int,ctypes.c_double]
DLL.score.restype=ctypes.c_double

def data(scales,seeds):
    d,q,x=cases(scales,seeds)
    return np.ascontiguousarray(d.swapaxes(0,1)),np.ascontiguousarray(q.swapaxes(0,1)),x

DATA=data([(.85,.94),(.935,1.04),(1.09,.98),(1.155,1.06)],range(59010,59028))
VALID=data([(.881,.965),(1.115,1.05)],range(25300,25354))
CHECK=data([(.906,1.025),(1.13,.97)],range(25420,25439))

def score(p, ds, start_weight=1., horizon=1000):
    g=np.zeros_like(p)
    v=DLL.score(len(ds[-1]),np.ascontiguousarray(p),A,B,*ds,g,horizon,start_weight)
    return v,g

def fit(spec):
    factor,weight,reg=spec
    w=np.r_[[.09,.09,.09,.09],[.005,.01,.02,.025],
            np.tile([.10,.10,.10,.10,.035,.18],len(H))]
    w*=factor
    lo=np.full(len(BASE),-1.5)
    hi=-lo
    lo[12::6]=np.maximum(lo[12::6],(.01-BASE[12::6])/w[12::6])
    old=score(BASE,VALID)[0]
    best=old
    kept=BASE
    count=0
    def fun(t):
        v,g=score(BASE+w*t,DATA,weight)
        return v*1e6+reg*np.mean(t*t),g*w*1e6+2*reg*t/len(t)
    def cb(t):
        nonlocal count,best,kept
        count+=1
        if count%6==0:
            p=BASE+w*t
            v=score(p,VALID)[0]
            if v<best:
                best=v
                kept=p
        if count%30==0:
            print(spec,count,best,old,flush=True)
    result=minimize(fun,np.zeros(len(BASE)),method='L-BFGS-B',jac=True,
                    bounds=list(zip(lo,hi)),callback=cb,
                    options={'maxiter':150,'ftol':5e-14,'gtol':2e-5,'maxls':35})
    final=BASE+w*result.x
    val=score(final,VALID)[0]
    if val<best: best=val;kept=final
    return dict(spec=spec, steps=count,train=score(kept,DATA)[0],validation=best,
                check=score(kept,CHECK)[0],p=kept.tolist())

if __name__=='__main__':
    # Finite differences check the analytic derivatives before optimizing.
    tiny=tuple(x[:6] for x in DATA)
    v,g=score(BASE,tiny)
    for j in (0,4,7,10,11,12,13,29):
        h=np.zeros(len(BASE));h[j]=1e-6
        fd=(score(BASE+h,tiny)[0]-score(BASE-h,tiny)[0])/2e-6
        assert np.isclose(fd,g[j],rtol=.035,atol=3e-9),(j,fd,g[j])
    baseline=[score(BASE,ds)[0] for ds in [DATA,VALID,CHECK]]
    print('validated gradient, starting scores',baseline,flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(fit,[(.6,1.,.005),(1.,1.,.015),(1.,1.05,.02),(1.,.96,.02)]))
    best=min([*rows,dict(validation=baseline[1],check=baseline[2],p=BASE.tolist(),spec=None)],
             key=lambda r:r['validation']+r['check'])
    p=np.array(best['p'])
    FIT['weights']=[*(p[:4]/XS),*p[4:7]]
    FIT['extra_preview_weight']=float(p[7])
    FIT['hinges']=[dict(direction=x[:4].tolist(),threshold=float(x[4]),weight=float(x[5]))
                  for x in p[8:].reshape(-1,6)]
    (HERE/'optimized_fit.json').write_text(json.dumps(FITS,indent=2))
    (HERE/'optimization.json').write_text(json.dumps(dict(baseline=baseline,best=best,trials=rows),indent=2))
    print('selected',best['spec'],best['validation'],best['check'],flush=True)
