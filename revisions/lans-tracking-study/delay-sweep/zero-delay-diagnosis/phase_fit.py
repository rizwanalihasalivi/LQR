import ctypes
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from adjoint import HERE,FITS,BASE,DATA,VALID,CHECK,VEC,A,B,XS
from cases import G,H,T

DLL=ctypes.CDLL(str(HERE/'phase.so'))
DLL.score.argtypes=[ctypes.c_int,*[VEC]*7,ctypes.c_int,ctypes.c_double,VEC]
DLL.score.restype=ctypes.c_double

def score(p,ds,phase,w=1.):
    g=np.zeros_like(p)
    v=DLL.score(len(ds[-1]),np.ascontiguousarray(p),A,B,*ds,g,1000,w,phase)
    return v,g

def periodic_pairs():
    pairs=[]
    h=.01*np.arange(1,5)
    for freq in np.r_[np.linspace(.48,1.,20),np.linspace(2.35,3.65,35),np.linspace(1.1,6.5,45)]:
        exp=np.exp(freq*.01j)
        mat=np.block([[exp*np.eye(4)-A,-B.reshape(4,1)],[np.eye(4)[:1],np.zeros((1,1))]])
        d=(G*(freq*1j)**2+H*(freq*1j)**3)/25
        target=np.linalg.solve(mat,np.r_[d,0.])
        q=((1+freq*1j*h-np.exp(freq*1j*h))/(10*h**2))@T
        weight=min(1.,(3.5/freq)**5)
        pairs.append((q*weight,target[:4]/XS*weight,target[4]*weight))
    return [np.concatenate([np.array([z[i] for z in pairs]).real,
                            np.array([z[i] for z in pairs]).imag]) for i in range(3)]

Q,X,U=periodic_pairs()
EQ = np.linalg.lstsq(Q,X,rcond=1e-9)[0].T.copy()
REGS=np.r_[[.1,.1,.1,.1],[.008,.02,.05,.04],
           np.tile([.04,.04,.04,.04,.02,.075],14)]

def fit(spec):
    blend,regularization,start_weight=spec
    M=np.ascontiguousarray(EQ*blend)
    p0=BASE.copy()
    dirs=BASE[8:].reshape(-1,6)[:,:4]
    knots=BASE[8:].reshape(-1,6)[:,4]
    weights=BASE[8:].reshape(-1,6)[:,5]
    z=(X-Q@M.T)@dirs.T
    correction=(np.maximum(z-knots,0)-np.maximum(-z-knots,0))@weights
    ff=np.linalg.lstsq(Q,U-X@p0[:4]-correction,rcond=1e-9)[0]
    old=(score(BASE,VALID,np.zeros((4,4)))[0],score(BASE,CHECK,np.zeros((4,4)))[0])
    best=None
    logs=[]
    for interpolation in [0.,.6,1.]:
        p0[4:8]=(1-interpolation)*BASE[4:8]+interpolation*ff
        initial=p0.copy()
        local=None
        calls=0
        lo=np.full(len(BASE),-2.5);hi=-lo
        lo[12::6]=np.maximum(lo[12::6],(.008-initial[12::6])/REGS[12::6])
        def metrics(p):
            return [score(p,ds,M)[0] for ds in [VALID,CHECK]]
        def record(p):
            nonlocal local
            v=metrics(p)
            if local is None or sum(v)<sum(local[0]):local=(v,p.copy())
        record(initial)
        def objective(v):
            p=initial+REGS*v
            f,g=score(p,DATA,M,start_weight)
            return f*1e6+regularization*np.mean(v*v),g*REGS*1e6+2*regularization*v/len(v)
        def callback(v):
            nonlocal calls
            calls+=1
            if calls%10==0:
                record(initial+REGS*v)
        sol=minimize(objective,np.zeros(len(BASE)),jac=True,method='L-BFGS-B',
                     bounds=list(zip(lo,hi)),callback=callback,
                     options=dict(maxiter=165,ftol=8e-14,gtol=2e-5,maxls=30))
        record(initial+REGS*sol.x)
        if best is None or sum(local[0])<sum(best[0]):best=local
        logs.append([interpolation,calls,local[0]])
        print(spec,interpolation,local[0],old,flush=True)
    return dict(spec=spec,metrics=best[0],p=best[1].tolist(),log=logs)

if __name__=='__main__':
    specs=[(.28,.01,1.),(.6,.01,1.),(1.,.007,1.),(1.,.007,1.03)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fit,specs))
    baseline=[score(BASE,s,np.zeros((4,4)))[0] for s in [VALID,CHECK]]
    best=min([*results,dict(spec=(0.,None,None),metrics=baseline,p=BASE.tolist())],
             key=lambda r:sum(r['metrics']))
    p=np.array(best['p'])
    f=FITS[0]
    f['weights']=[*(p[:4]/XS),*p[4:7]]
    f['extra_preview_weight']=float(p[7])
    M=best['spec'][0]*EQ
    f['hinges']=[dict(direction=w[:4].tolist(),threshold=w[4],weight=w[5],
                      preview_shift=(-w[:4]@M).tolist()) for w in p[8:].reshape(-1,6)]
    (HERE/'phase_fit.json').write_text(json.dumps(FITS,indent=2))
    (HERE/'phase_validation.json').write_text(json.dumps(dict(
        baseline=baseline,selected=best,trials=results),indent=2))
    print('Selected',best['spec'],best['metrics'],baseline,flush=True)
