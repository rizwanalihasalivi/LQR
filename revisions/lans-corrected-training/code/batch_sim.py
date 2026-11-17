from pathlib import Path
import sys
import numpy as np
from core import teacher,model,KR,beta,XS
from dynamic_env import reference
def build_cases(seeds,scales=[(1.,1.)]):
 cases=[]
 for amp,ts in scales:
  for path in ['dlc','step','sine','slalom']:
   for delay in ['none','mixed','fixed200']:
    for seed in seeds:cases.append(dict(path=path,delay=delay,mass=1650.,amplitude=amp,time_scale=ts,seed=seed))
  for mass in [1100.,2200.]:
   for seed in seeds:cases.append(dict(path='dlc',delay='none',mass=mass,amplitude=amp,time_scale=ts,seed=seed))
 return cases
def prepare(cases):
 n=len(cases);refs=[];As=[];Bs=[];Gs=[];Hs=[];initial=[];delays=[];applied=[]
 for c in cases:
  rng=np.random.default_rng(c['seed']);initial.append(rng.normal(0,[.05,.02,.005,.002]))
  arr=np.asarray(reference(c['path'],np.arange(1006)*.01*c['time_scale']))
  arr*=c['amplitude']*np.array([1,c['time_scale'],c['time_scale']**2,c['time_scale']**3])[:,None];refs.append(arr)
  _,_,a,b,g,hh=model('signed',c['mass']);As.append(a);Bs.append(b);Gs.append(g);Hs.append(hh)
  d=np.zeros(1001)
  if c['delay']=='mixed':
   mask=rng.random(1001)<.1;d[mask]=rng.uniform(5,130,mask.sum())
  elif c['delay']=='fixed200':d[:]=200.
  delays.append(d)
  releases=np.maximum.accumulate(np.arange(1000)+np.ceil(d[:1000]/10).astype(int))
  applied.append(np.searchsorted(releases,np.arange(1000),side='right')-1)
 return dict(cases=cases,A=np.array(As),B=np.array(Bs),G=np.array(Gs),H=np.array(Hs),ref=np.array(refs),initial=np.array(initial),delays=np.array(delays),applied=np.array(applied))
def rollout(data,params,collect=False):
 n=len(data['cases']);x=data['initial'].copy();initial=x[:,0].copy();commands=np.zeros((1000,n));errors=np.zeros((1000,n));first=np.full(n,1000);active=np.ones(n,bool);corpus=[];targets=[]
 ref=data['ref'];delays=data['delays'];h=.01*np.arange(1,5);indices=np.arange(n)
 for k in range(1000):
  preview=(ref[:,0,k,None]+x[:,0,None]+(ref[:,1,k,None]+25*x[:,2,None])*h-ref[:,0,k+np.arange(1,5)])
  q=(ref[:,0,k,None]+ref[:,1,k,None]*h-ref[:,0,k+np.arange(1,5)])/(10*h*h)
  obs=np.column_stack([x/XS,np.ones(n),q,delays[:,k-1]/200 if k else np.zeros(n),delays[:,max(0,k-10):k].mean(1)/200 if k else np.zeros(n)]).astype(np.float32)
  u=teacher(obs,params)
  res=np.clip((u+x@KR)/beta(np.minimum(200,np.maximum(obs[:,9]*200,params['proxy']*obs[:,10]*200))),-1,1)
  cmd=np.clip(-x@KR+beta(delays[:,k])*res,-.5,.5);commands[k]=cmd
  source=data['applied'][:,k];applied=np.where(source>=0,commands[np.maximum(0,source),indices],0.)
  nextx=np.einsum('nij,nj->ni',data['A'],x)+data['B']*applied[:,None]+data['G']*(ref[:,2,k]/25)[:,None]+data['H']*(ref[:,3,k]/25)[:,None]
  x=np.where(active[:,None],nextx,x);errors[k]=x[:,0]
  fail=active&((abs(x[:,0])>2.5)|(abs(x[:,2])>.5)|~np.isfinite(x).all(1))
  first[fail]=k+1;active[fail]=False
  if collect and k%2==0:
   keep=np.flatnonzero(active)
   corpus.append(obs[keep]);targets.append(np.clip(u[keep],-.5,.5))
 sse=initial**2+sum(np.where(np.arange(1000)[:,None]<first[None,:],errors**2,0.))
 pen=np.sqrt((sse+(1000-first)*np.maximum(2.5,abs(x[:,0]))**2)/1001)
 if collect:return pen,active,np.concatenate(corpus),np.concatenate(targets)
 return pen,active
