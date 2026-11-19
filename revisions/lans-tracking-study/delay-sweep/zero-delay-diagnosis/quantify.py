import csv,sys
from collections import defaultdict
from statistics import mean
from pathlib import Path

base=Path(__file__).resolve().parent
mode=sys.argv[1] if len(sys.argv)>1 else 'diagnose'
rows=list(csv.DictReader((base/f'{mode}_traces.csv').open()))
fields=['sse_0_0','sse_1_10','sse_11_30','sse_31_100','sse_101_1000']
groups=defaultdict(list)
for r in rows:
    groups[(r['group'],r['path'],r['physical_seed'],r['amplitude'],r['time_scale'],r['controller'])].append(r)
out=[]
for (*base_key,ctrl),m in groups.items():
    if ctrl!='MPC':continue
    x=groups[(*base_key,'LANS')]
    if not x: continue
    assert len(x)==3 and len(m)==1
    rmse_delta=mean(float(r['rmse_m']) for r in x)-float(m[0]['rmse_m'])
    diff={k:mean(float(r[k]) for r in x)-float(m[0][k]) for k in fields}
    s0=float(m[0]['rmse_m'])
    # Approximate phase contributions to total RMSE using each matched
    # trajectory's exact difference-of-squares denominator.
    contributions={f'{k}_delta_um':mean(
        (float(r[k])-float(m[0][k]))/1001/(float(r['rmse_m'])+s0)*1e6
        for r in x) for k in fields}
    group,path,seed,amp,ts=base_key
    out.append(dict(group=group,path=path,seed=seed,amplitude=amp,time_scale=ts,
                    lans_rmse_m=s0+rmse_delta,mpc_rmse_m=s0,delta_um=rmse_delta*1e6,
                    **contributions,**{f'{k}_delta':v for k,v in diff.items()}))
with (base/f'{mode}_paired_phases.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(out[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(out)
for group in sorted(set(x['group'] for x in out)):
    a=[x for x in out if x['group']==group]
    print(group,len(a),mean(x['delta_um'] for x in a),[round(mean(x[f'{k}_delta_um'] for x in a),5) for k in fields])
    for path in sorted(set(x['path'] for x in a)):
        r=[x for x in a if x['path']==path]
        print(group,path,round(mean(x['delta_um'] for x in r),6))
