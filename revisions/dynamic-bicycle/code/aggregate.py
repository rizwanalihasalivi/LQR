"""Aggregate only fresh campaign outputs. Training-seed means are the RL inference unit."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.stats import t as student
ROOT=Path(__file__).resolve().parents[1]
def write(name,rows):
 with (ROOT/'results'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[]
for seed in [-1,42,43,44]:
 p=ROOT/'results'/f'episodes_{seed}.csv';r=list(csv.DictReader(p.open()));assert len(r)==(145 if seed==-1 else 360), (seed,len(r))
 rows.extend(r)
write('episodes_all.csv',rows)
keys=sorted(set((r['kind'],r['path'],r['delay'],r['scope'],r['mass']) for r in rows))
summary=[]
for kind,path,delay,scope,mass in keys:
 group=[r for r in rows if (r['kind'],r['path'],r['delay'],r['scope'],r['mass'])==(kind,path,delay,scope,mass)]
 units=sorted(set(r['train_seed'] if kind not in ['lqr','mpc'] else r['physical_seed'] for r in group))
 means=[np.mean([float(r['rmse_penalized']) for r in group if (r['train_seed'] if kind not in ['lqr','mpc'] else r['physical_seed'])==u]) for u in units]
 m=float(np.mean(means));sd=float(np.std(means,ddof=1)) if len(means)>1 else 0.;hw=float(student.ppf(.975,len(means)-1)*sd/np.sqrt(len(means))) if len(means)>1 else 0.
 summary.append(dict(kind=kind,path=path,delay=delay,scope=scope,mass=mass,episodes=len(group),completed=sum(r['completed']=='True' for r in group),independent_units=len(means),mean_penalized_rmse=m,sd_across_units=sd,ci95_low=m-hw,ci95_high=m+hw,mean_observed_rmse=float(np.mean([float(r['rmse_observed']) for r in group])),mean_residual_rms=float(np.mean([float(r['rms_residual']) for r in group])),mean_saturation=float(np.mean([float(r['saturated_fraction']) for r in group])),mpc_optimizer_failures=sum(int(r['mpc_optimizer_failures']) for r in group)))
write('summary.csv',summary)
# Paired mean differences, full vs others, within seed/physical case.
paired=[]
for other in ['no_reward','no_gain','no_lqr','ppo','lqr','mpc']:
 for path in ['dlc','step','sine','slalom']:
  for delay in ['none','mixed','fixed200']:
   diffs=[]
   for seed in [42,43,44]:
    delta=[]
    for ps in range(3200,3205):
     def find(kind,ts):return next(r for r in rows if r['kind']==kind and int(r['train_seed'])==ts and int(r['physical_seed'])==ps and r['path']==path and r['delay']==delay and r['scope']=='full' and float(r['mass'])==1650)
     a=find('full',seed);b=find(other,seed if other not in ['lqr','mpc'] else -1)
     delta.append(float(a['rmse_penalized'])-float(b['rmse_penalized']))
    diffs.append(np.mean(delta))
   mean=float(np.mean(diffs));sd=float(np.std(diffs,ddof=1));hw=float(student.ppf(.975,2)*sd/np.sqrt(3))
   paired.append(dict(comparison='full_minus_'+other,path=path,delay=delay,n_training_seeds=3,mean_difference_m=mean,ci95_low=mean-hw,ci95_high=mean+hw,seed42_difference=float(diffs[0]),seed43_difference=float(diffs[1]),seed44_difference=float(diffs[2])))
write('paired_differences.csv',paired)
print('episodes',len(rows),'summary rows',len(summary),'paired rows',len(paired))
