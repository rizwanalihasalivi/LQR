from pathlib import Path
import csv,json
import numpy as np
from scipy.stats import t
from study_plan import PLAN
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
data=[]
for kind in ['lans','ppo','lqr','fast_lqr','mpc']:
 for seed in PLAN['training_seeds'] if kind in ['lans','ppo'] else [-1]:
  rr=list(csv.DictReader((R/'test'/f'{kind}_{seed}.csv').open()));assert len(rr)==140
  data+=rr
assert len(data)==1260
def write(name,rows):
 with (R/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('all_episodes.csv',data)
conditions=sorted(set((r['path'],r['delay'],r['mass']) for r in data));summary=[]
for c in conditions:
 group=[]
 for kind in ['lans','ppo','lqr','fast_lqr','mpc']:
  rr=[r for r in data if r['kind']==kind and (r['path'],r['delay'],r['mass'])==c]
  unit='train_seed' if kind in ['lans','ppo'] else 'physical_seed'
  units=sorted(set(r[unit] for r in rr));means=[np.mean([float(r['rmse_penalized']) for r in rr if r[unit]==u]) for u in units]
  m=float(np.mean(means));sd=float(np.std(means,ddof=1));hw=float(t.ppf(.975,len(means)-1)*sd/np.sqrt(len(means)))
  group.append(dict(kind=kind,path=c[0],delay=c[1],mass=c[2],mean_penalized_rmse=m,sd=sd,ci95_low=m-hw,ci95_high=m+hw,completed=sum(r['completed']=='True' for r in rr),episodes=len(rr),units=len(units),optimizer_failures=sum(int(r['optimizer_failures']) for r in rr)))
 for r in group:
  primary=[q for q in group if q['kind']!='fast_lqr']
  r['primary_four_rank']=1+sum(q['mean_penalized_rmse']<r['mean_penalized_rmse']-1e-12 for q in primary) if r['kind']!='fast_lqr' else ''
  r['five_controller_rank']=1+sum(q['mean_penalized_rmse']<r['mean_penalized_rmse']-1e-12 for q in group)
 summary+=group
write('summary.csv',summary)
overall=[]
for kind in ['lans','ppo','lqr','fast_lqr','mpc']:
 rr=[r for r in summary if r['kind']==kind]
 overall.append(dict(kind=kind,mean_primary_rank=float(np.mean([r['primary_four_rank'] for r in rr])) if kind!='fast_lqr' else '',mean_five_controller_rank=float(np.mean([r['five_controller_rank'] for r in rr])),primary_first_conditions=sum(r['primary_four_rank']==1 for r in rr),five_controller_first_conditions=sum(r['five_controller_rank']==1 for r in rr),completion=sum(r['completed'] for r in rr),episodes=sum(r['episodes'] for r in rr),mean_penalized_rmse=float(np.mean([r['mean_penalized_rmse'] for r in rr]))))
write('overall.csv',overall)
# Paired differences conditional on common physical cases, with policy seed as unit.
paired=[]
for other in ['lqr','fast_lqr','ppo','mpc']:
 for c in conditions:
  diffs=[]
  for seed in PLAN['training_seeds']:
   values=[]
   for ps in PLAN['test_seeds']:
    for amp,ts in PLAN['test_reference_scales']:
     def find(kind,s):return next(r for r in data if r['kind']==kind and int(r['train_seed'])==s and int(r['physical_seed'])==ps and float(r['amplitude'])==amp and float(r['time_scale'])==ts and (r['path'],r['delay'],r['mass'])==c)
     a=find('lans',seed);b=find(other,seed if other=='ppo' else -1);values.append(float(a['rmse_penalized'])-float(b['rmse_penalized']))
   diffs.append(np.mean(values))
  m=float(np.mean(diffs));h=float(t.ppf(.975,2)*np.std(diffs,ddof=1)/np.sqrt(3))
  paired.append(dict(comparison='lans_minus_'+other,path=c[0],delay=c[1],mass=c[2],mean_difference=m,ci95_low=m-h,ci95_high=m+h,training_seeds=3))
write('paired_differences.csv',paired)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
lans=next(r for r in overall if r['kind']=='lans');gain=json.loads((R/'robust_gain.json').read_text())['chosen']
headers=['Condition','LANS','PPO','Robust LQR','Fast LQR','MPC','LANS completed']
tab=[]
for c in conditions:
 vals={r['kind']:r for r in summary if (r['path'],r['delay'],r['mass'])==c}
 tab.append([f'{c[0]}/{c[1]}/{float(c[2]):g}',*[f"{vals[k]['mean_penalized_rmse']:.4f}" for k in ['lans','ppo','lqr','fast_lqr','mpc']],f"{vals['lans']['completed']}/{vals['lans']['episodes']}"])
report=f"""# LANS with a tuned delay-robust LQR fallback: prospective results

**Measured target outcome:** LANS ranks first or joint-first in **{lans['primary_first_conditions']}/14** primary four-controller conditions, and **{lans['five_controller_first_conditions']}/14** when the original fast LQR is also included. These are mean rankings, not universal or statistically proved superiority. Ties count as joint-first. The requested every-scenario target must not be asserted unless the actual count is14.

## Framework preservation

The physical dynamic bicycle model, mass/inertia/axle/stiffness magnitudes,100Hz exact ZOH, full steering-command FIFO delay,11 original observation channels and64x64ReLU networks are unchanged. No queue/history features, observer, predictor or additional controller branch is added.

The LQR component is retuned through Q/R, which changes controller parameters within the LQR family. This explicitly replaces the previously fixed aggressive fallback gain. The original exponential gain (5ms threshold,0.8/ms decay) and final±0.5rad actuator bound are retained. PPO remains a scalar additive residual. LANS residual scale is0.2rad; the PPO-only baseline retains0.5rad command range.

The four preview inputs are invertibly rescaled as in the preceding improvement study. Both policy families are initialized by encoding the previous fast-LQR prior in their existing ReLU weights. Their initial deterministic total commands match within the unclipped residual region, verified on representative small states. This is not a claim of identical clipped command/exploration distributions on all states. There is no new deployed teacher, planner or model-prediction module.

## Validation tuning and local certificate

Thirty-three LQR Q/R candidates are retained. Eligibility requires the nominal linear augmented system to be Schur stable for each fixed integer delay0..200ms in10ms steps. Eligible gains are selected by completion, then mean penalized RMSE, using validation seeds8000–8001, never this final test.

The chosen gain is {gain['K']}; Q={gain['Q']}, R={gain['R']}. Its maximum spectral radius over the declared grid is {gain['maximum_fixed_delay_radius']:.6f}. Separate augmented-state Lyapunov checks pass for all21 fixed-delay cases. This is a local zero-reference/no-residual certificate, not a common varying-delay certificate, not global saturated stability and not a guarantee of tracking every forced reference. Slalom tracking can fail despite homogeneous nominal linear stability.

## Training and untouched evaluation

Six final policies are retained: LANS/PPO each with seeds62,63,64 and200704steps per policy, total1,204,224steps. They receive the same explicit reward, curriculum, path/mass randomization and PPO settings. The training distribution and policy architecture are described in the preceding study and plan/source.

The final test was started only after all six checkpoints existed. It uses new initialization seeds18000–18004, held-out amplitude/time-scale combinations(0.9,1.05),(1.1,0.95), fourteen conditions and1,260 episodes. All reference families were trained; no unseen-family claim is made. Every seed, failure and raw trace is retained. Early failures use the same declared remaining-horizon penalty. SD/confidence intervals for learned controllers use3 training-seed means conditional on the physical test cases; baseline intervals use5 physical-seed means. These small-sample intervals do not establish universal dominance.

Both the matched robust-LQR component and original fast LQR are shown. MPC retains its previously declared20-step setting. Thus this study does not establish superiority over optimally tuned MPC or all possible baseline designs.

## All condition results: penalized RMSE in metres

{table(headers,tab)}

## Aggregate rankings

{table(['Controller','Four-controller mean rank','Five-controller mean rank','First/joint-first of14 (four)','Completed'],[[r['kind'],f"{r['mean_primary_rank']:.3f}" if r['kind']!='fast_lqr' else 'supplementary',f"{r['mean_five_controller_rank']:.3f}",r['primary_first_conditions'] if r['kind']!='fast_lqr' else 'supplementary',str(r['completion'])+'/'+str(r['episodes'])] for r in overall])}

The primary four are LANS, equally trained PPO, matched robust LQR and declared MPC. The supplementary five-controller ranking additionally includes the original aggressive LQR, preventing its stronger nominal tracking from being hidden.

## Causal interpretation and limits

At200ms the original exponential gain is1.78e-68. The neural residual therefore has no material authority, and LANS matches the retuned LQR fallback. Any improvement in these fixed200ms cases comes from the changed LQR Q/R, not learned delay suppression or the reward. Policy-seed repetitions at this delay are identical conditional on the same physical initialization; they are not30 independent physical experiments. Low-delay improvements relative to robust LQR can be attributed to the residual intervention, but beating the robust component does not automatically mean beating the original fast LQR or MPC.

The first improvement campaign and all original adverse results are preserved. This study must not be presented as an unchanged-parameter reproduction of the earlier manuscript, a certified full-LANS stability proof, a vehicle safety result, an energy/timing superiority claim or guaranteed rank1 across all scenarios. New matched CPU timing is not measured here.

"""
(ROOT/'Robust_Fallback_Report.md').write_text(report)
print(json.dumps(dict(lans=lans,overall=overall),indent=2))
