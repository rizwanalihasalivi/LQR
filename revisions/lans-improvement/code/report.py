from pathlib import Path
import csv,json,hashlib
import numpy as np
from scipy.stats import t
from protocol import P
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
selection=json.loads((R/'selection.json').read_text());names=[selection['lans'],selection['ppo'],'lqr','mpc']
rows=[]
for name in names:
 for seed in ([-1] if name in ['lqr','mpc'] else P['training_seeds']):
  data=list(csv.DictReader((R/'test'/f'{name}_{seed}.csv').open()));assert len(data)==140
  rows+=data
assert len(rows)==1120
def write(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(R/'test/all_episodes.csv',rows)
conditions=sorted(set((r['path'],r['delay'],r['mass']) for r in rows))
summaries=[];rank_sums={n:0 for n in names};wins={n:0 for n in names}
for condition in conditions:
 group=[]
 for name in names:
  rr=[r for r in rows if r['candidate']==name and (r['path'],r['delay'],r['mass'])==condition]
  unit='physical_seed' if name in ['lqr','mpc'] else 'train_seed'
  units=sorted(set(r[unit] for r in rr));means=[np.mean([float(r['rmse_penalized']) for r in rr if r[unit]==u]) for u in units]
  mean=float(np.mean(means));sd=float(np.std(means,ddof=1));hw=float(t.ppf(.975,len(means)-1)*sd/np.sqrt(len(means)))
  group.append(dict(controller='LANS' if name==selection['lans'] else ('PPO' if name==selection['ppo'] else name.upper()),candidate=name,path=condition[0],delay=condition[1],mass=condition[2],mean_penalized_rmse=mean,sd_across_units=sd,ci95_low=mean-hw,ci95_high=mean+hw,independent_units=len(units),completed=sum(r['completed']=='True' for r in rr),episodes=len(rr),optimizer_failures=sum(int(r['optimizer_failures']) for r in rr)))
 for r in group:
  rank=1+sum(g['mean_penalized_rmse']<r['mean_penalized_rmse']-1e-12 for g in group);r['rank']=rank
  rank_sums[r['candidate']]+=rank;wins[r['candidate']]+=rank==1
 summaries+=group
write(R/'test/summary.csv',summaries)
overall=[]
for name in names:
 rr=[r for r in rows if r['candidate']==name]
 overall.append(dict(candidate=name,average_condition_rank=rank_sums[name]/len(conditions),first_place_conditions=int(wins[name]),conditions=len(conditions),completion_count=sum(r['completed']=='True' for r in rr),episodes=len(rr),mean_penalized_rmse=float(np.mean([float(r['rmse_penalized']) for r in rr]))))
overall.sort(key=lambda r:r['average_condition_rank']);write(R/'test/overall.csv',overall)
# Matched comparisons: average within each training seed, not treating episodes as independent policies.
paired=[]
for other in [selection['ppo'],'lqr','mpc']:
 for c in conditions:
  values=[]
  for seed in P['training_seeds']:
   diffs=[]
   for ps in P['test_seeds']:
    for amp,ts in [(.9,1.1),(1.1,.9)]:
     def find(n,s):return next(r for r in rows if r['candidate']==n and int(r['train_seed'])==s and int(r['physical_seed'])==ps and float(r['amplitude'])==amp and float(r['time_scale'])==ts and (r['path'],r['delay'],r['mass'])==c)
     a=find(selection['lans'],seed);b=find(other,seed if other not in ['lqr','mpc'] else -1)
     diffs.append(float(a['rmse_penalized'])-float(b['rmse_penalized']))
   values.append(np.mean(diffs))
  m=float(np.mean(values));h=float(t.ppf(.975,2)*np.std(values,ddof=1)/np.sqrt(3))
  paired.append(dict(comparison='LANS_minus_'+other,path=c[0],delay=c[1],mass=c[2],mean_difference=m,ci95_low=m-h,ci95_high=m+h,n_training_seeds=3))
write(R/'test/paired_differences.csv',paired)
def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in data])
overall_table=table(['Controller configuration','Mean rank','First-place conditions','Completed / episodes'],[[r['candidate'],f"{r['average_condition_rank']:.3f}",f"{r['first_place_conditions']}/14",f"{r['completion_count']}/{r['episodes']}"] for r in overall])
ranking=[]
for c in conditions:
 group=sorted([r for r in summaries if (r['path'],r['delay'],r['mass'])==c],key=lambda r:r['rank'])
 ranking.append([f'{c[0]} / {c[1]} / {float(c[2]):g}kg','; '.join(f"{r['controller']} rank {r['rank']} ({r['mean_penalized_rmse']:.4f}m, {r['completed']}/{r['episodes']})" for r in group)])
first=sum(r['rank']==1 for r in summaries if r['candidate']==selection['lans'])
report=f"""# Prospective LANS improvement study: measured outcome

The requested target was LANS first in every scenario. **The final test places the selected LANS first in {first}/14 conditions.** A first-place mean in a finite test is not universal or statistically established dominance. All failures and all trained seeds are retained.

## Preserved framework and changed training

The same dynamic bicycle plant, physical parameter magnitudes, sampled LQR gain, full-command queue, LQR + PPO + exponential latency gain + saturation, 11 causal features and 64x64 ReLU networks are retained. No predictor, observer, planner, oracle or extra vehicle-state input is added to LANS. Observation scaling is invertible; preview curvature information is exposed from the existing four inputs. Candidate reward/exploration, residual scale, gain decay and training distribution changes are explicit in the frozen protocol.

There are 12 new policies: 4 configurations × 3 independent seeds, each with 200,704 steps (2,408,448 steps total). PPO receives the same two-configuration training-search budget. This is a finite-budget study, not evidence of converged optimal policies.

Validation uses seeds8000–8001 and nominal reference parameters. A single whole configuration per family is selected by completion rate, then penalized RMSE, across all seeds/conditions. It is never switched by scenario. Selected LANS: **{selection['lans']}**; selected PPO: **{selection['ppo']}**.

The held-out test uses five new initialization seeds12000–12004 and two held-out amplitude/time-scale combinations(0.9,1.1),(1.1,0.9). These reference families were included in training: this is new-parameter testing, not unseen-family zero-shot generalization. There are1,120 final test episodes.

## Overall outcome

{overall_table}

Every condition is equally weighted in mean rank. Rank is based on mean failure-penalized RMSE, with ties sharing a rank. A failed case can receive a numeric rank from the declared failure penalty; that rank does not mean successful tracking. Completion counts are therefore displayed.

## Each condition

{table(['Condition','Results: rank, penalized RMSE, completion'],ranking)}

## Limits and interpretation

The old and new campaigns use different test reference parameters, seeds and training coverage. Do not treat a lower number here than in the old paper as a matched improvement estimate. New LANS and new PPO/LQR/MPC are compared within this test. The LQR and MPC settings are the previously declared fixed configurations; the results do not establish superiority over all possible baseline tuning. MPC has20-step preview while policy inputs retain four previews.

At200ms the original exponential gain is1.78e-68. Bounded PPO actions therefore cannot materially repair the original delayed LQR, and opposite extreme actions produce identical simulated trajectories in the diagnostic. The tolerant candidate openly changes the decay parameter while retaining the same exponential family. This does not establish arbitrary-delay stability.

Three independent training seeds give wide uncertainty. Seed-level paired confidence intervals are supplied in paired_differences.csv; a numerical rank alone does not establish statistical superiority. No energy, compute-speed, physical vehicle safety, nonlinear dynamics or global stability superiority is claimed.

Original manuscript/results/checkpoints remain unchanged. This experimental report must not be described as a successful proof of the requested target unless the actual result supports it.
"""
(ROOT/'Improvement_Report.md').write_text(report)
print(json.dumps(dict(selected_lans=selection['lans'],selected_ppo=selection['ppo'],lans_first_conditions=first,overall=overall),indent=2))
