from pathlib import Path
import csv,json
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
def controller_spec(strong):
 if strong not in ['limited_ppo','rl_ppo','original_ppo']:
  raise ValueError(f'Unsupported selected PPO reference: {strong}')
 spec={'limited_lans':[72,73,74],'limited_ppo':[72,73,74],'bc_lans':[72,73,74],'bc_ppo':[72,73,74],'original_ppo':[42,43,44],'previous_lans':[62,63,64],'robust_lqr':[-1],'fast_lqr':[-1],'mpc':[-1],'matched_mpc':[-1]}
 if strong not in spec:spec[strong]=[72,73,74]
 return spec

def main():
 selection=json.loads((R/'validation_selection.json').read_text());strong=selection['ppo_reference']
 spec=controller_spec(strong)
 rows=[]
 for name,seeds in spec.items():
  for seed in seeds:
   with (R/'test'/f'{name}_{seed}.csv').open() as f:rr=list(csv.DictReader(f))
   assert len(rr)==140
   rows+=rr
 assert len(rows)==140*sum(map(len,spec.values()))
 def write(name,data):
  with (R/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 write('episodes_all.csv',rows)
 conditions=sorted(set((r['path'],r['delay'],r['mass']) for r in rows));summary=[]
 for c in conditions:
  for name in spec:
   rr=[r for r in rows if r['kind']==name and (r['path'],r['delay'],r['mass'])==c]
   unit='train_seed' if len(spec[name])==3 else 'physical_seed'
   means=[np.mean([float(r['rmse_penalized']) for r in rr if r[unit]==u]) for u in sorted(set(r[unit] for r in rr))]
   mean=float(np.mean(means));sd=float(np.std(means,ddof=1));hw=float(t.ppf(.975,len(means)-1)*sd/np.sqrt(len(means)))
   summary.append(dict(kind=name,path=c[0],delay=c[1],mass=c[2],mean_penalized_rmse=mean,sd=sd,ci95_low=mean-hw,ci95_high=mean+hw,completed=sum(r['completed']=='True' for r in rr),episodes=len(rr),unit_count=len(means)))
 write('summary.csv',summary)
 groups={'original':['limited_lans','original_ppo','fast_lqr','mpc'],'equal_pretraining':['limited_lans',strong,'fast_lqr','mpc'],'objective_matched':['limited_lans',strong,'fast_lqr','matched_mpc'],'all_references':list(spec)}
 ranktables={};overall={}
 for label,names in groups.items():
  rankrows=[];total={n:0 for n in names};wins={n:0 for n in names}
  for c in conditions:
   d={r['kind']:r for r in summary if (r['path'],r['delay'],r['mass'])==c}
   row=dict(path=c[0],delay=c[1],mass=c[2])
   for name in names:
    rank=1+sum(d[n]['mean_penalized_rmse']<d[name]['mean_penalized_rmse']-1e-12 for n in names)
    row[name+'_rank']=rank;row[name+'_rmse']=d[name]['mean_penalized_rmse'];row[name+'_completed']=d[name]['completed'];row[name+'_episodes']=d[name]['episodes']
    total[name]+=rank;wins[name]+=rank==1
   rankrows.append(row)
  write('rank_'+label+'.csv',rankrows);ranktables[label]=rankrows
  overall[label]={n:dict(mean_rank=total[n]/14,first_conditions=wins[n],conditions=14) for n in names}
 (R/'ranking_summary.json').write_text(json.dumps(overall,indent=2))
 # Paired differences conditional on common physical/reference cases; learned seed is inference unit.
 paired=[]
 for other in ['limited_ppo','bc_lans','previous_lans','fast_lqr','mpc','matched_mpc']:
  for c in conditions:
   values=[]
   for i,seed in enumerate([72,73,74]):
    oldseed=spec[other][i] if len(spec[other])==3 else -1;ds=[]
    aa=[r for r in rows if r['kind']=='limited_lans' and int(r['train_seed'])==seed and (r['path'],r['delay'],r['mass'])==c]
    bb=[r for r in rows if r['kind']==other and int(r['train_seed'])==oldseed and (r['path'],r['delay'],r['mass'])==c]
    for a in aa:
     b=next(r for r in bb if all(r[k]==a[k] for k in ['physical_seed','amplitude','time_scale']))
     ds.append(float(a['rmse_penalized'])-float(b['rmse_penalized']))
    values.append(np.mean(ds))
   mean=float(np.mean(values));hw=float(t.ppf(.975,2)*np.std(values,ddof=1)/np.sqrt(3))
   paired.append(dict(comparison='limited_lans_minus_'+other,path=c[0],delay=c[1],mass=c[2],mean_difference=mean,ci95_low=mean-hw,ci95_high=mean+hw,unit_count=3,caution='Exploratory conditional seed intervals; different previous training seeds are not matched random training experiments.'))
 write('paired_differences.csv',paired)
 def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in data])
 totals={}
 for name in spec:
  rr=[r for r in rows if r['kind']==name];totals[name]=dict(completed=sum(r['completed']=='True' for r in rr),episodes=len(rr),mean_rmse=float(np.mean([float(r['rmse_penalized']) for r in rr])))
 (R/'aggregate.json').write_text(json.dumps(totals,indent=2))
 tab=[]
 for c in conditions:
  d={r['kind']:r for r in summary if (r['path'],r['delay'],r['mass'])==c}
  tab.append([f'{c[0]}/{c[1]}/{float(c[2]):g}',*[f"{d[n]['mean_penalized_rmse']:.5f}" for n in ['limited_lans','original_ppo','limited_ppo','mpc','matched_mpc']],f"{d['limited_lans']['completed']}/{d['limited_lans']['episodes']}"])
 ranktable=[]
 for label,values in overall.items():
  r=values['limited_lans'];ranktable.append([label,r['first_conditions'],f"{r['mean_rank']:.3f}"])
 report=f"""# LANS precision/tracking study: same deployed framework, new training

## Result and comparison boundary

Final PPO-trained LANS completed **{totals['limited_lans']['completed']}/{totals['limited_lans']['episodes']}** fresh test episodes. Against the original stronger PPO,original fast LQR and declared MPC it ranks first in **{overall['original']['limited_lans']['first_conditions']}/14** conditions by mean failure-penalized RMSE.

Stronger comparisons must not be hidden:

{table(['Comparison pool','LANS first/joint-first of14','Mean numerical rank'],ranktable)}

Original means the three previously reported baselines. Equal-pretraining uses the single strongest whole PPO reference selected on validation before testing: **{strong}**. Objective-matched additionally replaces the declared MPC objective with the new tracking/control objective. All-references also includes BC-only neural baselines and prior LANS. First-place numerical means do not establish statistical superiority,universal optimality or first place against every possible baseline.

## What stayed the same and what changed

The same signed dynamic bicycle model and parameter magnitudes,100Hz exact ZOH plant,full-command FIFO delay,LQR+scalar PPO residual+exponential latency-gain+saturation topology,11 causal observation channels and64x64ReLU actor/critic are retained. No queue/history input,online predictor,MPC branch or extra state is added to deployed LANS.

Changes are explicit: model-derived offline supervised pretraining of the existing policy; invertible linear conditioning of the same four40ms preview channels; residual scale1.0rad with final physical steering still bounded±0.5rad; exponential decay0.002/ms; lateral-error-focused reward; and conservative projected output-layer PPO fine-tuning. At200ms beta is about0.677,so the neural residual remains active. The old interpretation of complete neural shutdown at200ms is not retained. This is not an unchanged-parameter reproduction and must not be presented as purely model-free PPO training.

The analytic target in core.py is used offline only. It derives control labels from the nominal model and the same11 observations,not hidden plant parameters,new queue history or future reference beyond the existing samples. The deployed actor is the saved neural network. checks.py disables the teacher function while the saved network controls the original scalar plant. No analytic teacher is evaluated as LANS performance.

## Data,pretraining and failed trials

Teacher parameters were searched using development/validation seeds22000–22001 before network training. The384000-row corpus uses separate training seeds,random reference amplitude/time scaling,mass variation and perturbed current-state features; no final test data. All four reference families are in training,so no unseen-family zero-shot claim is made.

Three independent seeds72–74 are used for each of LANS and standalone PPO. Both receive6000 supervised updates with the same corpus,architecture and physical-control-loss accounting,then100352 actual PPO steps. The initial unrestricted PPO trial regressed LANS validation completion from84/84 to74/84. Its six checkpoints,failures and logs are preserved.

A validation-driven corrective trial was recorded before final testing. Actor hidden layers and logstd are frozen; output parameters are projected within1e-5 physical-action-scaled distance of their pretrained values before rollouts and after training. All six corrective checkpoints also execute100352 PPO steps. This is projected output-layer PPO fine-tuning,not unmodified vanilla PPO. The BC-only snapshots remain as ablations; they are not silently relabelled PPO-trained policies. Most improvement may come from supervision/conditioning rather than reinforcement updates.

The final reward is -1000ey_next²-.01heading_next²-.001command²-.01(command-previous_command)²,with failure penalty250. It does not pretend to control external computation latency.

## Fresh test and metrics

Final seeds26000–26004 and two new amplitude/time-scale combinations(0.88,1.04),(1.12,0.96) were frozen before network training. Testing starts after all corrective policies finish and after the whole PPO baseline is selected on validation. There is no per-scenario switching or best-seed choice. Original and previous stronger checkpoints remain matched references.

All{len(rows)} test episodes and raw traces are retained. Early failures use the previously declared remaining-horizon penalty and completion counts. Learned confidence intervals use3 training-seed means conditional on the physical/reference cases; baseline intervals use5 physical-seed means. These small-sample intervals do not establish universal dominance.

{table(['Condition','LANS','Original PPO','Equal-pretrained PPO','Declared MPC','Objective-matched MPC','LANS completed'],tab)}

Values are mean failure-penalized RMSE in metres. Lower is better. No-delay nominal cases and harsh-delay cases must not be pooled to claim every-condition superiority.

## Stronger baseline and ablation disclosure

Standalone PPO receives the same model-informed supervision,network size,features and optimization opportunity. Original PPO is retained too. The supplementary objective-matched MPC uses the same1000ey²+.01heading²+.001u²+.01delta-u² objective,a20-step horizon,OSQP1.1.3 with eps_abs/rel1e-8,50000 maximum iterations and polishing,and±0.5rad steering bound; it remains a declared finite-horizon baseline without delay prediction. The original L-BFGS-B implementation reported547 solver failures; those initial results are preserved separately and replaced by the same QP solved with OSQP,without changing LANS checkpoints or MPC cost. MPC has20-step reference preview whereas policy inputs retain four samples. No superiority over optimally tuned or delay-aware MPC is claimed.

{table(['Controller/checkpoint','Completed','Aggregate penalized RMSE'],[[name,str(v['completed'])+'/'+str(v['episodes']),f"{v['mean_rmse']:.6f}"] for name,v in totals.items()])}

BC versus corrective PPO differences are retained in paired_differences.csv. Tiny improvements or regressions after constrained fine-tuning do not establish a distinct PPO-learning contribution. A first-place result against original baselines is not proof of superiority over the equally pretrained policy or a matched-objective optimizer.

## Numerical and physical limits

The vectorized offline teacher simulator and canonical scalar teacher check differ by up to0.197mm in tested aggregate errors because fourth-derivative estimates amplify float32 preview rounding; completion decisions match. Strict1e-6 metric equality failed and is explicitly recorded. All final learned-controller results come solely from the original scalar simulator,not the vectorized offline search.

The vehicle model is a linear small-angle approximation without tire-force limits. Aggressive slalom tests are numerical stress tests,not proof of tire-feasible high-speed driving. No physical vehicle/ECU,energy,end-to-end hardware latency,certified WCET,global saturated stability or arbitrary-delay guarantee is established. Earlier manuscripts and adverse studies remain unchanged and must not be described as already incorporating these results.
"""
 (ROOT/'Tracking_Report.md').write_text(report)
 print(json.dumps(dict(primary_LANS=totals['limited_lans'],rankings={k:v['limited_lans'] for k,v in overall.items()},BC_LANS=totals['bc_lans']),indent=2))

if __name__=='__main__':main()
