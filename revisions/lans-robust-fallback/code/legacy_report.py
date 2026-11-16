from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
old=[]
for kind in ['full','ppo']:
 for seed in [42,43,44]:old+=list(csv.DictReader((R/'legacy_reference'/f'{kind}_{seed}.csv').open()))
assert len(old)==840
new=list(csv.DictReader((R/'all_episodes.csv').open()))
conditions=sorted(set((r['path'],r['delay'],r['mass']) for r in new));rows=[]
for c in conditions:
 row=dict(path=c[0],delay=c[1],mass=c[2])
 for label,name,data,key in [('new_lans','lans',new,'kind'),('legacy_lans','legacy_full',old,'candidate'),('new_ppo','ppo',new,'kind'),('legacy_ppo','legacy_ppo',old,'candidate'),('robust_lqr','lqr',new,'kind'),('fast_lqr','fast_lqr',new,'kind'),('mpc','mpc',new,'kind')]:
  group=[r for r in data if r[key]==name and (r['path'],r['delay'],r['mass'])==c]
  row[label+'_rmse']=float(np.mean([float(r['rmse_penalized']) for r in group]))
  row[label+'_completed']=sum(r['completed']=='True' for r in group);row[label+'_episodes']=len(group)
 rows.append(row)
with (R/'legacy_reference/matched_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
ranks={k:0 for k in ['new_lans','legacy_ppo','fast_lqr','mpc']}
wins={k:0 for k in ranks}
for row in rows:
 for k in ranks:
  rank=1+sum(row[j+'_rmse']<row[k+'_rmse']-1e-12 for j in ranks)
  ranks[k]+=rank;wins[k]+=rank==1
ranks={k:v/14 for k,v in ranks.items()}
overall={k:dict(mean_rmse=float(np.mean([row[k+'_rmse'] for row in rows])),completed=sum(row[k+'_completed'] for row in rows),episodes=sum(row[k+'_episodes'] for row in rows)) for k in ['new_lans','legacy_lans','legacy_ppo','new_ppo','fast_lqr','robust_lqr','mpc']}
lines=['\n## Matched comparison with the original stronger PPO and LANS\n','Original final checkpoints (seeds42–44) were run on exactly the same fresh cases using their original observation API. These840 supplementary episodes are not used to alter or retune the new policies. They prevent a weaker freshly retrained PPO baseline from creating an apparent LANS improvement.\n','| Condition | New LANS RMSE | Original LANS RMSE | Original PPO RMSE |','|---|---:|---:|---:|']
for row in rows:lines.append(f"| {row['path']}/{row['delay']}/{float(row['mass']):g}kg | {row['new_lans_rmse']:.4f} | {row['legacy_lans_rmse']:.4f} | {row['legacy_ppo_rmse']:.4f} |")
lines.append('\nWhen the original strong PPO and original fast LQR are retained with MPC, the four-controller mean condition ranks are:\n')
for k,v in sorted(ranks.items(),key=lambda x:x[1]):lines.append(f'- {k}: {v:.3f}; first/joint-first in {wins[k]}/14 conditions.')
lines.append('\nThe overall penalized RMSE averaged equally over14 conditions is a different aggregate from average ordinal rank. Both are reported; the metric is not switched to claim the requested outcome:\n')
for k,v in overall.items():lines.append(f"- {k}: {v['mean_rmse']:.6f}m; completed {v['completed']}/{v['episodes']}.")
lines.append('\nAt200ms, the retuned fallback explains the gain, not the learned residual. Ordinary nominal tracking and high-delay completion must be presented separately; lower overall failure-penalized RMSE does not mean best tracking in each scenario. No statistically established universal superiority is claimed.')
p=ROOT/'Robust_Fallback_Report.md';body=p.read_text().split('\n## Matched comparison')[0];p.write_text(body+'\n'.join(lines)+'\n')
result=dict(supplementary_episodes=len(old),four_controller_mean_ranks_with_original_baselines=ranks,first_conditions_with_original_baselines=wins,aggregate_scores=overall)
(R/'legacy_reference/summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
