"""Post-hoc no-learning ablation; not used for tuning or selection."""
from pathlib import Path
import json,csv
import numpy as np
from robust_env import RobustEnv,robust_gain
from dynamic_env import design
from evaluate import cases
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
K=robust_gain();KF=design()
rows=[];out=R/'prior_reference';out.mkdir(exist_ok=True);(out/'traces').mkdir(exist_ok=True)
for c in cases('test'):
 e=RobustEnv(kind='lans',path=c['path'],mass=c['mass'],delay_mode=c['delay'],amplitude=c['amplitude'],time_scale=c['time_scale'])
 e.reset(seed=c['physical_seed']);initial=float(e.x[0]);trace=[]
 for k in range(1000):
  before=e.x.copy()
  # Exact deterministic function of the initialized actor, before PPO updates.
  action=[np.clip((K-KF)@e.x/.2,-1,1)]
  _,_,t,tr,i=e.step(action)
  trace.append([k*.01,i['error'],i['heading_error'],i['u'],i['applied'],i['residual'],i['gain'],i['delay_ms'],i['queue'],*before])
  if t or tr:break
 arr=np.array(trace);n=len(arr);sse=initial**2+np.square(arr[:,1]).sum()
 rows.append(dict(kind='prior_lans',**c,steps=n,completed=not t and tr,rmse_penalized=float(np.sqrt((sse+(1000-n)*max(2.5,abs(arr[-1,1]))**2)/1001))))
 key=f"{c['physical_seed']}_{c['path']}_{c['delay']}_m{c['mass']:g}_a{c['amplitude']:g}_t{c['time_scale']:g}"
 np.savez_compressed(out/'traces'/f'{key}.npz',trace=arr,initial_error=initial)
with (out/'episodes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
new=list(csv.DictReader((R/'summary.csv').open()));table=[]
for c in sorted(set((r['path'],r['delay'],r['mass']) for r in new)):
 rr=[r for r in rows if (r['path'],r['delay'],str(r['mass']))==c]
 mean=float(np.mean([r['rmse_penalized'] for r in rr]));n=next(r for r in new if r['kind']=='lans' and (r['path'],r['delay'],r['mass'])==c)
 table.append(dict(path=c[0],delay=c[1],mass=c[2],prior_rmse=mean,trained_lans_rmse=float(n['mean_penalized_rmse']),trained_minus_prior=float(n['mean_penalized_rmse'])-mean,prior_completed=sum(r['completed'] for r in rr),prior_episodes=len(rr)))
with (out/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
summary=dict(episodes=len(rows),completed=sum(r['completed'] for r in rows),mean_prior_rmse=float(np.mean([r['rmse_penalized'] for r in rows])),mean_trained_rmse=float(np.mean([r['trained_lans_rmse'] for r in table])),scope='Post-hoc ablation on existing test cases, never used for model selection. Exposes whether PPO training improves on the encoded linear prior; not an additional tuned controller.')
(out/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['\n## Post-hoc no-learning ablation\n','To separate reinforcement learning from the encoded analytical prior, the exact deterministic initialized actor function is evaluated before any PPO updates. This controller is an ablation only, not a newly selected deployment variant. The same fresh cases are reused solely for diagnosis, with no retuning.\n','| Condition | No-learning prior RMSE | Trained LANS RMSE | Trained minus prior |','|---|---:|---:|---:|']
for r in table:lines.append(f"| {r['path']}/{r['delay']}/{float(r['mass']):g}kg | {r['prior_rmse']:.4f} | {r['trained_lans_rmse']:.4f} | {r['trained_minus_prior']:+.4f} |")
lines.append(f"\nNo-learning prior completed{summary['completed']}/{summary['episodes']} episodes, with mean penalized RMSE{summary['mean_prior_rmse']:.6f}m. Trained LANS mean is{summary['mean_trained_rmse']:.6f}m. This ablation prevents calling retuned fallback or encoded LQR gain scheduling a demonstrated PPO learning benefit. Positive differences mean PPO training worsened the metric; negative differences indicate improvement on this finite test.")
p=ROOT/'Robust_Fallback_Report.md';body=p.read_text().split('\n## Post-hoc no-learning')[0];p.write_text(body+'\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
