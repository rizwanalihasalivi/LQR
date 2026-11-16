from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
s=json.loads((R/'selection.json').read_text())
old=[]
for kind in ['full','ppo']:
 for seed in [42,43,44]:old+=list(csv.DictReader((R/'legacy_reference'/f'{kind}_{seed}.csv').open()))
assert len(old)==840
new=list(csv.DictReader((R/'test/all_episodes.csv').open()))
conditions=sorted(set((r['path'],r['delay'],r['mass']) for r in new))
rows=[]
for c in conditions:
 row=dict(path=c[0],delay=c[1],mass=c[2])
 for label,name,data in [('new_lans',s['lans'],new),('legacy_lans','legacy_full',old),('new_ppo',s['ppo'],new),('legacy_ppo','legacy_ppo',old),('lqr','lqr',new),('mpc','mpc',new)]:
  q=[r for r in data if r['candidate']==name and (r['path'],r['delay'],r['mass'])==c]
  assert q
  row[label+'_rmse']=float(np.mean([float(r['rmse_penalized']) for r in q]))
  row[label+'_completion']=str(sum(r['completed']=='True' for r in q))+'/'+str(len(q))
 rows.append(row)
with (R/'legacy_reference/matched_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rank={k:0. for k in ['new_lans','legacy_ppo','lqr','mpc']}
for r in rows:
 for k in rank:rank[k]+=1+sum(r[other+'_rmse']<r[k+'_rmse']-1e-12 for other in rank)
rank={k:v/14 for k,v in rank.items()}
lines=['\n## Matched checks against the earlier stronger checkpoints\n','The original full-LANS and PPO policies (seeds42–44) were also run on exactly these new test cases, using their exact original observation API. There are840 supplementary episodes. These checks do not change the locked validation selection; they prevent attributing a rank change to a weaker retrained PPO baseline.\n','| Condition | New LANS RMSE | Earlier LANS RMSE | Earlier PPO RMSE |','|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['path']} / {r['delay']} / {float(r['mass']):g}kg | {r['new_lans_rmse']:.4f} | {r['legacy_lans_rmse']:.4f} | {r['legacy_ppo_rmse']:.4f} |")
lines+=['\nA supplementary four-controller comparison retaining the stronger earlier PPO yields these mean condition ranks:\n']
for k,v in sorted(rank.items(),key=lambda t:t[1]):lines.append(f'- {k}: {v:.3f}')
lines.append('\nThis supplementary ranking is explicitly separate from the prospectively selected comparison. The target of LANS first in every condition remains unachieved. New-versus-old means do not establish statistically significant improvement; the policies use different training seeds and training distributions.')
text='\n'.join(lines)+'\n'
p=ROOT/'Improvement_Report.md';body=p.read_text().split('\n## Matched checks')[0];p.write_text(body+text)
(R/'legacy_reference/summary.json').write_text(json.dumps(dict(episodes=len(old),supplementary_mean_ranks=rank),indent=2))
print(json.dumps(dict(episodes=len(old),supplementary_mean_ranks=rank),indent=2))
