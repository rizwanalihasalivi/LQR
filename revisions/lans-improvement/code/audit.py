from pathlib import Path
import csv,json,hashlib
import numpy as np
from protocol import P
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
assert json.loads((R/'protocol.json').read_text())==P
training=[]
for name in P['candidates']:
 for seed in P['training_seeds']:
  path=ROOT/'models'/f'{name}_{seed}'
  m=json.loads((path/'training.json').read_text())
  assert m['candidate']==name and m['seed']==seed and m['actual_steps']==200704
  assert m['configuration']==P['candidates'][name]
  training.append(m)
s=json.loads((R/'selection.json').read_text())
assert s['protocol_sha256']==hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest()
for kind in ['lans','ppo']:
 candidates=[n for n,c in P['candidates'].items() if c['kind']==kind]
 def score(n):
  rows=[]
  for seed in P['training_seeds']:rows+=list(csv.DictReader((R/'validation'/f'{n}_{seed}.csv').open()))
  assert len(rows)==84
  return (-np.mean([r['completed']=='True' for r in rows]),np.mean([float(r['rmse_penalized']) for r in rows]),n)
 assert s[kind]==min(candidates,key=score)
rows=list(csv.DictReader((R/'test/all_episodes.csv').open()));assert len(rows)==1120
assert set(r['candidate'] for r in rows)=={s['lans'],s['ppo'],'lqr','mpc'}
keys=set()
for r in rows:
 name=r['candidate'];seed=r['train_seed'];key=f"{name}_{seed}_{r['physical_seed']}_{r['path']}_{r['delay']}_m{float(r['mass']):g}_a{float(r['amplitude']):g}_t{float(r['time_scale']):g}"
 assert key not in keys;keys.add(key)
 path=R/'test/traces'/f'{key}.npz'
 assert path.stat().st_mtime>=s['selected_at_unix']
 with np.load(path) as z:a=z['trace'];initial=float(z['initial_error']);columns=z['columns']
 assert len(a)==int(r['steps']) and a.shape[1]==13 and len(columns)==13 and np.isfinite(a).all()
 assert np.max(abs(a[:,3]))<=.5000001 and np.max(abs(a[:,4]))<=.5000001
 n=len(a);sse=initial**2+np.square(a[:,1]).sum()
 pen=np.sqrt((sse+(1000-n)*max(2.5,abs(a[-1,1]))**2)/1001)
 assert np.isclose(pen,float(r['rmse_penalized']),atol=1e-12)
 assert (r['completed']=='True') <= (n==1000)
assert sum(int(r['optimizer_failures']) for r in rows)==0
runtime=json.loads((R/'runtime.json').read_text())
for name,h in runtime['original_source_hashes'].items():
 p=ROOT.parent/'dynamic-bicycle/code'/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h
# Retain and verify all supplementary original-checkpoint references.
legacy_count=0
for kind in ['full','ppo']:
 for seed in [42,43,44]:
  legacy=list(csv.DictReader((R/'legacy_reference'/f'{kind}_{seed}.csv').open()))
  assert len(legacy)==140
  for r in legacy:
   key=f"{kind}_{seed}_{r['physical_seed']}_{r['path']}_{r['delay']}_m{float(r['mass']):g}_a{float(r['amplitude']):g}_t{float(r['time_scale']):g}"
   with np.load(R/'legacy_reference/traces'/f'{key}.npz') as z:a=z['trace'];initial=float(z['initial_error'])
   n=len(a);pen=np.sqrt((initial**2+np.square(a[:,1]).sum()+(1000-n)*max(2.5,abs(a[-1,1]))**2)/1001)
   assert n==int(r['steps']) and np.isclose(pen,float(r['rmse_penalized']),atol=1e-12)
  legacy_count+=len(legacy)
assert legacy_count==840
audit=dict(status='passed',trained_policies=len(training),training_steps=sum(m['actual_steps'] for m in training),validation_episodes=392,original_checkpoint_reference_episodes=legacy_count,final_test_episodes=len(rows),checks=['all configurations and seeds retained','frozen protocol unchanged','selection recomputed from validation','all final traces created after selection','complete trace/metric recomputation','actuator bound checks','source dynamic model unchanged','no MPC optimizer failures'],scope='Audits execution and consistency, not universal dominance or safety.')
(R/'audit.json').write_text(json.dumps(audit,indent=2))
manifest={}
for p in sorted(ROOT.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts and p.name!='integrity_manifest.json' and p.suffix!='.zip':
  manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'integrity_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(audit,indent=2))
