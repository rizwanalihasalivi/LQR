from pathlib import Path
import csv,json,hashlib
import numpy as np
from study_plan import PLAN
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
assert json.loads((R/'plan.json').read_text())==PLAN
meta=[]
for kind in PLAN['trained_families']:
 for seed in PLAN['training_seeds']:
  m=json.loads((ROOT/'models'/f'{kind}_{seed}'/'training.json').read_text())
  assert m['kind']==kind and m['seed']==seed and m['actual_steps']==200704
  meta.append(m)
gain=json.loads((R/'robust_gain.json').read_text());assert len(gain['candidates'])==33
eligible=[g for g in gain['candidates'] if g['eligible']]
chosen=min(eligible,key=lambda g:(-g['completion'],g['mean_penalized_rmse'],g['R']))
assert chosen==gain['chosen']
certificate=json.loads((R/'certificate.json').read_text());assert len(certificate['cases'])==21 and all(r['spectral_radius']<1 and r['lyapunov_residual']<1e-6 for r in certificate['cases'])
marker=json.loads((R/'test_started.json').read_text())
rows=list(csv.DictReader((R/'all_episodes.csv').open()));assert len(rows)==1260
keys=set()
for r in rows:
 key=f"{r['kind']}_{r['train_seed']}_{r['physical_seed']}_{r['path']}_{r['delay']}_m{float(r['mass']):g}_a{float(r['amplitude']):g}_t{float(r['time_scale']):g}"
 assert key not in keys;keys.add(key)
 path=R/'test/traces'/f'{key}.npz';assert path.stat().st_mtime>=marker['started_at_unix']
 with np.load(path) as z:a=z['trace'];initial=float(z['initial_error'])
 assert len(a)==int(r['steps']) and a.shape[1]==13 and np.isfinite(a).all()
 assert np.max(abs(a[:,3]))<=.5000001 and np.max(abs(a[:,4]))<=.5000001
 n=len(a);pen=np.sqrt((initial**2+np.square(a[:,1]).sum()+(1000-n)*max(2.5,abs(a[-1,1]))**2)/1001)
 assert np.isclose(pen,float(r['rmse_penalized']),atol=1e-12)
 assert (r['completed']=='True') <= (n==1000)
# High-delay improvement is inherited from the matched fallback, not PPO.
for r in rows:
 if r['kind']=='lans' and r['delay']=='fixed200':
  other=next(b for b in rows if b['kind']=='lqr' and all(b[k]==r[k] for k in ['physical_seed','path','delay','mass','amplitude','time_scale']))
  assert r['rmse_penalized']==other['rmse_penalized'] and r['completed']==other['completed']
assert sum(int(r['optimizer_failures']) for r in rows)==0
old=[]
for kind in ['full','ppo']:
 for seed in [42,43,44]:
  rr=list(csv.DictReader((R/'legacy_reference'/f'{kind}_{seed}.csv').open()));assert len(rr)==140
  for r in rr:
   key=f"{kind}_{seed}_{r['physical_seed']}_{r['path']}_{r['delay']}_m{float(r['mass']):g}_a{float(r['amplitude']):g}_t{float(r['time_scale']):g}"
   with np.load(R/'legacy_reference/traces'/f'{key}.npz') as z:a=z['trace'];initial=float(z['initial_error'])
   n=len(a);pen=np.sqrt((initial**2+np.square(a[:,1]).sum()+(1000-n)*max(2.5,abs(a[-1,1]))**2)/1001)
   assert n==int(r['steps']) and np.isclose(pen,float(r['rmse_penalized']),atol=1e-12)
  old+=rr
assert len(old)==840
prior=list(csv.DictReader((R/'prior_reference/episodes.csv').open()))
assert len(prior)==140
for r in prior:
 key=f"{r['physical_seed']}_{r['path']}_{r['delay']}_m{float(r['mass']):g}_a{float(r['amplitude']):g}_t{float(r['time_scale']):g}"
 with np.load(R/'prior_reference/traces'/f'{key}.npz') as z:a=z['trace'];initial=float(z['initial_error'])
 n=len(a);pen=np.sqrt((initial**2+np.square(a[:,1]).sum()+(1000-n)*max(2.5,abs(a[-1,1]))**2)/1001)
 assert n==int(r['steps']) and np.isclose(pen,float(r['rmse_penalized']),atol=1e-12)
result=dict(status='passed',trained_policies=6,training_steps=sum(m['actual_steps'] for m in meta),final_test_episodes=len(rows),original_checkpoint_reference_episodes=len(old),no_learning_reference_episodes=len(prior),tuning_candidates=len(gain['candidates']),tuning_episode_evaluations=sum(len(g.get('validation_rows',[])) for g in gain['candidates']),fixed_delay_lyapunov_cases=21,checks=['frozen plan','all seeds/final checkpoints','validation-only gain choice recomputed','all test traces later than sealed start','all primary and legacy metrics recomputed','same physical actuation bounds','LANS equals robust fallback at fixed200','zero MPC solver failures'],scope='Consistency and finite campaign evidence, not every-scenario superiority, full hybrid stability or physical safety.')
(R/'audit.json').write_text(json.dumps(result,indent=2))
manifest={}
for p in sorted(ROOT.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts and p.name!='integrity_manifest.json' and p.suffix!='.zip':
  manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'integrity_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(result,indent=2))
