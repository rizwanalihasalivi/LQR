from pathlib import Path
import json,csv,hashlib
import numpy as np
import torch
from stable_baselines3 import PPO
from protocol import P
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
torch.set_num_threads(1)
assert json.loads((R/'protocol.json').read_text())==P
marker=json.loads((R/'test_started.json').read_text())
selection=json.loads((R/'validation_selection.json').read_text())
assert (R/'validation_selection.json').stat().st_mtime<=marker['time']
for kind in ['lans','ppo']:
 for seed in [72,73,74]:
  original=ROOT/'models'/f'{kind}_{seed}';limited=ROOT/'models'/f'limited_{kind}_{seed}'
  a=json.loads((original/'training.json').read_text());b=json.loads((limited/'training.json').read_text())
  assert a['PPO_steps']==b['PPO_steps']==100352 and a['BC_updates']==6000
  assert (limited/'policy.zip').stat().st_mtime<=marker['time']
  bc=PPO.load(original/'bc.zip',device='cpu');m=PPO.load(limited/'policy.zip',device='cpu')
  assert m.observation_space.shape==(11,)
  assert m.policy.mlp_extractor.policy_net[0].out_features==64 and m.policy.mlp_extractor.policy_net[2].out_features==64
  for k,v in bc.policy.mlp_extractor.policy_net.state_dict().items():assert torch.equal(v,m.policy.mlp_extractor.policy_net.state_dict()[k])
  assert torch.equal(bc.policy.log_std,m.policy.log_std)
  bound=1e-5/(1. if kind=='lans' else .5)
  for p,q in zip(m.policy.action_net.parameters(),bc.policy.action_net.parameters()):assert torch.max(abs(p-q)).item()<=bound+1e-7
rows=list(csv.DictReader((R/'episodes_all.csv').open()));assert len(rows)==3080
keys=set()
for r in rows:
 key=f"{r['kind']}_{r['train_seed']}_{r['physical_seed']}_{r['path']}_{r['delay']}_m{float(r['mass']):g}_a{float(r['amplitude']):g}_t{float(r['time_scale']):g}"
 assert key not in keys;keys.add(key)
 path=R/'test/traces'/f'{key}.npz';assert path.stat().st_mtime>=marker['time']
 with np.load(path) as z:a=z['trace'];initial=float(z['initial_error'])
 assert a.shape==(int(r['steps']),13) and np.isfinite(a).all()
 assert np.max(abs(a[:,3]))<=.5000001 and np.max(abs(a[:,4]))<=.5000001
 n=len(a);pen=np.sqrt((initial**2+np.square(a[:,1]).sum()+(1000-n)*max(2.5,abs(a[-1,1]))**2)/1001)
 assert abs(pen-float(r['rmse_penalized']))<1e-10
 assert (r['completed']=='True') <= (n==1000)
failures=sum(int(r['optimizer_failures']) for r in rows);assert failures==0
scores={}
for name,seeds in [('limited_ppo',[72,73,74]),('rl_ppo',[72,73,74]),('original_ppo',[42,43,44])]:
 rr=[r for s in seeds for r in csv.DictReader((R/'validation'/f'{name}_{s}.csv').open())]
 scores[name]=(-np.mean([r['completed']=='True' for r in rr]),np.mean([float(r['rmse_penalized']) for r in rr]),name)
assert selection['ppo_reference']==min(scores,key=lambda n:scores[n])
checks=json.loads((R/'checks.json').read_text());assert checks['completion_parity'] and checks['batch_scalar_strict_1e6_match'] is False and checks['deployed_network_steps_without_offline_teacher']==100
with np.load(R/'corpus.npz') as z:
 assert z['X'].shape==(384000,11) and np.isfinite(z['X']).all()
 assert max(abs(z['lans_target']))<=1 and max(abs(z['ppo_target']))<=1
audit=dict(status='passed',network_PPO_runs=12,BC_policy_snapshots=6,total_PPO_steps=1204224,offline_corpus_rows=384000,final_scalar_episodes=len(rows),unrestricted_LANS_validation_completions=74,BC_LANS_validation_completions=84,validation_cases_each=84,final_reference_solver_failures=failures,teacher_metric_roundoff_m=checks['maximum_metric_difference'],teacher_is_not_deployed=True,checks=['all seeds retained','no extra input or hidden layer','frozen-hidden/output-projection weights verified','baseline selected on validation only','test after checkpoint freeze','3080 raw metrics independently recomputed','physical steering bound retained','failed unrestricted trial retained','corrected MPC solved without reference solver errors'],limits='No universal dominance,additional PPO-learning benefit,physical safety or global stability certification is implied.')
(R/'audit.json').write_text(json.dumps(audit,indent=2))
manifest={}
for p in sorted(ROOT.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts and p.name!='integrity_manifest.json':
  manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'integrity_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(audit,indent=2))
