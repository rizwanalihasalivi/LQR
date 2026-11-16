from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';rows=list(csv.DictReader((R/'episodes_all.csv').open()))
assert len(rows)==1225
assert len(set((r['kind'],r['train_seed'],r['physical_seed'],r['path'],r['delay'],r['scope'],r['mass']) for r in rows))==1225
metadata=[]
for kind in ['full','no_reward','ppo']:
 for seed in [42,43,44]:
  m=json.loads((ROOT/'models'/f'{kind}_{seed}'/'training.json').read_text())
  assert m['seed']==seed and m['kind']==kind and m['actual_steps']==200704
  assert (ROOT/'models'/f'{kind}_{seed}'/'policy.zip').stat().st_size>0
  metadata.append(m)
for r in rows:
 key=f"{r['kind']}_t{r['train_seed']}_p{r['physical_seed']}_{r['path']}_{r['delay']}_{r['scope']}_m{float(r['mass']):g}"
 with np.load(R/'traces'/f'{key}.npz') as z:a=z['trace']
 n=int(r['steps']);assert len(a)==n and a.shape[1]==11 and np.isfinite(a).all()
 assert np.max(np.abs(a[:,3]))<=.5000001 and np.max(np.abs(a[:,4]))<=.5000001
 initial=np.random.default_rng(int(r['physical_seed'])).normal(0,[.05,.02,.005,.002])[0]
 sse=initial**2+np.square(a[:,1]).sum()
 pen=float(np.sqrt((sse+(1000-n)*max(2.5,abs(a[-1,1]))**2)/1001))
 assert np.isclose(pen,float(r['rmse_penalized']),atol=1e-12)
 assert (r['completed']=='True') <= (n==1000)
 assert (a[:,10]<=np.arange(n)).all()
# No-gain and full must coincide at delay zero because original gain is one.
lookup={(r['kind'],r['train_seed'],r['physical_seed'],r['path'],r['delay'],r['scope'],r['mass']):r for r in rows}
for key,r in lookup.items():
 if key[0]=='full' and key[4]=='none':
  other=lookup[('no_gain',*key[1:])]
  assert r['rmse_penalized']==other['rmse_penalized']
tim=list(csv.DictReader((R/'timing_raw.csv').open()));assert len(tim)==21000
for k in set(r['kind'] for r in tim):
 rr=[r for r in tim if r['kind']==k];assert len(rr)==3000 and len(set(r['sample'] for r in rr))==3000
 assert min(int(r['ns']) for r in rr)>0
assert sum(int(r['mpc_optimizer_failures']) for r in rows)==0
from zipfile import ZipFile
from lxml import etree
for stem in ['LANS_Revised_Manuscript','Response_to_Reviewer_1']:
 with ZipFile(ROOT/'deliverables'/f'{stem}.docx') as z:
  assert z.testzip() is None
  root=etree.fromstring(z.read('word/document.xml'));text=''.join(root.itertext())
  if stem=='LANS_Revised_Manuscript':
   assert len(root.xpath('//*[local-name()="oMath"]'))>=25
   assert '1225' in text or '1,225' in text
  else:
   assert all(f'Comment {i}:' in text for i in range(1,13))
audit=dict(status='passed',trained_policies=9,total_training_steps=sum(m['actual_steps'] for m in metadata),evaluation_episodes=len(rows),per_step_trace_files=len(list((R/'traces').glob('*.npz'))),timing_observations=len(tim),mpc_optimizer_failures=0,checks=['checkpoint budget/seed validation','unique episode keys','all trace lengths and finite values','all steering constraints','penalized metrics recomputed from traces','command generation causality','zero-delay frozen full/no-gain equality','timing sample counts','valid editable DOCX and native Word mathematics','all twelve reviewer responses present'],scope='Integrity checks confirm reported experiment execution and document consistency, not physical safety or reviewer acceptance.')
(R/'audit.json').write_text(json.dumps(audit,indent=2))
manifest={}
for folder in ['code','models','results','deliverables']:
 for p in sorted((ROOT/folder).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.name!='integrity_manifest.json':
   manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['README.md','requirements_snapshot.txt']:
 p=ROOT/name
 manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'integrity_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(audit,indent=2))
