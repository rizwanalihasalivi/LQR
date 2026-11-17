"""Freeze all comparison checkpoints after training and before final testing."""
from pathlib import Path
import hashlib
import json
from run_eval import expected_checkpoints

ROOT = Path(__file__).resolve().parents[1]
dest = ROOT / 'results/checkpoint_manifest.json'
if dest.exists() or (ROOT / 'results/test').exists():
    raise FileExistsError('Manifest/test already exists; preserve the frozen experiment.')
for family in ('lans','ppo'):
    for seed in (82,83,84):
        meta = json.loads((ROOT/'models'/f'corrected_{family}_{seed}'/'training.json').read_text())
        assert meta['PPO_steps'] == 100352
        assert meta['training_delay_coverage']['at_200ms'] > 0
        assert meta['projection_optimizer_steps'] == 15680
hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
          for name,path in expected_checkpoints().items()}
assert len(hashes) == 18
with dest.open('x') as f:
    json.dump(dict(checkpoints=hashes,
                   protocol_sha256=hashlib.sha256((ROOT/'results/protocol.json').read_bytes()).hexdigest(),
                   rule='All three seeds and BC ablations, frozen before final evaluation; no final-result tuning.'),f,indent=2)
print('Frozen',len(hashes),'checkpoints')
