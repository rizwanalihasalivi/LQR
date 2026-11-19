import csv
import math
from pathlib import Path
from statistics import mean

out = Path(__file__).parent
baseline = out.parent / 'results' / 'test'
reference = {}
key = lambda r: (r['path'], r['physical_seed'], float(r['amplitude']), float(r['time_scale']))
for label, name, seeds in [('LQR', 'robust_lqr', [-1]), ('MPC', 'matched_mpc', [-1]), ('LANS', 'limited_lans', [72,73,74]), ('PPO', 'limited_ppo', [72,73,74])]:
    for seed in seeds:
        with (baseline / f'{name}_{seed}.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['delay'] in ('none', 'fixed200') and float(row['mass']) == 1650.:
                    reference[label, str(seed), '0' if row['delay']=='none' else '200', key(row)] = row
with (out / 'episodes.csv').open() as stream:
    episodes = list(csv.DictReader(stream))
checked = 0
for row in episodes:
    if row['delay_ms'] in ('0', '200'):
        ref = reference[row['controller'], row['train_seed'], row['delay_ms'], key(row)]
        assert row['completed'] == str(int(ref['completed'] == 'True'))
        assert row['steps'] == ref['steps']
        for col in ('rmse_observed', 'rmse_penalized'):
            assert math.isclose(float(row[col + '_m']), float(ref[col]), abs_tol=1e-10)
        checked += 1
assert checked == 640
print(f'Cross-checked {checked} episodes against saved 0 and 200 ms results.')
with (out / 'summary.csv').open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 44
for delay in sorted({int(r['delay_ms']) for r in rows}):
    group = [r for r in rows if int(r['delay_ms']) == delay]
    by = {r['controller']: r for r in group}
    cells = [f"{float(by[c]['rmse_penalized_m']):.4f} ({by[c]['completed']}/{by[c]['episodes']})" for c in ['LQR','PPO','MPC','LANS']]
    print('| ' + ' | '.join([str(delay), *cells, ' > '.join(r['controller'] for r in group)]) + ' |')
overall = {c: mean(float(r['rmse_penalized_m']) for r in rows if r['controller']==c) for c in ['LQR','PPO','MPC','LANS']}
print(sorted(overall.items(), key=lambda r: r[1]))
