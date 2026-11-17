"""Validate a complete campaign and report every controller and failed episode."""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('corrected_runner', ROOT / 'code/run_eval.py')
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
evaluation = runner.evaluation
GROUPS = {
    'historical_tracking': ('corrected_lans', 'previous_ppo', 'fast_lqr', 'mpc'),
    'equal_pretraining': ('corrected_lans', 'corrected_ppo', 'fast_lqr', 'mpc'),
    'objective_matched': ('corrected_lans', 'corrected_ppo', 'fast_lqr', 'matched_mpc'),
    'all_references': tuple(evaluation.KINDS),
}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_key(row):
    return (row['path'], row['delay'], float(row['mass']), float(row['amplitude']),
            float(row['time_scale']), int(row['physical_seed']))


def trace_key(row):
    return (f"{row['kind']}_{row['train_seed']}_{row['physical_seed']}_{row['path']}_{row['delay']}"
            f"_m{float(row['mass']):g}_a{float(row['amplitude']):g}_t{float(row['time_scale']):g}")


def verify_trace(path, row):
    with np.load(path, allow_pickle=False) as data:
        trace, initial = data['trace'], float(data['initial_error'])
    count = int(row['steps'])
    if not 1 <= count <= 1000 or trace.shape != (count, 13):
        raise ValueError(f'Invalid trace size: {path}')
    if not np.isfinite(trace).all() or not np.isfinite(initial):
        raise ValueError(f'Nonfinite trace: {path}')
    if np.max(abs(trace[:, 3:5])) > .5000001:
        raise ValueError(f'Steering bound violated: {path}')
    if row['completed'] not in ('True', 'False') or (row['completed'] == 'True' and count != 1000):
        raise ValueError(f'Invalid completion flag: {path}')
    sse = initial ** 2 + np.square(trace[:, 1]).sum()
    penalized = np.sqrt((sse + (1000 - count) * max(2.5, abs(trace[-1, 1])) ** 2) / 1001)
    observed = np.sqrt(sse / (count + 1))
    for name, expected in [('rmse_penalized', penalized), ('rmse_observed', observed)]:
        if not np.isclose(float(row[name]), expected, rtol=0, atol=1e-10):
            raise ValueError(f'{name} does not match raw trace: {path}')
    return runner.digest(path)


def load_campaign(root, stage):
    if stage not in evaluation.PLANS:
        raise ValueError(f'Unsupported stage: {stage}')
    protocol = root / 'results/protocol.json'
    spec = json.loads(protocol.read_text())
    prefix = 'test' if stage == 'test' else 'development'
    seeds, scales = evaluation.PLANS[stage]
    if spec[f'{prefix}_seeds'] != list(seeds) or spec[f'{prefix}_scales'] != [list(x) for x in scales]:
        raise ValueError('Evaluator cases do not match the frozen protocol.')
    if stage == 'test':
        runner.verify_manifest(root)
    out = root / 'results' / stage
    expected = {case_key(case) for case in evaluation.cases(stage)}
    hashes, rows = {}, []
    for kind, seed in runner.jobs():
        path = out / f'{kind}_{seed}.csv'
        episodes = read_csv(path)
        keys = [case_key(row) for row in episodes]
        if len(episodes) != len(expected) or len(set(keys)) != len(expected) or set(keys) != expected:
            raise ValueError(f'Incomplete, duplicate or unexpected cases: {path}')
        marker = out / f'{kind}_{seed}.run.json'
        run = json.loads(marker.read_text())
        if (run['status'] != 'completed' or run['episodes'] != len(expected)
                or run['stage'] != stage or run['kind'] != kind or int(run['seed']) != seed):
            raise ValueError(f'Incomplete run marker: {marker}')
        if run['protocol_sha256'] != runner.digest(protocol):
            raise ValueError(f'Run used a different protocol: {marker}')
        checkpoint = evaluation.checkpoint(kind, seed)
        if checkpoint is not None:
            # Resolve checkpoint identity relative to this campaign, including
            # preserved sibling historical policies.
            import os
            checkpoint = root / os.path.relpath(checkpoint, evaluation.ROOT)
            if run['checkpoint_sha256'] != runner.digest(checkpoint):
                raise ValueError(f'Run checkpoint hash mismatch: {marker}')
        for row in episodes:
            if row['kind'] != kind or int(row['train_seed']) != seed:
                raise ValueError(f'Mislabeled controller or seed: {path}')
            trace = out / 'traces' / (trace_key(row) + '.npz')
            hashes[str(trace.relative_to(root))] = verify_trace(trace, row)
        rows.extend(episodes)
    actual = {str(path.relative_to(root)) for path in (out / 'traces').glob('*.npz')}
    if actual != set(hashes):
        raise ValueError('Unaccounted or missing traces in the selected campaign.')
    return rows, hashes


def interval(values):
    mean = float(np.mean(values))
    half = float(t.ppf(.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join('---' for _ in headers) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def teacher_evidence(root):
    teacher_audit = None
    teacher_text = 'No per-case offline-teacher audit was supplied for this campaign.'
    teacher_path, corpus_path = root / 'results/teacher_cases.json', root / 'results/corpus.json'
    if teacher_path.exists() or corpus_path.exists():
        teacher_cases = json.loads(teacher_path.read_text())
        corpus = json.loads(corpus_path.read_text())
        teacher_completed = sum(case['completed'] is True for case in teacher_cases)
        teacher_mean = float(np.mean([case['rmse_penalized'] for case in teacher_cases]))
        if (len(teacher_cases) != corpus['rollout_cases'] or teacher_completed != corpus['teacher_completed']
                or not np.isclose(teacher_mean, corpus['mean_penalized_rmse'], rtol=0, atol=1e-12)):
            raise ValueError('Offline teacher aggregate does not match its retained per-case audit.')
        teacher_audit = dict(cases=len(teacher_cases), completed=teacher_completed,
                                       failed=len(teacher_cases) - teacher_completed,
                                       mean_penalized_rmse=teacher_mean,
                                       teacher_cases_sha256=runner.digest(teacher_path),
                                       corpus_metadata_sha256=runner.digest(corpus_path))
        teacher_text = (f"The offline teacher completed {teacher_completed}/{len(teacher_cases)} corpus-generation cases; "
                        f"all {len(teacher_cases) - teacher_completed} failures remain in results/teacher_cases.json. "
                        'Those teacher outcomes are training-data evidence, not deployed LANS results. '
                        'The corpus includes the recorded prefixes of failed teacher rollouts rather than claiming every teacher case succeeded.')
    return teacher_text, teacher_audit


def generate(root=ROOT, stage='test'):
    rows, traces = load_campaign(root, stage)
    out = root / 'results' / stage
    spec = {kind: [seed for name, seed in runner.jobs() if name == kind] for kind in evaluation.KINDS}
    conditions = sorted({(r['path'], r['delay'], float(r['mass'])) for r in rows})
    if len(conditions) != 14:
        raise ValueError('Expected all 14 conditions.')
    summary, cells = [], {}
    for condition in conditions:
        for kind, seeds in spec.items():
            episodes = [r for r in rows if r['kind'] == kind and
                        (r['path'], r['delay'], float(r['mass'])) == condition]
            unit = 'train_seed' if len(seeds) == 3 else 'physical_seed'
            values = [np.mean([float(r['rmse_penalized']) for r in episodes if r[unit] == seed])
                      for seed in sorted({r[unit] for r in episodes})]
            mean, low, high = interval(values)
            entry = dict(kind=kind, path=condition[0], delay=condition[1], mass=condition[2],
                         mean_penalized_rmse=mean, ci95_low=low, ci95_high=high,
                         unit=unit, unit_count=len(values), episodes=len(episodes),
                         completed=sum(r['completed'] == 'True' for r in episodes))
            summary.append(entry)
            cells[condition, kind] = entry
    ranking_summary = {}
    for label, names in GROUPS.items():
        ranks = []
        for condition in conditions:
            entry = dict(path=condition[0], delay=condition[1], mass=condition[2])
            for name in names:
                value = cells[condition, name]['mean_penalized_rmse']
                entry[name + '_rank'] = 1 + sum(cells[condition, other]['mean_penalized_rmse'] < value - 1e-12 for other in names)
                entry[name + '_rmse'] = value
            ranks.append(entry)
        write_csv(out / f'rank_{label}.csv', ranks)
        ranking_summary[label] = {name: dict(first_or_joint_first=sum(r[name + '_rank'] == 1 for r in ranks),
                                             mean_rank=float(np.mean([r[name + '_rank'] for r in ranks]))) for name in names}
    paired = []
    for other in (kind for kind in evaluation.KINDS if kind != 'corrected_lans'):
        for condition in conditions:
            differences = []
            for index, seed in enumerate(spec['corrected_lans']):
                old_seed = spec[other][index] if len(spec[other]) == 3 else -1
                a = {case_key(r): float(r['rmse_penalized']) for r in rows if r['kind'] == 'corrected_lans'
                     and int(r['train_seed']) == seed and (r['path'], r['delay'], float(r['mass'])) == condition}
                b = {case_key(r): float(r['rmse_penalized']) for r in rows if r['kind'] == other
                     and int(r['train_seed']) == old_seed and (r['path'], r['delay'], float(r['mass'])) == condition}
                differences.append(np.mean([a[key] - b[key] for key in a]))
            mean, low, high = interval(differences)
            paired.append(dict(comparison='corrected_lans_minus_' + other, path=condition[0], delay=condition[1],
                               mass=condition[2], mean_difference=mean, ci95_low=low, ci95_high=high,
                               unit_count=3, caution='Exploratory conditional training-seed means; no multiple-comparison adjustment. Historical training seeds are not matched random experiments.'))
    totals = {}
    for kind in spec:
        episodes = [r for r in rows if r['kind'] == kind]
        totals[kind] = dict(completed=sum(r['completed'] == 'True' for r in episodes), episodes=len(episodes),
                            mean_penalized_rmse=float(np.mean([float(r['rmse_penalized']) for r in episodes])))
    write_csv(out / 'episodes_all.csv', rows)
    write_csv(out / 'summary.csv', summary)
    write_csv(out / 'paired_differences.csv', paired)
    (out / 'ranking_summary.json').write_text(json.dumps(ranking_summary, indent=2))
    (out / 'aggregate.json').write_text(json.dumps(totals, indent=2))
    checkpoints = {name: runner.digest(path) for name, path in runner.expected_checkpoints(root).items()}
    # Hash stable training/evaluation logs, excluding this report's own live stdout log.
    log_paths = set((root / 'results').glob('train_*.log'))
    for kind, seed in runner.jobs():
        for path in [out / f'{kind}_{seed}.log', root / 'results' / f'{stage}_{kind}_{seed}.log']:
            if path.is_file():
                log_paths.add(path)
    logs = {str(path.relative_to(root)): runner.digest(path) for path in sorted(log_paths)}
    audit = dict(status='passed', stage=stage, episodes=len(rows), raw_traces=len(traces), conditions=len(conditions),
                 completed=sum(r['completed'] == 'True' for r in rows),
                 optimizer_failures=sum(int(r['optimizer_failures']) for r in rows),
                 protocol_sha256=runner.digest(root / 'results/protocol.json'),
                 checkpoint_sha256=checkpoints, log_sha256=logs, trace_sha256=traces,
                 checks=['all expected controller/case/seed identities', 'no missing or duplicate episodes',
                         'run markers and checkpoint/protocol hashes', 'all raw metrics recomputed',
                         'steering bounds', 'no unaccounted traces'],
                 limits='Finite-case scalar simulator evidence; not universal superiority, statistical dominance, physical validation, or proof of additional PPO learning.')
    teacher_text, teacher_audit = teacher_evidence(root)
    if teacher_audit is not None:
        audit['offline_teacher'] = teacher_audit
    (out / 'audit.json').write_text(json.dumps(audit, indent=2))
    lans = totals['corrected_lans']
    previous = totals['previous_lans']
    deployment = (f"On these matched current cases, experimental LANS completed {lans['completed']}/{lans['episodes']} "
                  f"with aggregate penalized RMSE {lans['mean_penalized_rmse']:.6f} m; "
                  f"previous LANS completed {previous['completed']}/{previous['episodes']} "
                  f"with {previous['mean_penalized_rmse']:.6f} m.")
    if lans['completed'] / lans['episodes'] < previous['completed'] / previous['episodes']:
        deployment += ' Completion regressed against the previous checkpoint on matched cases.'
    if lans['mean_penalized_rmse'] > previous['mean_penalized_rmse']:
        deployment += ' Aggregate penalized RMSE also regressed against the previous checkpoint.'

    failed_lans = [row for row in rows if row['kind'] == 'corrected_lans' and row['completed'] == 'False']
    failure_rows = []
    for row in failed_lans:
        with np.load(out / 'traces' / (trace_key(row) + '.npz'), allow_pickle=False) as data:
            trace = data['trace']
        reason = ('heading limit' if abs(trace[-1, 2]) > .5 else
                  ('lateral limit' if abs(trace[-1, 1]) > 2.5 else 'early termination'))
        completed_peers = [str(other['train_seed']) for other in rows if other['kind'] == 'corrected_lans'
                           and case_key(other) == case_key(row) and other['completed'] == 'True']
        failure_rows.append([row['train_seed'], row['physical_seed'],
                             f"{row['path']}/{row['delay']}/{float(row['mass']):g}",
                             f"{float(row['amplitude']):g}/{float(row['time_scale']):g}", row['steps'],
                             reason, f'{trace[-1, 2]:.6f}',
                             f'{np.max(abs(trace[:, 3] - trace[:, 4])):.6f}',
                             int(np.max(trace[:, 8])), ','.join(completed_peers) or 'none'])
    if failure_rows:
        failure_text = markdown_table(
            ['Train seed', 'Physical seed', 'Condition', 'Amplitude/time scale', 'Steps',
             'Exit', 'Final heading (rad)', 'Max command/applied gap (rad)', 'Max queue', 'Completed peer seeds'],
            failure_rows)
        failure_text += ('\n\nThe recorded command and applied-command gap reflects the full-command FIFO delay: '
                         'a newly saturated command can coexist with an older command still being applied. '
                         'A finite-state heading-limit exit is retained as a control-robustness failure, not a nonfinite-state event. '
                         'A simulator defect has not been identified by this trace audit. '
                         'The seed differences are reported rather than selecting only the successful policy.')
    else:
        failure_text = 'No corrected-LANS episode failed in this selected campaign.'
    table = []
    for condition in conditions:
        table.append([f'{condition[0]}/{condition[1]}/{condition[2]:g}'] +
                     [f"{cells[condition, kind]['mean_penalized_rmse']:.6f}" for kind in
                      ('corrected_lans', 'corrected_ppo', 'new_bc_lans', 'previous_lans', 'previous_ppo', 'matched_mpc')])
    outcome = ('The target of first place in every declared comparison is not established.'
               if ranking_summary['all_references']['corrected_lans']['first_or_joint_first'] != 14
               else 'LANS has the lowest or joint-lowest numerical mean in all 14 declared conditions; this does not establish statistical or universal superiority.')
    review = f'''# Corrected LANS training: {stage} review

LANS completed {lans['completed']}/{lans['episodes']} episodes. {outcome}

This is a separate study; historical results and policies were retained. All {len(rows)} current episodes and {len(traces)} raw traces were checked, including failures. Values below are initial-inclusive failure-penalized RMSE in metres, lower is better.

{markdown_table(['Comparison pool', 'LANS first/joint first of 14', 'Mean numerical rank'], [[label, values['corrected_lans']['first_or_joint_first'], f"{values['corrected_lans']['mean_rank']:.3f}"] for label, values in ranking_summary.items()])}

The historical-tracking pool means the previous tracking PPO plus the original LQR and MPC; it does not relabel the historical policy as a newly trained comparator. Equal-pretraining uses the new corrected PPO; objective-matched additionally replaces the original MPC with matched MPC. All-references includes both BC-only policies and both historical learned families. No per-condition or best-seed switching is used.

{markdown_table(['Condition', 'LANS', 'New PPO', 'BC LANS', 'Previous LANS', 'Previous PPO', 'Matched MPC'], table)}

## Retained LANS failures

{failure_text}

{markdown_table(['Controller', 'Completed episodes', 'Aggregate penalized RMSE (m)'], [[kind, str(value['completed']) + '/' + str(value['episodes']), f"{value['mean_penalized_rmse']:.6f}"] for kind, value in totals.items()])}

## Issues corrected

The inherited 200704-step training curriculum supplied only zero-delay and 0/10ms episodes during the actual 100352-step PPO budget; its mixed and high-delay phases were never reached. Training now covers the declared delay distribution within its actual budget. The output-layer projection previously occurred only before rollouts and after training, allowing intermediate PPO updates outside the declared parameter box; the corrected update hook enforces the constraint after every optimizer step. Both fixes apply to LANS and the equally pretrained PPO.

The preview adapter previously subtracted present-state terms after float32 rounding, amplifying cancellation in derivative-sensitive features. The new adapter computes the algebraically equivalent eleven observations before the final float32 conversion. The offline teacher was selected on the declared development grid; this changes training targets and is not a correction to the physical plant or proof that the deployed controller is an analytic teacher.

Evaluation now rejects unknown controller labels, preflights complete output sets, preserves partial runs, includes matched MPC automatically, and records checkpoint/protocol hashes. Reports include every declared comparator and recompute metrics from raw traces. Historical reporting was also repaired to handle an unrestricted PPO validation winner instead of silently omitting it.

## Checkpoint recommendation

This campaign is experimental. Preserve the stable previous checkpoint; the new policies must not replace it merely because they are newer. {deployment} Any future replacement needs an explicit acceptance decision based on the matched comparisons, including failures. All final outcomes are retained; no parameters are retuned after this final test.

## Teacher and training-loss disclosure

{teacher_text}

The training log field named physical_steering_mse is an action-scale-weighted output-loss proxy. For LANS it measures the residual prediction error before the latency gain and final steering saturation; it is not measured post-gain physical steering MSE, trajectory RMSE, or deployment performance. Its existing field name is retained to preserve the frozen training records.

## Interpretation and limits

The new BC-only checkpoints are explicit ablations. Differences after constrained PPO updates do not by themselves establish an additional reinforcement-learning benefit. The three training-seed confidence intervals in summary.csv and paired_differences.csv are exploratory and conditional on these physical cases, without multiple-comparison correction. Historical training seeds are not matched random training experiments.

The new policies keep the same 11 causal inputs, 64x64 actor, scalar residual, LQR, latency gain, steering saturation and dynamic bicycle plant. The nominal teacher cannot directly observe unknown vehicle mass. Policy preview is four samples, whereas matched MPC uses twenty. The test is a linear small-angle simulation with no tire, hardware, energy or global-stability validation. Current audit: {audit['optimizer_failures']} recorded optimizer failures; passed metric-integrity checks do not turn unsuccessful episodes into successes.
'''
    (out / 'REVIEW.md').write_text(review)
    if stage == 'test':
        (root / 'REVIEW.md').write_text(review)
    print(json.dumps(dict(stage=stage, LANS=lans, ranking=ranking_summary['all_references']['corrected_lans'], episodes=len(rows)), indent=2))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=evaluation.PLANS, default='test')
    args = parser.parse_args()
    generate(stage=args.stage)
